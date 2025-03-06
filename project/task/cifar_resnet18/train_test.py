"""CIFAR training and testing functions, local and federated."""

from copy import deepcopy
import logging
from logging import ERROR, WARNING
import pickle
from flwr.common import log

from collections.abc import Sized, Callable
from pathlib import Path
from typing import cast

import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from torch.nn.utils import prune

from project.fed.utils.utils import generic_get_parameters, generic_set_parameters
from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)
from project.task.cifar_resnet18.models import (
    get_parameters_to_prune,
    set_spectral_global_exponent,
)

bounds = [(0, 40), (40, 70), (70, 100)]
sparsities: list[float] = []


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member acces and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float
    final_learning_rate: float  # ? to remove
    # tot_rounds: int
    curr_round: int

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    config: TrainConfig = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=0.001,
    )

    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()
            optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss / len(
            cast(Sized, trainloader.dataset)
        ),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


def fixed_train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    config: TrainConfig = TrainConfig(**_config)
    del _config

    # FLASH
    mask = []
    for param in net.parameters():
        mask.append((param != 0).float())

    net.to(config.device)
    net.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=0.001,
    )

    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()

            # FLASH
            # apply the mask to the gradients
            with torch.no_grad():
                for param, m in zip(net.parameters(), mask, strict=True):
                    param.grad *= m.to(config.device)

            optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss / len(
            cast(Sized, trainloader.dataset)
        ),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


def get_fixed_train_and_prune(
    alpha: float = 1.0, amount: float = 0.0, pruning_method: str = "l1"
) -> Callable[[nn.Module, DataLoader, dict, Path], tuple[int, dict]]:
    """Return the training loop with one step pruning at the end.

    Think about moving 'amount' to the config file
    """
    if pruning_method == "base":  # ? not working
        pruning_method = prune.BasePruningMethod
    elif pruning_method == "l1":
        pruning_method = prune.L1Unstructured
    else:
        log(ERROR, f"Pruning method {pruning_method} not recognised, using base")

    def train_and_prune(
        net: nn.Module,
        trainloader: DataLoader,
        _config: dict,
        _working_dir: Path,
    ) -> tuple[int, dict]:
        """Training and pruning process."""
        # log(logging.DEBUG, f"[CLient-{_config['cid']}] Start training")
        sparsity = amount

        # HETERO EXPS
        # <---
        if int(_config["curr_round"]) > 1 and len(sparsities) > 0:
            for i, bound in enumerate(bounds):
                if int(_config["cid"]) >= bound[0] and int(_config["cid"]) < bound[1]:
                    sparsity = sparsities[i]
                    # print(f"Client {_config['cid']}, amount: {sparsity}")
                    break

            # Check for existing mask at the specified sparsity level
            mask_path = _working_dir / "masks" / f"mask_sparsity_{sparsity:.2f}.pkl"
            # print(f"mask_path: {mask_path}")
            if sparsity != 0 and mask_path.exists():
                log(
                    logging.INFO,
                    f"Loading pre-existing mask with sparsity {sparsity:.2f}",
                )
                try:
                    # Load masks
                    with open(mask_path, "rb") as f:
                        loaded_masks = pickle.load(f)
                    # Get current parameters
                    current_params = generic_get_parameters(net)
                    # Apply masks
                    masked_params = [
                        param * mask
                        for param, mask in zip(
                            current_params, loaded_masks, strict=True
                        )
                    ]
                    # Update network
                    generic_set_parameters(net, masked_params)

                except Exception as e:
                    log(
                        WARNING,
                        f"Error applying mask: {e!s}. Proceeding with regular"
                        " training.",
                    )
        # --->

        # FLASH
        if _config["curr_round"] == 1 and _config["warmup"] > 0:
            # temp_net = deepcopy(net)
            log(logging.DEBUG, "First round, warmup training")
            _config["epochs"] = _config["warmup"]

        # train the network, with the current parameter
        metrics = fixed_train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
        )

        if sparsity != 0:
            parameters_to_prune = get_parameters_to_prune(net)

            prune.global_unstructured(
                parameters=[
                    (module, tensor_name)
                    for module, tensor_name, _ in parameters_to_prune
                ],
                pruning_method=pruning_method,
                amount=sparsity,
            )
            for module, name, _ in parameters_to_prune:
                prune.remove(module, name)

        torch.cuda.empty_cache()
        metrics[1]["sparsity"] = sparsity

        return metrics

    return train_and_prune


def get_train_and_prune(
    alpha: float = 1.0,
    amount: float = 0.0,
    pruning_method: str = "l1",
) -> Callable[[nn.Module, DataLoader, dict, Path], tuple[int, dict]]:
    """Return the training loop with one step pruning at the end.

    Think about moving 'amount' to the config file
    """
    if pruning_method == "base":  # ? not working
        pruning_method = prune.BasePruningMethod
    elif pruning_method == "l1":
        pruning_method = prune.L1Unstructured
    else:
        log(ERROR, f"Pruning method {pruning_method} not recognised, using base")

    def train_and_prune(
        net: nn.Module,
        trainloader: DataLoader,
        _config: dict,
        _working_dir: Path,
    ) -> tuple[int, dict]:
        """Training and pruning process."""
        log(logging.DEBUG, "Start training")

        sparsity = amount

        # HETERO EXPS
        # <---
        # bounds = [(0, 40), (40, 70), (70, 100)]
        # sparsities = [ ]

        # print(f"Client {_config}")

        # assign the sparsity based on the cid
        if int(_config["curr_round"]) > 1 and len(sparsities) > 1:
            for i, bound in enumerate(bounds):
                if int(_config["cid"]) >= bound[0] and int(_config["cid"]) < bound[1]:
                    sparsity = sparsities[i]
                    # print cid and amount
                    # print(f"Client {_config['cid']}, amount: {sparsity}")
                    break
            # sparsify the model
            if sparsity != 0:
                parameters_to_prune = get_parameters_to_prune(net)

                prune.global_unstructured(
                    parameters=[
                        (module, tensor_name)
                        for module, tensor_name, _ in parameters_to_prune
                    ],
                    pruning_method=prune.L1Unstructured,
                    amount=sparsity,
                )

                for module, name, _ in parameters_to_prune:
                    prune.remove(module, name)

                # update the model
                # generic_set_parameters(net, generic_get_parameters(temp_net))
                generic_set_parameters(net, generic_get_parameters(net))

                torch.cuda.empty_cache()
        # --->

        # train the network, with the current parameter
        metrics = train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
        )

        # dense_model = deepcopy(net)

        if sparsity > 0:
            """
            The net must be pruned:
            - at the first round if we are using powerprop
            - every round if we are not using powerprop (alpha=1.0)
            """
            parameters_to_prune = get_parameters_to_prune(net)

            prune.global_unstructured(
                parameters=[
                    (module, tensor_name)
                    for module, tensor_name, _ in parameters_to_prune
                ],
                pruning_method=pruning_method,
                amount=sparsity,
            )
            for module, name, _ in parameters_to_prune:
                prune.remove(module, name)

            # Prevent layer collapse
            # generic_set_parameters(net, generic_get_parameters(net))
            # sparsity_after_pruning = get_nonzeros(net)
            # net = prevent_layer_collapse(dense_model, net, 1-sparsity)
            # sparsity_after_prevent = get_nonzeros(net)
            # if sparsity_after_prevent != sparsity_after_pruning:
            #     log(
            #         logging.INFO,
            #         f"Pruned model to sparsity {sparsity_after_pruning:.2f}, "
            #         f"prevented layer collapse to sparsity"
            #         f"{sparsity_after_prevent:.2f}",
            #     )

        torch.cuda.empty_cache()
        metrics[1]["sparsity"] = sparsity

        return metrics

    return train_and_prune


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member acces and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.

    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    config: TestConfig = TestConfig(**_config)
    del _config

    # print("EVAL conf: ", config)

    # sparsities = [ 0.0, 0.9, 0.95, 0.99, 0.995, 0.999 ]
    # sparsities = [ ]
    # sparsities = [ 0.0 ]

    sparse_accuracy = {}
    sparse_loss = {}

    net.to(config.device)
    net.eval()

    # SPECTRAL EXPONENT
    # <---
    avg_exponent = set_spectral_global_exponent(net, False)
    log(logging.INFO, f"[test] Average spectral exponent: {avg_exponent}")
    # --->

    # get the global model

    # Evaluate the dense model first
    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = (
                images.to(
                    config.device,
                ),
                labels.to(config.device),
            )
            outputs = net(images)
            per_sample_loss += criterion(
                outputs,
                labels,
            ).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    sparse_accuracy["test_accuracy"] = float(correct) / len(
        cast(Sized, testloader.dataset)
    )
    sparse_loss["loss"] = per_sample_loss / len(cast(Sized, testloader.dataset))
    # print(f"Sparsity: 0, Accuracy: {correct / len(cast(Sized, testloader.dataset))}")

    # Evaluate the sparse models
    for sparsity in sparsities:
        if sparsity == 0:
            continue

        sparse_model = deepcopy(net)
        parameters_to_prune = get_parameters_to_prune(sparse_model)

        prune.global_unstructured(
            parameters=[
                (module, tensor_name) for module, tensor_name, _ in parameters_to_prune
            ],
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )

        for module, name, _ in parameters_to_prune:
            prune.remove(module, name)

        generic_set_parameters(sparse_model, generic_get_parameters(sparse_model))
        torch.cuda.empty_cache()

        correct, per_sample_loss = 0, 0.0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = (
                    images.to(
                        config.device,
                    ),
                    labels.to(config.device),
                )
                outputs = sparse_model(images)
                per_sample_loss += criterion(
                    outputs,
                    labels,
                ).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        torch.cuda.empty_cache()

        sparse_accuracy[f"test_accuracy_{sparsity}"] = float(correct) / len(
            cast(Sized, testloader.dataset)
        )
        sparse_accuracy[f"centralized_test_accuracy_{sparsity}"] = float(correct) / len(
            cast(Sized, testloader.dataset)
        )
        sparse_loss[f"loss_{sparsity}"] = per_sample_loss / len(
            cast(Sized, testloader.dataset)
        )

    # criterion = nn.CrossEntropyLoss()
    # correct, per_sample_loss = 0, 0.0

    # with torch.no_grad():
    #     for images, labels in testloader:
    #         images, labels = (
    #             images.to(
    #                 config.device,
    #             ),
    #             labels.to(config.device),
    #         )
    #         outputs = net(images)
    #         per_sample_loss += criterion(
    #             outputs,
    #             labels,
    #         ).item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         correct += (predicted == labels).sum().item()

    torch.cuda.empty_cache()

    # print("sparse_accuracy", sparse_accuracy)

    sparse_accuracy["exponent"] = avg_exponent
    # print(f"Average spectral exponent: {avg_exponent}")

    return (
        # per_sample_loss / len(cast(Sized, testloader.dataset)),
        sparse_loss["loss"],  # dense model loss
        len(cast(Sized, testloader.dataset)),
        sparse_accuracy,
        # {
        #     "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
        # },
    )


def test_hetero_flash(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set with different sparsity masks."""
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError("Testloader can't be 0, exiting...")

    config: TestConfig = TestConfig(**_config)
    del _config

    # print("EVAL conf: ", config)

    sparse_accuracy = {}
    sparse_loss = {}

    net.to(config.device)
    net.eval()
    criterion = nn.CrossEntropyLoss()

    # Evaluate the dense model first
    correct, per_sample_loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            outputs = net(images)
            per_sample_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    sparse_accuracy["test_accuracy"] = float(correct) / len(
        cast(Sized, testloader.dataset)
    )
    sparse_loss["loss"] = per_sample_loss / len(cast(Sized, testloader.dataset))
    # print(f"Sparsity: 0, Accuracy: {correct / len(cast(Sized, testloader.dataset))}")

    # Evaluate with different sparsity masks
    for sparsity in sparsities:
        if sparsity == 0:
            continue

        # Get the mask path for current sparsity
        mask_path = _working_dir / "masks" / f"mask_sparsity_{sparsity:.2f}.pkl"

        if not mask_path.exists():
            # print(f"Warning: Mask file not found for sparsity {sparsity}")
            continue

        try:
            # Create a copy of the model for this sparsity level
            sparse_model = deepcopy(net)
            sparse_model.to(config.device)

            # Load and apply mask
            with open(mask_path, "rb") as f:
                masks = pickle.load(f)

            # Get current parameters and apply mask
            current_params = generic_get_parameters(sparse_model)
            masked_params = [
                param * mask for param, mask in zip(current_params, masks, strict=True)
            ]

            # Update model with masked parameters
            generic_set_parameters(sparse_model, masked_params)

            # Evaluate masked model
            correct, per_sample_loss = 0, 0.0
            with torch.no_grad():
                for images, labels in testloader:
                    images = images.to(config.device)
                    labels = labels.to(config.device)
                    outputs = sparse_model(images)
                    per_sample_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()

            accuracy = float(correct) / len(cast(Sized, testloader.dataset))
            sparse_accuracy[f"test_accuracy_{sparsity}"] = accuracy
            sparse_accuracy[f"centralized_test_accuracy_{sparsity}"] = accuracy
            sparse_loss[f"loss_{sparsity}"] = per_sample_loss / len(
                cast(Sized, testloader.dataset)
            )

            # print(f"Sparsity: {sparsity}, Accuracy: {accuracy}")

        except Exception:
            # print(f"Error evaluating sparsity {sparsity}: {e!s}")
            continue

        finally:
            torch.cuda.empty_cache()

    # print("sparse_accuracy", sparse_accuracy)

    return (
        sparse_loss["loss"],  # dense model loss
        len(cast(Sized, testloader.dataset)),
        sparse_accuracy,
    )


# Use defaults as they are completely determined
# by the other functions defined in mnist_classification
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
