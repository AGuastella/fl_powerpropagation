"""Training and testing functions for TinyViT on CUB-200-2011 in federated setting."""

import logging
from logging import ERROR
from collections.abc import Sized
from pathlib import Path
from typing import cast
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from project.fed.utils.utils import generic_get_parameters, generic_set_parameters
from project.task.cub_vit.models import get_parameters_to_prune
from pydantic import BaseModel
from flwr.common import log


from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)

from torch.nn.utils import prune
import wandb


# class TrainConfig(BaseModel):
#     """Training configuration."""
#     device: torch.device
#     epochs: int = 1  # Single epoch per round
#     learning_rate: float = 1e-4
#     weight_decay: float = 0.001   # Reduced weight decay for small datasets
#     curr_round: int = 0          # Track FL round for logging

#     class Config:
#         """Allow torch.device type."""
#         arbitrary_types_allowed = True


class TrainConfig(BaseModel):
    """Training configuration."""

    cid: int
    device: torch.device
    epochs: int
    learning_rate: float = 1e-4  # Lower learning rate for transformer
    momentum: float = 0.9  # SGD momentum
    weight_decay: float = 0.001  # Weight decay for regularization
    warmup_epochs: int = 0  # Linear warmup period
    min_learning_rate: float = 1e-6
    nesterov: bool = False  # Nesterov momentum
    curr_round: int
    gradient_clip_val: float = 1.0  # Gradient clipping for transformer stability

    class Config:
        """Allow torch.device type."""

        arbitrary_types_allowed = True


def validate_data_batch(
    data: torch.Tensor, target: torch.Tensor, num_classes: int
) -> None:
    """Validate input data and target tensors."""
    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError(
            f"Target labels must be in range [0, {num_classes - 1}], "
            f"but got range [{target.min()}, {target.max()}]"
        )

    # if data.dim() != 4:
    #     raise ValueError(f"Expected 4D input tensor, got {data.dim()}D")

    if not target.dtype == torch.long:
        raise ValueError(f"Expected target dtype torch.long, got {target.dtype}")


# log
def train_local_log(
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    alpha: float = 1.0,
    amount: float = 0.0,
    use_mask: bool = False,
) -> tuple[int, dict]:
    """Train the model with W&B batch-level monitoring.

    Parameters
    ----------
    net : nn.Module
        The neural network to train
    trainloader : DataLoader
        The DataLoader containing the training data
    _config : dict
        Training configuration dictionary
    _working_dir : Path
        Working directory for saving checkpoints
    use_mask : bool
        Whether to use fixed mask training

    Returns
    -------
    tuple[int, dict]
        Number of samples processed and metrics dictionary
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError("Trainloader cannot be empty")

    config = TrainConfig(**_config)
    wandb_config = {
        "train_config": _config,
        "batch_size": trainloader.batch_size,
        "alpha": alpha,
        "amount": amount,
    }
    del _config

    # Initialize W&B run for this client's training
    run = wandb.init(
        project="fl-client-training",
        group=f"round_{config.curr_round}",
        name=f"client_{config.cid}_round_{config.curr_round}",
        reinit=True,  # Allow multiple runs in same process
        config=wandb_config,
    )

    log(logging.INFO, f"Starting training with{'out' if not use_mask else ''} mask")

    # Create masks if needed
    masks = []
    if use_mask:
        for name, param in net.named_parameters():
            if "weight" in name:
                mask = (param != 0).float()
                masks.append(mask)
            else:
                masks.append(None)

    net.to(config.device)
    net.train()

    # Get number of classes from model
    # num_classes = net.num_classes if hasattr(net, "num_classes") else 200

    optimizer = AdamW(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.min_learning_rate,
    )

    criterion = nn.CrossEntropyLoss()
    total_samples = len(cast(Sized, trainloader.dataset))

    final_epoch_loss = 0.0
    num_correct = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        num_correct = 0

        for batch_idx, (data, target) in enumerate(trainloader):
            target = target.long()
            data, target = data.to(config.device), target.to(config.device)

            # Warmup learning rate
            if epoch < config.warmup_epochs:
                lr_scale = min(
                    1.0,
                    float(batch_idx + epoch * len(trainloader))
                    / (config.warmup_epochs * len(trainloader)),
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * config.learning_rate

            optimizer.zero_grad()
            output = net(data)

            loss = criterion(output, target)
            loss.backward()

            if use_mask:
                # Apply masks to gradients before optimization step
                with torch.no_grad():
                    for param, mask in zip(net.parameters(), masks, strict=True):
                        if mask is not None and param.grad is not None:
                            param.grad *= mask.to(config.device)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            optimizer.step()

            if use_mask:
                # Ensure masked weights remain zero after optimization
                with torch.no_grad():
                    for param, mask in zip(net.parameters(), masks, strict=True):
                        if mask is not None:
                            param.data *= mask.to(config.device)

            # Update metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_correct = (output.max(1)[1] == target).sum().item()
            num_correct += batch_correct
            batch_accuracy = batch_correct / len(target)

            # Log batch metrics to W&B
            wandb.log({
                "batch": batch_idx + epoch * len(trainloader),
                "batch_loss": batch_loss,
                "batch_accuracy": batch_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            })

            if batch_idx % 10 == 0:
                log(
                    logging.INFO,
                    f"Epoch {epoch}/{config.epochs} "
                    f"[{batch_idx * len(data)}/{total_samples} "
                    f"({100. * batch_idx / len(trainloader):.0f}%)] "
                    f"Loss: {batch_loss:.6f}",
                )

        # Update learning rate
        if epoch >= config.warmup_epochs:
            scheduler.step()

        final_epoch_loss = epoch_loss / len(trainloader)
        epoch_accuracy = num_correct / total_samples

        # Log epoch metrics to W&B
        wandb.log({
            "epoch_loss": final_epoch_loss,
            "epoch_accuracy": epoch_accuracy,
            "epoch": epoch,
        })

        log(
            logging.INFO,
            f"Epoch {epoch}: Loss = {final_epoch_loss:.4f}, "
            f"Accuracy = {100. * epoch_accuracy:.2f}%",
        )

    if use_mask:
        # Final application of masks
        with torch.no_grad():
            for param, mask in zip(net.parameters(), masks, strict=True):
                if mask is not None:
                    param.data *= mask.to(config.device)

    # Close the W&B run
    if run is not None:
        run.finish()

    return total_samples, {
        "train_loss": final_epoch_loss,
        "train_accuracy": float(num_correct) / total_samples,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


# AdamW
def train(
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    alpha: float = 1.0,
    amount: float = 0.0,
    use_mask: bool = False,
) -> tuple[int, dict]:
    """Train the TinyViT model with optional fixed mask training.

    Parameters
    ----------
    net : nn.Module
        The neural network to train
    trainloader : DataLoader
        The DataLoader containing the training data
    _config : dict
        Training configuration dictionary
    _working_dir : Path
        Working directory for saving checkpoints
    use_mask : bool
        Whether to use fixed mask training

    Returns
    -------
    tuple[int, dict]
        Number of samples processed and metrics dictionary
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError("Trainloader cannot be empty")

    config = TrainConfig(**_config)
    del _config

    log(logging.INFO, f"Starting training with{'out' if not use_mask else ''} mask")

    # Create masks if needed
    masks = []
    if use_mask:
        for name, param in net.named_parameters():
            if "weight" in name:
                mask = (param != 0).float()
                masks.append(mask)
                # log(logging.INFO,
                #     f"Created mask for {name}, "
                #     f"sparsity: {100 * (1 - torch.sum(mask)/mask.numel()):.2f}%")
            else:
                masks.append(None)

    net.to(config.device)
    net.train()

    # Get number of classes from model
    num_classes = net.num_classes if hasattr(net, "num_classes") else 200

    # Initialize optimizer with weight decay
    optimizer = AdamW(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Cosine learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.min_learning_rate,
    )

    criterion = nn.CrossEntropyLoss()
    total_samples = len(cast(Sized, trainloader.dataset))

    final_epoch_loss = 0.0
    num_correct = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        num_correct = 0

        for batch_idx, (data, target) in enumerate(trainloader):
            target = target.long()
            data, target = data.to(config.device), target.to(config.device)

            try:
                validate_data_batch(data, target, num_classes)
            except ValueError as e:
                log(logging.ERROR, f"Data validation failed: {e!s}")
                raise

            # Warmup learning rate
            if epoch < config.warmup_epochs:
                lr_scale = min(
                    1.0,
                    float(batch_idx + epoch * len(trainloader))
                    / (config.warmup_epochs * len(trainloader)),
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * config.learning_rate

            optimizer.zero_grad()
            output = net(data)

            loss = criterion(output, target)
            loss.backward()

            if use_mask:
                # Apply masks to gradients before optimization step
                with torch.no_grad():
                    for param, mask in zip(net.parameters(), masks, strict=True):
                        if mask is not None and param.grad is not None:
                            param.grad *= mask.to(config.device)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            optimizer.step()

            if use_mask:
                # Ensure masked weights remain zero after optimization
                with torch.no_grad():
                    for param, mask in zip(net.parameters(), masks, strict=True):
                        if mask is not None:
                            param.data *= mask.to(config.device)

            # Update metrics
            epoch_loss += loss.item()
            num_correct += (output.max(1)[1] == target).sum().item()

            if batch_idx % 10 == 0:
                log(
                    logging.INFO,
                    f"Epoch {epoch}/{config.epochs} "
                    f"[{batch_idx * len(data)}/{total_samples} "
                    f"({100. * batch_idx / len(trainloader):.0f}%)] "
                    f"Loss: {loss.item():.6f}",
                )

        # Update learning rate
        if epoch >= config.warmup_epochs:
            scheduler.step()

        final_epoch_loss = epoch_loss / len(trainloader)
        log(
            logging.INFO,
            f"Epoch {epoch}: Loss = {final_epoch_loss:.4f}, "
            f"Accuracy = {100. * num_correct / total_samples:.2f}%",
        )

    if use_mask:
        # Final application of masks
        with torch.no_grad():
            for param, mask in zip(net.parameters(), masks, strict=True):
                if mask is not None:
                    param.data *= mask.to(config.device)

    return total_samples, {
        "train_loss": final_epoch_loss,
        "train_accuracy": float(num_correct) / total_samples,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


# SGD
def train_sgd(
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    alpha: float = 1.0,
    amount: float = 0.0,
    use_mask: bool = False,
) -> tuple[int, dict]:
    """Train the TinyViT model using simplified SGD approach.

    Follows ResNet training pattern while maintaining transformer stability.

    Parameters
    ----------
    net : nn.Module
        The neural network to train
    trainloader : DataLoader
        The DataLoader containing the training data
    _config : dict
        Training configuration dictionary
    _working_dir : Path
        Working directory for saving checkpoints
    use_mask : bool
        Whether to use fixed mask training
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError("Trainloader cannot be empty")

    config = TrainConfig(**_config)
    del _config

    log(logging.INFO, f"Starting SGD training with{'out' if not use_mask else ''} mask")
    log(logging.INFO, f"Config: {config}")

    # Create masks if needed
    masks = []
    if use_mask:
        for name, param in net.named_parameters():
            if "weight" in name:
                mask = (param != 0).float()
                masks.append(mask)
            else:
                masks.append(None)

    net.to(config.device)
    net.train()

    # Get number of classes from model
    # num_classes = net.num_classes if hasattr(net, "num_classes") else 200

    # Initialize SGD optimizer - matching ResNet configuration
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        nesterov=config.nesterov,
        weight_decay=config.weight_decay,  # Same as ResNet
    )

    criterion = nn.CrossEntropyLoss()
    total_samples = len(cast(Sized, trainloader.dataset))

    final_epoch_loss = 0.0
    num_correct = 0

    # Training loop following ResNet pattern
    for _ in range(config.epochs):
        final_epoch_loss = 0.0
        num_correct = 0

        for data, target in trainloader:
            target = target.long()
            data, target = data.to(config.device), target.to(config.device)

            optimizer.zero_grad()
            output = net(data)

            loss = criterion(output, target)
            loss.backward()

            if use_mask:
                # Apply masks to gradients before optimization step
                with torch.no_grad():
                    for param, mask in zip(net.parameters(), masks, strict=True):
                        if mask is not None and param.grad is not None:
                            param.grad *= mask.to(config.device)

            # Minimal gradient clipping for transformer stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip_val)

            optimizer.step()

            if use_mask:
                # Ensure masked weights remain zero after optimization
                with torch.no_grad():
                    for param, mask in zip(net.parameters(), masks, strict=True):
                        if mask is not None:
                            param.data *= mask.to(config.device)

            # Update metrics
            final_epoch_loss += loss.item()
            num_correct += (output.max(1)[1] == target).sum().item()

        # Prune the model
        # if amount > 0:
        #     parameters_to_prune = get_parameters_to_prune(net)
        #     prune.global_unstructured(
        #         parameters=[
        #             (module, tensor_name)
        #             for module, tensor_name, _ in parameters_to_prune
        #         ],
        #         pruning_method=prune.L1Unstructured,
        #         amount=amount,
        #     )
        #     for module, name, _ in parameters_to_prune:
        #         prune.remove(module, name)
        #     generic_set_parameters(net, generic_get_parameters(net))

    if use_mask:
        # Final application of masks
        with torch.no_grad():
            for param, mask in zip(net.parameters(), masks, strict=True):
                if mask is not None:
                    param.data *= mask.to(config.device)

    torch.cuda.empty_cache()

    return total_samples, {
        "train_loss": final_epoch_loss / len(cast(Sized, trainloader.dataset)),
        "train_accuracy": float(num_correct) / total_samples,
    }


def get_train_and_prune(
    alpha: float = 1.0,
    amount: float = 0.0,
    pruning_method: str = "l1",
    use_mask: bool = False,
) -> Callable[[nn.Module, DataLoader, dict, Path], tuple[int, dict]]:
    """Return the training loop with optional mask training and pruning.

    Parameters
    ----------
    alpha : float
        Not used in this implementation but kept for interface compatibility
    amount : float
        Pruning amount (0 to 1)
    pruning_method : str
        Type of pruning to use ("l1" or "base")
    use_mask : bool
        Whether to use masked training

    Returns
    -------
    Callable
        Training function with the specified configuration
    """
    if pruning_method == "base":
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
        # log(logging.DEBUG, "Start training")

        # FLASH
        if _config["curr_round"] == 1 and _config["warmup"] > 0:
            # temp_net = deepcopy(net)
            log(logging.DEBUG, "First round, warmup training")
            warmup_config = _config.copy()
            warmup_config["epochs"] = _config["warmup"]

            train(
                net=net,
                trainloader=trainloader,
                _config=warmup_config,
                _working_dir=_working_dir,
                use_mask=use_mask,
                amount=amount,
                alpha=alpha,
            )
            # Prune the model
            parameters_to_prune = get_parameters_to_prune(net)
            prune.global_unstructured(
                parameters=[
                    (module, tensor_name)
                    for module, tensor_name, _ in parameters_to_prune
                ],
                pruning_method=pruning_method,
                amount=amount,
            )
            for module, name, _ in parameters_to_prune:
                prune.remove(module, name)
            # Set back the parameter in the model since they are in the wrong order
            generic_set_parameters(net, generic_get_parameters(net))

        # Train the network
        metrics = train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
            use_mask=use_mask,
            amount=amount,
            alpha=alpha,
        )

        # Apply pruning if needed
        if amount > 0:
            parameters_to_prune = get_parameters_to_prune(net)

            prune.global_unstructured(
                parameters=[
                    (module, tensor_name)
                    for module, tensor_name, _ in parameters_to_prune
                ],
                pruning_method=pruning_method,
                amount=amount,
            )
            for module, name, _ in parameters_to_prune:
                prune.remove(module, name)

        torch.cuda.empty_cache()
        metrics[1]["sparsity"] = amount

        return metrics

    return train_and_prune


class TestConfig(BaseModel):
    """Testing configuration."""

    device: torch.device

    class Config:
        """Allow torch.device type."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[float, int, dict]:
    """Evaluate the TinyViT model on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to evaluate
    testloader : DataLoader
        The DataLoader containing the test data
    _config : dict
        Testing configuration dictionary
    _working_dir : Path
        Working directory for saving results

    Returns
    -------
    tuple[float, int, dict]
        Loss, number of samples, and metrics dictionary
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError("Testloader cannot be empty")

    config = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_correct = 0
    total_samples = len(cast(Sized, testloader.dataset))

    # !? Avoid centralized evaluation for large dataset
    if total_samples > 5000:  # noqa: PLR2004
        # Just avoid the centralized evaluation in this stage since is too large
        log(logging.INFO, "Testloader dataset is too large")
        return (
            0.0,
            total_samples,
            {
                "test_loss": 0.0,
                "test_accuracy": 0.0,
            },
        )

    # Evaluate with no gradient computation
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(config.device), target.to(config.device)
            output = net(data)
            total_loss += criterion(output, target).item()
            num_correct += (output.max(1)[1] == target).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = float(num_correct) / total_samples

    log(
        logging.INFO,
        f"Test set: Average loss = {avg_loss:.4f}, Accuracy = {100. * accuracy:.2f}%",
    )

    return (
        avg_loss,
        total_samples,
        {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
        },
    )


# Use defaults for configuration functions
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
