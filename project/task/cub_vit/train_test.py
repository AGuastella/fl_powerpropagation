"""Training and testing functions for TinyViT on CUB-200-2011 in federated setting."""

import logging
from logging import ERROR, WARNING
from collections.abc import Sized
from pathlib import Path
from typing import Callable, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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

class TrainConfig(BaseModel):
    """Training configuration."""
    device: torch.device
    epochs: int
    learning_rate: float = 1e-4  # Lower learning rate for transformer
    weight_decay: float = 0.05   # Weight decay for regularization
    warmup_epochs: int = 2       # Linear warmup period
    min_learning_rate: float = 1e-6
    
    class Config:
        """Allow torch.device type."""
        arbitrary_types_allowed = True


def train(
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[int, dict]:
    """Train the TinyViT model.
    
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
        
    Returns
    -------
    tuple[int, dict]
        Number of samples processed and metrics dictionary
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError("Trainloader cannot be empty")

    config = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()
    
    # Get number of classes from model
    num_classes = net.num_classes if hasattr(net, 'num_classes') else 200
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Cosine learning rate scheduler with warmup
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.min_learning_rate
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
                lr_scale = min(1., float(batch_idx + epoch * len(trainloader)) / 
                             (config.warmup_epochs * len(trainloader)))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * config.learning_rate
            
            optimizer.zero_grad()
            output = net(data)
            
            if output.shape[1] != num_classes:
                raise ValueError(
                    f"Model output has {output.shape[1]} classes but expected {num_classes}"
                )
            
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_correct += (output.max(1)[1] == target).sum().item()
            
            if batch_idx % 10 == 0:
                log(logging.INFO, 
                    f"Epoch {epoch}/{config.epochs} "
                    f"[{batch_idx * len(data)}/{total_samples} "
                    f"({100. * batch_idx / len(trainloader):.0f}%)] "
                    f"Loss: {loss.item():.6f}")
        
        # Update learning rate
        if epoch >= config.warmup_epochs:
            scheduler.step()
        
        final_epoch_loss = epoch_loss / len(trainloader)
        log(logging.INFO, 
            f"Epoch {epoch}: Loss = {final_epoch_loss:.4f}, "
            f"Accuracy = {100. * num_correct / total_samples:.2f}%")

    return total_samples, {
        "train_loss": final_epoch_loss,
        "train_accuracy": float(num_correct) / total_samples,
        "learning_rate": optimizer.param_groups[0]['lr']
    }


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


        # train the network, with the current parameter
        metrics = train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
        )

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

        torch.cuda.empty_cache()
        metrics[1]["sparsity"] = sparsity

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
    print(f"Total samples: {total_samples}")

    if total_samples < 10:
        print("Testloader dataset is too small")
        return (
            0.0,
            total_samples,
            {
                "test_loss": 0.0,
                "test_accuracy": 0.0,
            },
        )
    elif total_samples > 5000:
        # Just avoid the centralized evaluation in this stage since is too large
        print("Testloader dataset is too large")
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
