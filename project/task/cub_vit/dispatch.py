"""Dispatch functionality for the CUB-200-2011 federated learning task using TinyViT.

This module provides dispatch functions that dynamically select the appropriate
functions for training and data handling based on the hydra configuration file. The
dispatch system follows a pipeline pattern, returning None if it cannot match the
configuration, allowing other dispatchers in the chain to handle the request.
"""

from pathlib import Path
from omegaconf import DictConfig

from project.task.default.dispatch import dispatch_config as dispatch_default_config
from project.task.cub_vit.dataset import get_dataloader_generators
from project.task.cub_vit.models import (
    get_network_generator_vit_sparsyfed,
    get_network_generator_vit_sparsyfed_no_act,
    get_tinyvit,
)
from project.task.cub_vit.train_test import (
    get_train_and_prune,
    train,
    test,
    get_fed_eval_fn,
)
from project.types.common import DataStructure, TrainStructure


def dispatch_train(
    cfg: DictConfig,
) -> TrainStructure | None:
    """Dispatch the training, testing, and evaluation functions based on configuration.

    This function handles the selection of appropriate training functions for the
    TinyViT model on the CUB-200-2011 dataset. It maintains compatibility with the
    federated learning pipeline while providing specialized handling for transfer
    learning scenarios.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the training functions, loaded dynamically from the
        hydra config file.

    Returns
    -------
    Optional[TrainStructure]
        A tuple containing the train function, test function, and federated evaluation
        function. Returns None if the configuration cannot be matched.
    """
    train_structure: str | None = cfg.get("task", {}).get(
        "train_structure",
        None,
    )
    alpha: float = cfg.get("task", {}).get("alpha", 1.0)
    sparsity: float = cfg.get("task", {}).get("sparsity", 0.0)

    if train_structure is not None and train_structure.upper() == "CUB_TINYVIT":
        return (
            train,
            test,
            get_fed_eval_fn,
        )
    elif train_structure is not None and train_structure.upper() == "CUB_TINYVIT_PRUNE":
        return (
            get_train_and_prune(alpha=alpha, amount=sparsity, pruning_method="l1"),
            test,
            get_fed_eval_fn,
        )
    elif (
        train_structure is not None
        and train_structure.upper() == "CUB_TINYVIT_PRUNE_FIX"
    ):
        return (
            get_train_and_prune(
                alpha=alpha, amount=sparsity, pruning_method="l1", use_mask=True
            ),
            test,
            get_fed_eval_fn,
        )

    return None


def dispatch_data(cfg: DictConfig) -> DataStructure | None:
    """Dispatch the data and model generation functions based on configuration.

    This function handles the creation of appropriate model and data loader generators
    for the TinyViT model on the CUB-200-2011 dataset. It provides comprehensive
    configuration options for transfer learning and model initialization.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the data functions, loaded dynamically from the
        hydra config file.

    Returns
    -------
    Optional[DataStructure]
        A tuple containing the network generator, client dataloader generator, and
        federated dataloader generator. Returns None if the configuration cannot be
        matched.
    """
    client_model_and_data: str | None = cfg.get("task", {}).get("model_and_data", None)
    partition_dir: str | None = cfg.get("dataset", {}).get("partition_dir", None)

    if client_model_and_data is not None and partition_dir is not None:
        client_dataloader_gen, fed_dataloader_gen = get_dataloader_generators(
            Path(partition_dir),
        )
        alpha: float = cfg.get("task", {}).get("alpha", 1.0)
        sparsity: float = cfg.get("task", {}).get("sparsity", 0.0)

        if client_model_and_data.upper() == "CUB_TINYVIT":
            # Extract model configuration parameters
            num_classes: int = cfg.get("dataset", {}).get("num_classes", 200)
            pretrained: bool = cfg.get("task", {}).get("pretrained", True)
            freeze_backbone: bool = cfg.get("task", {}).get("freeze_backbone", False)
            return (
                get_tinyvit(
                    num_classes=num_classes,
                    pretrained=pretrained,
                    freeze_backbone=freeze_backbone,
                ),
                client_dataloader_gen,
                fed_dataloader_gen,
            )
        elif client_model_and_data.upper() == "CUB_TINYVIT_SPARSYFED":
            return (
                get_network_generator_vit_sparsyfed(
                    num_classes=200,
                    alpha=alpha,
                    sparsity=sparsity,
                    pretrained=True,
                    freeze_backbone=False,
                ),
                client_dataloader_gen,
                fed_dataloader_gen,
            )
        elif client_model_and_data.upper() == "CUB_TINYVIT_SPARSYFED_NO_ACT":
            return (
                get_network_generator_vit_sparsyfed_no_act(
                    num_classes=200,
                    alpha=alpha,
                    sparsity=sparsity,
                    pretrained=True,
                    freeze_backbone=False,
                ),
                client_dataloader_gen,
                fed_dataloader_gen,
            )

    return None


# Use the default configuration dispatch
dispatch_config = dispatch_default_config
