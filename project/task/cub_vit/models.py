"""TinyViT model adaptation for federated learning with CUB-200-2011 dataset."""

from copy import deepcopy
from collections.abc import Callable
from typing import Iterable
from project.task.utils.sparsyfed_modules_vit import SparsyFedLinear
from project.task.utils.sparsyfed_no_act_modules_vit import SparsyFed_no_act_linear
import timm
import torch
from torch import nn


class NetCubTinyViT(nn.Module):
    """TinyViT model adapted for CUB-200-2011 dataset."""

    def __init__(
        self,
        num_classes: int = 200,
        device: str = "cuda",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize the TinyViT network.

        Parameters
        ----------
        num_classes : int
            Number of output classes (200 for CUB-200-2011)
        device : str
            Device to initialize the model on
        pretrained : bool
            Whether to use pretrained weights
        freeze_backbone : bool
            Whether to freeze the backbone during initial training
        """
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        # Load the pretrained model using timm
        self.net = timm.create_model(
            "tiny_vit_5m_224.dist_in22k_ft_in1k",  # Using a small, efficient variant
            pretrained=pretrained,
            num_classes=0,  # Remove the classification head
        )

        # Add custom classification head
        num_features = self.net.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features), nn.Linear(num_features, num_classes)
        )

        if freeze_backbone:
            self._freeze_backbone()

        # Initialize the new classification head
        self._init_classifier()

    def _freeze_backbone(self) -> None:
        """Freeze all backbone layers."""
        for param in self.net.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.net.parameters():
            param.requires_grad = True

    def _init_classifier(self) -> None:
        """Initialize the classification head with appropriate scaling."""
        nn.init.trunc_normal_(self.classifier[1].weight, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, 224, 224)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes)
        """
        features = self.net(x)
        return self.classifier(features)


def get_tinyvit(
    num_classes: int = 200, pretrained: bool = True, freeze_backbone: bool = False
) -> Callable[[dict], NetCubTinyViT]:
    """TinyViT network generator for CUB-200-2011.

    Parameters
    ----------
    num_classes : int
        Number of classes in the dataset
    pretrained : bool
        Whether to use pretrained weights
    freeze_backbone : bool
        Whether to freeze the backbone during initial training

    Returns
    -------
    Callable[[dict], NetCubTinyViT]
        A function that generates a new instance of the network
    """
    untrained_net = NetCubTinyViT(
        num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone
    )

    def generated_net(_config: dict) -> NetCubTinyViT:
        """Return a deep copy of the untrained network."""
        return deepcopy(untrained_net)

    return generated_net



# sparsyfed
def replace_layer_with_sparsyfed(
    module: nn.Module,
    name: str = "Model",
    alpha: float = 1.0,
    sparsity: float = 0.0,
) -> None:
    """Replace Linear layers in ViT with SparsyFed versions while keeping old weights.
    
    Parameters
    ----------
    module : nn.Module
        The module to modify
    name : str
        Name of the module (for logging)
    alpha : float
        PowerProp alpha parameter
    sparsity : float
        Initial sparsity level
    """
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if isinstance(target_attr, nn.Linear):
            new_layer = SparsyFedLinear(
                alpha=alpha,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
                sparsity=sparsity,
            )
            # Copy weights and bias from the old layer
            new_layer.weight.data = target_attr.weight.data.clone()
            if target_attr.bias is not None:
                new_layer.bias.data = target_attr.bias.data.clone()
            setattr(module, attr_str, new_layer)

    # Recursively handle child modules
    for model, immediate_child_module in module.named_children():
        replace_layer_with_sparsyfed(immediate_child_module, model, alpha, sparsity)

def get_network_generator_vit_sparsyfed(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> Callable[[dict], NetCubTinyViT]:
    """PowerSWAT ViT generator.
    
    Parameters
    ----------
    alpha : float
        PowerProp alpha parameter
    sparsity : float
        Initial sparsity level
    num_classes : int
        Number of output classes
        
    Returns
    -------
    Callable[[dict], NetCubTinyViT]
        A function that generates a PowerSWAT ViT model
    """
    # Create base network
    untrained_net: NetCubTinyViT = NetCubTinyViT(
        num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone
    )
    
    # Replace standard layers with PowerSWAT versions
    replace_layer_with_sparsyfed(
        module=untrained_net,
        name="NetCubTinyViT",
        alpha=alpha,
        sparsity=sparsity,
    )

    # Initialize weights
    # def init_model(
    #     module: nn.Module,
    # ) -> None:
    #     """Initialize the weights of the layers."""
    #     if isinstance(module, SparsyFedLinear):
    #         init_weights(module)
    #     for _, immediate_child_module in module.named_children():
    #         init_model(immediate_child_module)
    # init_model(untrained_net)


    def generated_net(_config: dict) -> NetCubTinyViT:
        """Return a deep copy of the untrained network."""
        return deepcopy(untrained_net)

    return generated_net


# sparsyfed no act
def replace_layer_with_sparsyfed_no_act(
    module: nn.Module,
    name: str = "Model",
    alpha: float = 1.0,
    sparsity: float = 0.0,
) -> None:
    """Replace Linear layers in ViT with SparsyFed (no activation) versions while keeping old weights.
    
    Parameters
    ----------
    module : nn.Module
        The module to modify
    name : str
        Name of the module (for logging)
    alpha : float
        PowerProp alpha parameter
    sparsity : float
        Initial sparsity level
    """
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if isinstance(target_attr, nn.Linear):
            new_layer = SparsyFed_no_act_linear(
                alpha=alpha,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
                sparsity=sparsity,
            )
            # Copy weights and bias from the old layer
            new_layer.weight.data = target_attr.weight.data.clone()
            if target_attr.bias is not None:
                new_layer.bias.data = target_attr.bias.data.clone()
            setattr(module, attr_str, new_layer)

    # Recursively handle child modules
    for model, immediate_child_module in module.named_children():
        replace_layer_with_sparsyfed_no_act(immediate_child_module, model, alpha, sparsity)

def get_network_generator_vit_sparsyfed_no_act(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> Callable[[dict], NetCubTinyViT]:
    """PowerSWAT ViT generator.
    
    Parameters
    ----------
    alpha : float
        PowerProp alpha parameter
    sparsity : float
        Initial sparsity level
    num_classes : int
        Number of output classes
        
    Returns
    -------
    Callable[[dict], NetCubTinyViT]
        A function that generates a PowerSWAT ViT model
    """
    # Create base network
    net: NetCubTinyViT = NetCubTinyViT(
        num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone
    )
    
    # Replace standard layers with PowerSWAT versions
    replace_layer_with_sparsyfed_no_act(
        module=net,
        name="NetCubTinyViT",
        alpha=alpha,
        sparsity=sparsity,
    )

    # Initialize weights
    # def init_model(
    #     module: nn.Module,
    # ) -> None:
    #     """Initialize the weights of the layers."""
    #     if isinstance(module, SparsyFed_no_act_linear):
    #         init_weights(module)
    #     for _, immediate_child_module in module.named_children():
    #         init_model(immediate_child_module)
    # init_model(untrained_net)


    def generated_net(_config: dict) -> NetCubTinyViT:
        """Return a deep copy of the untrained network."""
        return deepcopy(net)

    return generated_net

def get_parameters_to_prune(
    net: nn.Module,
) -> list[tuple[nn.Module, str, str]]:
    """Return prunable parameters from the ViT model.
    
    Similar to ResNet implementation but adapted for ViT architecture.
    Prunes the linear layers in attention and MLP blocks.
    """
    parameters_to_prune = []

    def add_immediate_child(
        module: nn.Module,
        name: str,
    ) -> None:
        # Skip first projection layer if specified
        if isinstance(module, (nn.Linear, SparsyFedLinear)):
            parameters_to_prune.append((module, "weight", name))

        for _name, immediate_child_module in module.named_children():
            add_immediate_child(immediate_child_module, _name)

    add_immediate_child(net, "Net")
    return parameters_to_prune

