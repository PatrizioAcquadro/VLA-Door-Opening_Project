"""Model utility functions."""

import torch.nn as nn
from omegaconf import DictConfig


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model(cfg: DictConfig) -> nn.Module:
    """Factory function to create model from config.

    Args:
        cfg: Hydra configuration

    Returns:
        Instantiated model
    """
    model_type = cfg.model.architecture.type

    if model_type == "transformer":
        from models.transformer import TransformerModel

        return TransformerModel(cfg)
    elif model_type == "vlm":
        from models.vlm_backbone import load_vlm_backbone

        return load_vlm_backbone(cfg)
    elif model_type == "vla":
        from models.vla_model import load_vla_model

        return load_vla_model(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def format_params(num_params: int) -> str:
    """Format parameter count for display.

    Args:
        num_params: Number of parameters

    Returns:
        Human-readable string (e.g., "1.2B", "350M", "12K")
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    else:
        return str(num_params)
