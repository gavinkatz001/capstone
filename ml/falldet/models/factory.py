"""Model registry: name -> class mapping."""

import torch.nn as nn

from falldet.models.cnn1d import CNN1D

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn1d": CNN1D,
}


def create_model(name: str, **kwargs) -> nn.Module:
    """Create a model by name with the given keyword arguments."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)
