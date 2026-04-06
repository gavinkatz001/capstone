"""Loss functions for fall detection training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for binary classification. Down-weights easy examples."""

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
        )
        probs = torch.sigmoid(logits)
        # p_t = prob of correct class
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def get_loss_fn(name: str, pos_weight: float = 1.0, focal_gamma: float = 2.0) -> nn.Module:
    """Create a loss function by name."""
    if name == "bce":
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
    elif name == "focal":
        return FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss '{name}'. Available: 'bce', 'focal'")
