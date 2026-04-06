"""1D Convolutional Neural Network for fall detection on raw time-series."""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D-CNN for classifying (batch, 6, 3000) IMU windows.

    Architecture:
        Conv1d blocks with BatchNorm + ReLU + progressive downsampling,
        followed by adaptive pooling and a linear classifier.
    """

    def __init__(
        self,
        in_channels: int = 6,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        channels = channels or [32, 64, 128, 128]
        kernel_sizes = kernel_sizes or [7, 5, 5, 3]

        layers = []
        prev_ch = in_channels
        for ch, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size=ks, stride=2, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 6, T) raw IMU time-series.

        Returns:
            (batch, 1) logits (pre-sigmoid).
        """
        x = self.features(x)      # (batch, C, T')
        x = self.pool(x)          # (batch, C, 1)
        x = x.squeeze(-1)         # (batch, C)
        return self.classifier(x)  # (batch, 1)
