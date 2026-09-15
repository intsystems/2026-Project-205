"""Trainable image dataset and CNN models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LearnableImageDataset(nn.Module):
    """Represent every synthetic image by a directly trainable tensor."""

    def __init__(
        self,
        num_classes: int,
        samples_per_class: int,
        image_size: int = 16,
        channels: int = 1,
        init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if min(num_classes, samples_per_class, image_size, channels) < 1:
            raise ValueError("All dataset dimensions must be positive")

        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.image_size = image_size
        shape = (num_classes, samples_per_class, channels, image_size, image_size)
        self.raw_data = nn.Parameter(init_scale * torch.randn(shape))
        labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
        self.register_buffer("labels", labels, persistent=False)

    def forward(self) -> Tensor:
        """Map unconstrained parameters to grayscale intensities in [0, 1]."""
        return torch.sigmoid(self.raw_data).flatten(0, 1)

    def generate_dataset(self) -> tuple[Tensor, Tensor]:
        return self(), self.labels


class TwoConvClassifier(nn.Module):
    """Two convolutional layers followed by pooling and a linear head."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 16,
        channels: int = 1,
        first_channels: int = 8,
        second_channels: int = 16,
        pool_type: str = "max",
    ) -> None:
        super().__init__()
        if image_size % 2 != 0:
            raise ValueError("image_size must be even for 2x2 pooling")
        if pool_type == "max":
            pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == "average":
            pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("pool_type must be 'max' or 'average'")

        self.features = nn.Sequential(
            nn.Conv2d(channels, first_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            pooling,
        )
        pooled_size = image_size // 2
        self.classifier = nn.Linear(
            second_channels * pooled_size * pooled_size,
            num_classes,
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        return self.classifier(features.flatten(start_dim=1))


def make_two_conv_classifier(
    num_classes: int,
    *,
    image_size: int = 16,
    channels: int = 1,
    first_channels: int = 8,
    second_channels: int = 16,
    pool_type: str = "max",
) -> TwoConvClassifier:
    """Build the CNN with notebook-friendly keyword arguments."""
    return TwoConvClassifier(
        num_classes=num_classes,
        image_size=image_size,
        channels=channels,
        first_channels=first_channels,
        second_channels=second_channels,
        pool_type=pool_type,
    )
