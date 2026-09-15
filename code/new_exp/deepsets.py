"""Trainable set dataset and permutation-invariant classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LearnableSetDataset(nn.Module):
    """Represent every element of every synthetic set by a trainable value."""

    def __init__(
        self,
        num_classes: int,
        samples_per_class: int,
        set_size: int,
        element_dim: int = 2,
        init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if min(num_classes, samples_per_class, set_size, element_dim) < 1:
            raise ValueError("All dataset dimensions must be positive")

        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.set_size = set_size
        self.element_dim = element_dim

        shape = (num_classes, samples_per_class, set_size, element_dim)
        self.raw_data = nn.Parameter(init_scale * torch.randn(shape))
        labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
        self.register_buffer("labels", labels, persistent=False)

    def forward(self) -> Tensor:
        """Map the unconstrained parameters to a bounded data domain."""
        return torch.tanh(self.raw_data).flatten(0, 1)

    def generate_dataset(self) -> tuple[Tensor, Tensor]:
        """Return sets and their fixed class labels."""
        return self(), self.labels


class DeepSetsClassifier(nn.Module):
    """Classify sets with shared element encoding and mean aggregation."""

    def __init__(
        self,
        num_classes: int,
        element_dim: int = 2,
        hidden_dim: int = 32,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(element_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_sets(self, sets: Tensor) -> Tensor:
        """Return permutation-invariant representations before classification."""
        if sets.ndim != 3:
            raise ValueError("Expected input shape [batch, set_size, element_dim]")
        element_embeddings = self.phi(sets)
        return element_embeddings.mean(dim=1)

    def forward(self, sets: Tensor) -> Tensor:
        return self.rho(self.encode_sets(sets))


class OrderedMLPClassifier(nn.Module):
    """Classify a set after flattening it in its presented order."""

    def __init__(
        self,
        num_classes: int,
        set_size: int,
        element_dim: int = 2,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(set_size * element_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sets: Tensor) -> Tensor:
        if sets.ndim != 3:
            raise ValueError("Expected input shape [batch, set_size, element_dim]")
        return self.network(sets)


def make_deepsets_classifier(
    num_classes: int,
    *,
    element_dim: int = 2,
    hidden_dim: int = 32,
    embedding_dim: int = 32,
) -> DeepSetsClassifier:
    """Build a Deep Sets learner with notebook-friendly defaults."""
    return DeepSetsClassifier(
        num_classes=num_classes,
        element_dim=element_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    )


def make_ordered_mlp_classifier(
    num_classes: int,
    *,
    set_size: int,
    element_dim: int = 2,
    hidden_dim: int = 32,
) -> OrderedMLPClassifier:
    """Build the order-sensitive baseline used in the permutation audit."""
    return OrderedMLPClassifier(
        num_classes=num_classes,
        set_size=set_size,
        element_dim=element_dim,
        hidden_dim=hidden_dim,
    )
