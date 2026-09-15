"""Losses and regularizers shared by the synthetic-data experiments."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy for integer class labels."""
    return F.cross_entropy(logits, targets)


def class_balanced_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    dispersion_weight: float = 0.5,
) -> Tensor:
    """Penalize the mean class error and unequal errors across classes."""
    sample_losses = F.cross_entropy(logits, targets, reduction="none")
    class_losses = torch.stack(
        [
            sample_losses[targets == class_id].mean()
            for class_id in torch.unique(targets, sorted=True)
        ]
    )
    dispersion = class_losses.std(unbiased=False)
    return class_losses.mean() + dispersion_weight * dispersion


def pairwise_class_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    temperature: float = 0.2,
) -> Tensor:
    """Focus the outer loss on the least separable pair of classes."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    classes = torch.unique(targets, sorted=True)
    if classes.numel() < 2:
        raise ValueError("At least two classes are required")

    pair_losses = []
    for first_index in range(classes.numel() - 1):
        for second_index in range(first_index + 1, classes.numel()):
            first_class = classes[first_index]
            second_class = classes[second_index]
            pair_mask = (targets == first_class) | (targets == second_class)
            pair_logits = logits[pair_mask][:, [first_class, second_class]]
            pair_targets = (targets[pair_mask] == second_class).long()
            pair_losses.append(F.cross_entropy(pair_logits, pair_targets))

    pair_losses = torch.stack(pair_losses)
    # Normalization makes the loss equal to CE when only one pair exists.
    normalization = pair_losses.new_tensor(pair_losses.numel()).log()
    return temperature * (
        torch.logsumexp(pair_losses / temperature, dim=0) - normalization
    )


def probability_mse_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """MSE between predicted probabilities and one-hot labels."""
    probabilities = logits.softmax(dim=-1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).to(probabilities.dtype)
    return F.mse_loss(probabilities, one_hot)


def intra_class_diversity(
    x: Tensor,
    y: Tensor,
    *,
    weight: float = 0.1,
) -> Tensor:
    """Reward average pairwise distance within each class."""
    flattened = x.flatten(start_dim=1)
    scale = flattened.shape[1] ** 0.5
    class_terms = []

    for class_id in torch.unique(y, sorted=True):
        class_samples = flattened[y == class_id]
        if class_samples.shape[0] < 2:
            continue
        distances = torch.pdist(class_samples, p=2) / scale
        class_terms.append(distances.mean())

    if not class_terms:
        return x.new_zeros(())
    return -weight * torch.stack(class_terms).mean()


def within_set_separation(
    x: Tensor,
    y: Tensor,
    *,
    minimum_distance: float = 0.2,
    weight: float = 0.1,
) -> Tensor:
    """Penalize points that are too close within the same set."""
    del y  # Class labels do not affect within-set geometry.
    if x.ndim != 3:
        raise ValueError("Expected input shape [batch, set_size, element_dim]")
    if x.shape[1] < 2:
        return x.new_zeros(())

    # Select each unordered pair once and impose a soft distance floor.
    distances = torch.cdist(x, x, p=2)
    pair_mask = torch.triu(
        torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device),
        diagonal=1,
    )
    pairwise_distances = distances[:, pair_mask]
    violations = F.relu(x.new_tensor(minimum_distance) - pairwise_distances)
    return weight * violations.square().mean()


def aligned_element_separation(
    x: Tensor,
    y: Tensor,
    *,
    minimum_distance: float = 0.2,
    weight: float = 0.1,
) -> Tensor:
    """Penalize matching positions that coincide across sets of one class."""
    if x.ndim != 3:
        raise ValueError("Expected input shape [batch, set_size, element_dim]")

    penalties = []
    for class_id in torch.unique(y, sorted=True):
        class_sets = x[y == class_id]
        if class_sets.shape[0] < 2:
            continue

        # Distances are computed across samples separately for every element index.
        positioned_points = class_sets.transpose(0, 1)
        for points_at_position in positioned_points:
            distances = torch.pdist(points_at_position, p=2)
            violations = F.relu(x.new_tensor(minimum_distance) - distances)
            penalties.append(violations.square().mean())

    if not penalties:
        return x.new_zeros(())
    return weight * torch.stack(penalties).mean()


def variance_floor(
    x: Tensor,
    y: Tensor,
    *,
    minimum: float = 0.02,
    weight: float = 0.1,
) -> Tensor:
    """Penalize classes whose generated samples nearly collapse."""
    penalties = []
    flattened = x.flatten(start_dim=1)
    for class_id in torch.unique(y, sorted=True):
        class_variance = flattened[y == class_id].var(dim=0, unbiased=False).mean()
        penalties.append(F.relu(x.new_tensor(minimum) - class_variance).square())
    return weight * torch.stack(penalties).mean()


def combine_regularizers(*regularizers: Callable[[Tensor, Tensor], Tensor]):
    """Combine regularizers while preserving their configured weights."""

    def combined(x: Tensor, y: Tensor) -> Tensor:
        if not regularizers:
            return x.new_zeros(())
        return torch.stack([regularizer(x, y) for regularizer in regularizers]).sum()

    return combined


def make_diversity_regularizer(weight: float = 0.1):
    """Create a configured intra-class diversity regularizer."""
    return partial(intra_class_diversity, weight=weight)


def make_separation_regularizer(
    minimum_distance: float = 0.2,
    weight: float = 0.1,
):
    """Create a configured within-set separation regularizer."""
    return partial(
        within_set_separation,
        minimum_distance=minimum_distance,
        weight=weight,
    )
