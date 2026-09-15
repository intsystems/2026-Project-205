"""Trainable sequences and recurrent regression models."""

from __future__ import annotations

import torch
from collections.abc import Callable

from torch import Tensor, nn


class LearnableSequenceDataset(nn.Module):
    """Store several complete synthetic sequences as a trainable tensor."""

    def __init__(
        self,
        num_sequences: int,
        sequence_length: int,
        init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if num_sequences < 2:
            raise ValueError("At least two sequences are required")
        if sequence_length < 2:
            raise ValueError("sequence_length must include a context and target")
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.raw_sequences = nn.Parameter(
            init_scale * torch.randn(num_sequences, sequence_length)
        )

    def forward(self) -> Tensor:
        """Bound sequence values to [-1, 1]."""
        return self.raw_sequences


class VanillaRNNRegressor(nn.Module):
    """Predict the next value from the final vanilla-RNN state."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, context: Tensor) -> Tensor:
        hidden_states, _ = self.rnn(context)
        return self.output(hidden_states[:, -1])


def make_rnn_regressor(hidden_size: int = 16) -> VanillaRNNRegressor:
    return VanillaRNNRegressor(hidden_size=hidden_size)


def sequence_windows(sequences: Tensor, context_length: int) -> tuple[Tensor, Tensor]:
    """Convert complete sequences into overlapping next-step examples."""
    if sequences.ndim != 2:
        raise ValueError("Expected sequences with shape [num_sequences, length]")
    if context_length >= sequences.shape[1]:
        raise ValueError("context_length must be smaller than sequence length")
    chunks = sequences.unfold(dimension=1, size=context_length + 1, step=1)
    contexts = chunks[..., :-1].reshape(-1, context_length, 1)
    targets = chunks[..., -1].reshape(-1, 1)
    return contexts, targets


def normalized_mse(predictions: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize prediction error by target variance."""
    return (predictions - targets).square().mean() / (targets.var(unbiased=False) + eps)


def sequence_variance_floor(
    sequences: Tensor,
    *,
    minimum: float = 0.05,
    weight: float = 0.1,
) -> Tensor:
    """Prevent individual sequences from collapsing to constants."""
    variances = sequences.var(dim=1, unbiased=False)
    return weight * torch.relu(sequences.new_tensor(minimum) - variances).square().mean()


def temporal_difference_floor(
    sequences: Tensor,
    *,
    minimum: float = 0.01,
    weight: float = 0.1,
) -> Tensor:
    """Require a small amount of temporal variation in every sequence."""
    difference_energy = (sequences[:, 1:] - sequences[:, :-1]).square().mean(dim=1)
    return weight * torch.relu(
        sequences.new_tensor(minimum) - difference_energy
    ).square().mean()


def combine_sequence_regularizers(
    *regularizers: Callable[[Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    """Add configured sequence regularizers."""

    def combined(sequences: Tensor) -> Tensor:
        if not regularizers:
            return sequences.new_zeros(())
        return torch.stack([regularizer(sequences) for regularizer in regularizers]).sum()

    return combined
