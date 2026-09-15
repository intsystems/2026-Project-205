"""Differentiable meta-learning loop for next-step sequence prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import learn2learn as l2l
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rnn import normalized_mse, sequence_windows


@dataclass
class SequenceMetaStepResult:
    loss: float
    task_loss: float
    regularization: float
    mse: float


class SequenceMetaLearner:
    """Optimize held-out-sequence generalization of a recurrent learner."""

    def __init__(
        self,
        generator: nn.Module,
        learner_factory: Callable[[], nn.Module],
        *,
        context_length: int,
        regularization_fn: Optional[Callable[[Tensor], Tensor]] = None,
        meta_lr: float = 1e-3,
        inner_lr: float = 0.1,
        inner_steps: int = 10,
        train_fraction: float = 0.75,
        windows_per_split: int = 128,
        first_order: bool = False,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if min(context_length, inner_steps, windows_per_split) < 1:
            raise ValueError("Context, step, and window counts must be positive")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.generator = generator.to(self.device)
        self.learner_factory = learner_factory
        self.context_length = context_length
        self.regularization_fn = regularization_fn
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.train_fraction = train_fraction
        self.windows_per_split = windows_per_split
        self.first_order = first_order
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), lr=meta_lr)
        self.history: list[SequenceMetaStepResult] = []

    def _sequence_split(self, count: int) -> tuple[Tensor, Tensor]:
        order = torch.randperm(count, generator=self.rng)
        split = round(self.train_fraction * count)
        split = min(max(split, 1), count - 1)
        return order[:split].to(self.device), order[split:].to(self.device)

    def _sample_windows(self, sequences: Tensor) -> tuple[Tensor, Tensor]:
        contexts, targets = sequence_windows(sequences, self.context_length)
        sample_count = min(self.windows_per_split, contexts.shape[0])
        indices = torch.randperm(contexts.shape[0], generator=self.rng)[:sample_count]
        indices = indices.to(contexts.device)
        return contexts[indices], targets[indices]

    def meta_step(self) -> SequenceMetaStepResult:
        sequences = self.generator()
        train_idx, test_idx = self._sequence_split(sequences.shape[0])
        x_train, y_train = self._sample_windows(sequences[train_idx])
        x_test, y_test = self._sample_windows(sequences[test_idx])

        learner = l2l.algorithms.MAML(
            self.learner_factory().to(self.device),
            lr=self.inner_lr,
            first_order=self.first_order,
            allow_unused=False,
            allow_nograd=False,
        )
        for _ in range(self.inner_steps):
            learner.adapt(F.mse_loss(learner(x_train), y_train))

        predictions = learner(x_test)
        task_loss = normalized_mse(predictions, y_test)
        regularization = (
            self.regularization_fn(sequences)
            if self.regularization_fn is not None
            else task_loss.new_zeros(())
        )
        meta_loss = task_loss + regularization
        self.meta_optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        self.meta_optimizer.step()

        result = SequenceMetaStepResult(
            loss=float(meta_loss.detach()),
            task_loss=float(task_loss.detach()),
            regularization=float(regularization.detach()),
            mse=float(F.mse_loss(predictions, y_test).detach()),
        )
        self.history.append(result)
        return result

    def train(self, steps: int, log_every: int = 100) -> list[SequenceMetaStepResult]:
        for step in range(1, steps + 1):
            result = self.meta_step()
            if log_every > 0 and (step == 1 or step % log_every == 0 or step == steps):
                print(
                    f"step={step:04d} loss={result.loss:.5f} "
                    f"mse={result.mse:.5f}"
                )
        return self.history.copy()

    @torch.no_grad()
    def frozen_sequences(self) -> Tensor:
        return self.generator().detach().cpu().clone()
