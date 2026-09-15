"""Reusable differentiable meta-learning loop for synthetic datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import learn2learn as l2l
import torch
from torch import Tensor, nn


MetaLoss = Callable[[Tensor, Tensor], Tensor]
Regularizer = Callable[[Tensor, Tensor], Tensor]
LearnerFactory = Callable[[int], nn.Module]


@dataclass
class MetaStepResult:
    """Metrics recorded after one outer-loop update."""

    loss: float
    task_loss: float
    regularization: float
    accuracy: float


class MetaLearner:
    """Optimize a tensor generator through a differentiable inner loop."""

    def __init__(
        self,
        generator: nn.Module,
        learner_factory: LearnerFactory,
        meta_loss_fn: MetaLoss,
        num_classes: int,
        *,
        inner_loss_fn: Optional[MetaLoss] = None,
        regularization_fn: Optional[Regularizer] = None,
        meta_lr: float = 1e-3,
        inner_lr: float = 0.1,
        inner_steps: int = 10,
        learners_per_step: int = 1,
        train_fraction: float = 0.75,
        first_order: bool = False,
        seed: int = 0,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        if learners_per_step < 1:
            raise ValueError("learners_per_step must be positive")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.generator = generator.to(self.device)
        self.learner_factory = learner_factory
        self.meta_loss_fn = meta_loss_fn
        self.inner_loss_fn = inner_loss_fn or meta_loss_fn
        self.regularization_fn = regularization_fn
        self.num_classes = num_classes
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.learners_per_step = learners_per_step
        self.train_fraction = train_fraction
        self.first_order = first_order

        # A private RNG keeps experiment randomness reproducible.
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), lr=meta_lr)
        self.history: list[MetaStepResult] = []

    def _stratified_split(self, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Create a balanced split with every class represented on both sides."""
        train_parts = []
        test_parts = []

        for class_id in range(self.num_classes):
            class_indices = torch.where(labels.detach().cpu() == class_id)[0]
            if class_indices.numel() < 2:
                raise ValueError("Each class needs at least two samples")

            order = torch.randperm(class_indices.numel(), generator=self.rng)
            class_indices = class_indices[order]
            split = round(self.train_fraction * class_indices.numel())
            split = min(max(split, 1), class_indices.numel() - 1)
            train_parts.append(class_indices[:split])
            test_parts.append(class_indices[split:])

        train_idx = torch.cat(train_parts)
        test_idx = torch.cat(test_parts)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=self.rng)]
        test_idx = test_idx[torch.randperm(test_idx.numel(), generator=self.rng)]
        return train_idx.to(labels.device), test_idx.to(labels.device)

    def meta_step(self) -> MetaStepResult:
        """Run one differentiable inner loop and update the generator once."""
        self.generator.train()
        x, y = self.generator.generate_dataset()
        x = x.to(self.device)
        y = y.to(self.device, dtype=torch.long)
        train_idx, test_idx = self._stratified_split(y)

        task_losses = []
        accuracies = []

        # Average several fresh learners to reduce initialization-specific gradients.
        for _ in range(self.learners_per_step):
            base_learner = self.learner_factory(self.num_classes).to(self.device)
            learner = l2l.algorithms.MAML(
                base_learner,
                lr=self.inner_lr,
                first_order=self.first_order,
                allow_unused=False,
                allow_nograd=False,
            )

            for _ in range(self.inner_steps):
                train_logits = learner(x[train_idx])
                inner_loss = self.inner_loss_fn(train_logits, y[train_idx])
                learner.adapt(inner_loss)

            test_logits = learner(x[test_idx])
            task_losses.append(self.meta_loss_fn(test_logits, y[test_idx]))
            with torch.no_grad():
                accuracies.append(
                    (test_logits.argmax(dim=-1) == y[test_idx]).float().mean()
                )

        task_loss = torch.stack(task_losses).mean()
        regularization = (
            self.regularization_fn(x, y)
            if self.regularization_fn is not None
            else task_loss.new_zeros(())
        )
        meta_loss = task_loss + regularization

        self.meta_optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        self.meta_optimizer.step()

        accuracy = torch.stack(accuracies).mean()

        result = MetaStepResult(
            loss=float(meta_loss.detach()),
            task_loss=float(task_loss.detach()),
            regularization=float(regularization.detach()),
            accuracy=float(accuracy.detach()),
        )
        self.history.append(result)
        return result

    def train(self, steps: int, log_every: int = 100) -> list[MetaStepResult]:
        """Run meta-training and return a copy of the recorded history."""
        if steps < 1:
            raise ValueError("steps must be positive")

        for step in range(1, steps + 1):
            result = self.meta_step()
            if log_every > 0 and (step == 1 or step % log_every == 0 or step == steps):
                print(
                    f"step={step:04d} loss={result.loss:.4f} accuracy={result.accuracy:.3f}"
                )

        return self.history.copy()

    @torch.no_grad()
    def frozen_dataset(self) -> tuple[Tensor, Tensor]:
        """Return a detached CPU copy for independent post-hoc evaluation."""
        self.generator.eval()
        x, y = self.generator.generate_dataset()
        return x.detach().cpu().clone(), y.detach().cpu().clone()
