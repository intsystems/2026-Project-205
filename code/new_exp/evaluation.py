"""Independent permutation-invariance evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def stratified_split(
    labels: Tensor,
    train_fraction: float,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Split every class independently into train and test subsets."""
    train_parts = []
    test_parts = []
    for class_id in torch.unique(labels, sorted=True):
        indices = torch.where(labels == class_id)[0]
        order = torch.randperm(indices.numel(), generator=rng)
        split = round(train_fraction * indices.numel())
        split = min(max(split, 1), indices.numel() - 1)
        train_parts.append(indices[order[:split]])
        test_parts.append(indices[order[split:]])
    return torch.cat(train_parts), torch.cat(test_parts)


def permute_set_elements(sets: Tensor, rng: torch.Generator) -> Tensor:
    """Apply an independent random permutation to every set."""
    permutations = torch.stack(
        [torch.randperm(sets.shape[1], generator=rng) for _ in range(sets.shape[0])]
    )
    batch_indices = torch.arange(sets.shape[0]).unsqueeze(1)
    return sets[batch_indices, permutations]


def _train_model(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    x_train: Tensor,
    y_train: Tensor,
    *,
    num_classes: int,
    training_steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Train one newly initialized model with the inner-loop settings."""
    torch.manual_seed(seed)
    model = model_class(num_classes=num_classes, **model_kwargs).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return model.eval()


def evaluate_permutation_invariance(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    x: Tensor,
    y: Tensor,
    *,
    num_classes: int,
    trials: int,
    permutations_per_trial: int,
    train_fraction: float,
    training_steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device | str | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate all original/permuted train-test combinations for one model class."""
    if trials < 1 or permutations_per_trial < 1 or training_steps < 1:
        raise ValueError("Trial, permutation, and training counts must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x = x.detach().cpu()
    y = y.detach().cpu().long()
    records = {
        "original_train": {"original_test": [], "permuted_test": []},
        "permuted_train": {"original_test": [], "permuted_test": []},
    }

    for trial in range(trials):
        rng = torch.Generator().manual_seed(seed + trial)
        train_idx, test_idx = stratified_split(y, train_fraction, rng)
        x_train = x[train_idx]
        y_train = y[train_idx].to(run_device)
        x_test = x[test_idx]
        y_test = y[test_idx].to(run_device)

        # The shuffled training set is fixed within a trial.
        x_train_permuted = permute_set_elements(x_train, rng)
        model_original = _train_model(
            model_class,
            model_kwargs,
            x_train.to(run_device),
            y_train,
            num_classes=num_classes,
            training_steps=training_steps,
            learning_rate=learning_rate,
            seed=seed + 100_000 + 2 * trial,
            device=run_device,
        )
        model_permuted = _train_model(
            model_class,
            model_kwargs,
            x_train_permuted.to(run_device),
            y_train,
            num_classes=num_classes,
            training_steps=training_steps,
            learning_rate=learning_rate,
            seed=seed + 100_001 + 2 * trial,
            device=run_device,
        )

        with torch.no_grad():
            original_logits_from_original = model_original(x_test.to(run_device))
            original_logits_from_permuted = model_permuted(x_test.to(run_device))
        records["original_train"]["original_test"].append(
            float((original_logits_from_original.argmax(dim=-1) == y_test).float().mean())
        )
        records["permuted_train"]["original_test"].append(
            float((original_logits_from_permuted.argmax(dim=-1) == y_test).float().mean())
        )

        permuted_scores_original = []
        permuted_scores_permuted = []
        for _ in range(permutations_per_trial):
            x_test_permuted = permute_set_elements(x_test, rng).to(run_device)
            with torch.no_grad():
                logits_original = model_original(x_test_permuted)
                logits_permuted = model_permuted(x_test_permuted)
            permuted_scores_original.append(
                float((logits_original.argmax(dim=-1) == y_test).float().mean())
            )
            permuted_scores_permuted.append(
                float((logits_permuted.argmax(dim=-1) == y_test).float().mean())
            )

        records["original_train"]["permuted_test"].append(
            float(np.mean(permuted_scores_original))
        )
        records["permuted_train"]["permuted_test"].append(
            float(np.mean(permuted_scores_permuted))
        )

    summary = {}
    for train_condition, test_records in records.items():
        summary[train_condition] = {}
        for test_condition, values in test_records.items():
            summary[train_condition][f"{test_condition}_mean"] = float(np.mean(values))
            summary[train_condition][f"{test_condition}_std"] = float(np.std(values))
    return summary


def print_permutation_audit(
    model_name: str,
    summary: dict[str, dict[str, float]],
) -> None:
    """Print the four train-test conditions as mean plus-or-minus std."""
    print(model_name)
    for train_condition, train_label in (
        ("original_train", "original train"),
        ("permuted_train", "permuted train"),
    ):
        metrics = summary[train_condition]
        for test_condition, test_label in (
            ("original_test", "original test"),
            ("permuted_test", "permuted test"),
        ):
            mean = metrics[f"{test_condition}_mean"]
            std = metrics[f"{test_condition}_std"]
            print(f"  {train_label} -> {test_label}: {mean:.4f} +- {std:.4f}")


def evaluate_noise_robustness(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    x: Tensor,
    y: Tensor,
    *,
    num_classes: int,
    noise_levels: list[float],
    trials: int,
    noise_samples_per_trial: int,
    train_fraction: float,
    training_steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device | str | None = None,
) -> dict[float, dict[str, float]]:
    """Train on clean images and evaluate prediction stability under noise."""
    if not noise_levels or any(level < 0 for level in noise_levels):
        raise ValueError("noise_levels must contain non-negative values")
    if trials < 1 or noise_samples_per_trial < 1:
        raise ValueError("trials and noise_samples_per_trial must be positive")

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x = x.detach().cpu()
    y = y.detach().cpu().long()
    records = {
        float(level): {"accuracy": [], "flip_rate": [], "probability_shift": []}
        for level in noise_levels
    }

    for trial in range(trials):
        split_rng = torch.Generator().manual_seed(seed + trial)
        noise_rng = torch.Generator().manual_seed(seed + 100_000 + trial)
        train_idx, test_idx = stratified_split(y, train_fraction, split_rng)
        x_train = x[train_idx].to(run_device)
        y_train = y[train_idx].to(run_device)
        x_test_cpu = x[test_idx]
        y_test = y[test_idx].to(run_device)
        model = _train_model(
            model_class,
            model_kwargs,
            x_train,
            y_train,
            num_classes=num_classes,
            training_steps=training_steps,
            learning_rate=learning_rate,
            seed=seed + 200_000 + trial,
            device=run_device,
        )

        with torch.no_grad():
            clean_logits = model(x_test_cpu.to(run_device))
            clean_probabilities = clean_logits.softmax(dim=-1)
            clean_predictions = clean_logits.argmax(dim=-1)

        for level in noise_levels:
            level_accuracies = []
            level_flips = []
            level_probability_shifts = []
            for _ in range(noise_samples_per_trial):
                if level == 0:
                    noisy_images = x_test_cpu
                else:
                    noise = torch.randn(x_test_cpu.shape, generator=noise_rng)
                    noisy_images = (x_test_cpu + level * noise).clamp(0.0, 1.0)

                with torch.no_grad():
                    noisy_logits = model(noisy_images.to(run_device))
                    noisy_probabilities = noisy_logits.softmax(dim=-1)
                    noisy_predictions = noisy_logits.argmax(dim=-1)
                level_accuracies.append(
                    float((noisy_predictions == y_test).float().mean())
                )
                level_flips.append(
                    float((noisy_predictions != clean_predictions).float().mean())
                )
                level_probability_shifts.append(
                    float((noisy_probabilities - clean_probabilities).abs().sum(dim=-1).mean())
                )

            records[float(level)]["accuracy"].append(float(np.mean(level_accuracies)))
            records[float(level)]["flip_rate"].append(float(np.mean(level_flips)))
            records[float(level)]["probability_shift"].append(
                float(np.mean(level_probability_shifts))
            )

    summary = {}
    for level, metrics in records.items():
        summary[level] = {}
        for metric_name, values in metrics.items():
            summary[level][f"{metric_name}_mean"] = float(np.mean(values))
            summary[level][f"{metric_name}_std"] = float(np.std(values))
    return summary


def print_noise_audit(summary: dict[float, dict[str, float]]) -> None:
    """Print noise metrics as mean plus-or-minus std across trials."""
    for level in sorted(summary):
        metrics = summary[level]
        print(f"sigma={level:.3f}")
        for metric_name in ("accuracy", "flip_rate", "probability_shift"):
            mean = metrics[f"{metric_name}_mean"]
            std = metrics[f"{metric_name}_std"]
            print(f"  {metric_name}: {mean:.4f} +- {std:.4f}")
