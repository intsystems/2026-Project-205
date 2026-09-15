"""Post-hoc fading-memory analysis for generated sequences."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rnn import sequence_windows


def _sample_examples(
    sequences: Tensor,
    context_length: int,
    sample_count: int,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    contexts, targets = sequence_windows(sequences, context_length)
    count = min(sample_count, contexts.shape[0])
    indices = torch.randperm(contexts.shape[0], generator=rng)[:count]
    return contexts[indices], targets[indices]


def train_sequence_model(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    sequences: Tensor,
    *,
    context_length: int,
    train_fraction: float,
    training_steps: int,
    learning_rate: float,
    train_windows: int,
    seed: int,
    device: torch.device | str | None = None,
) -> tuple[nn.Module, Tensor]:
    """Train a fresh model and return it with held-out complete sequences."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sequences = sequences.detach().cpu()
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(sequences.shape[0], generator=rng)
    split = round(train_fraction * sequences.shape[0])
    split = min(max(split, 1), sequences.shape[0] - 1)
    train_sequences = sequences[order[:split]]
    test_sequences = sequences[order[split:]]
    x_train, y_train = _sample_examples(
        train_sequences, context_length, train_windows, rng
    )

    torch.manual_seed(seed + 100_000)
    model = model_class(**model_kwargs).to(run_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x_train.to(run_device)), y_train.to(run_device))
        loss.backward()
        optimizer.step()
    return model.eval(), test_sequences


def predict_complete_sequences(
    model: nn.Module,
    sequences: Tensor,
    *,
    context_length: int,
) -> Tensor:
    """Predict every available next value and preserve sequence grouping."""
    run_device = next(model.parameters()).device
    sequences = sequences.detach().cpu()
    windows_per_sequence = sequences.shape[1] - context_length
    contexts, _ = sequence_windows(sequences, context_length)
    with torch.no_grad():
        predictions = model(contexts.to(run_device)).squeeze(-1).cpu()
    return predictions.reshape(sequences.shape[0], windows_per_sequence)


def evaluate_fading_memory(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    sequences: Tensor,
    *,
    context_length: int,
    trials: int,
    perturbations_per_lag: int,
    train_fraction: float,
    training_steps: int,
    learning_rate: float,
    train_windows: int,
    test_windows: int,
    seed: int,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray | float]:
    """Measure how prediction sensitivity changes with temporal lag."""
    if context_length < 1 or perturbations_per_lag < 1 or trials < 1:
        raise ValueError("Context, perturbation, and trial counts must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sequences = sequences.detach().cpu()
    lag_error_increases = []
    lag_prediction_shifts = []
    baseline_mses = []

    for trial in range(trials):
        rng = torch.Generator().manual_seed(seed + trial)
        order = torch.randperm(sequences.shape[0], generator=rng)
        split = round(train_fraction * sequences.shape[0])
        split = min(max(split, 1), sequences.shape[0] - 1)
        train_sequences = sequences[order[:split]]
        test_sequences = sequences[order[split:]]
        x_train, y_train = _sample_examples(
            train_sequences, context_length, train_windows, rng
        )
        x_test, y_test = _sample_examples(
            test_sequences, context_length, test_windows, rng
        )
        x_train = x_train.to(run_device)
        y_train = y_train.to(run_device)
        x_test_device = x_test.to(run_device)
        y_test = y_test.to(run_device)

        torch.manual_seed(seed + 100_000 + trial)
        model = model_class(**model_kwargs).to(run_device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        model.train()
        for _ in range(training_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x_train), y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            baseline_predictions = model(x_test_device)
            baseline_mse = F.mse_loss(baseline_predictions, y_test)
        baseline_mses.append(float(baseline_mse))
        trial_error_increases = []
        trial_prediction_shifts = []

        # Lag 1 is the most recent element; lag context_length is the oldest.
        for lag in range(1, context_length + 1):
            error_increases = []
            prediction_shifts = []
            position = context_length - lag
            for _ in range(perturbations_per_lag):
                donor_order = torch.randperm(x_test.shape[0], generator=rng)
                perturbed = x_test.clone()
                perturbed[:, position] = x_test[donor_order, position]
                with torch.no_grad():
                    predictions = model(perturbed.to(run_device))
                    perturbed_mse = F.mse_loss(predictions, y_test)
                error_increases.append(float(perturbed_mse - baseline_mse))
                prediction_shifts.append(
                    float((predictions - baseline_predictions).abs().mean())
                )
            trial_error_increases.append(float(np.mean(error_increases)))
            trial_prediction_shifts.append(float(np.mean(prediction_shifts)))

        lag_error_increases.append(trial_error_increases)
        lag_prediction_shifts.append(trial_prediction_shifts)

    error_array = np.asarray(lag_error_increases)
    shift_array = np.asarray(lag_prediction_shifts)
    return {
        "lags": np.arange(1, context_length + 1),
        "baseline_mse_mean": float(np.mean(baseline_mses)),
        "baseline_mse_std": float(np.std(baseline_mses)),
        "error_increase_mean": error_array.mean(axis=0),
        "error_increase_std": error_array.std(axis=0),
        "prediction_shift_mean": shift_array.mean(axis=0),
        "prediction_shift_std": shift_array.std(axis=0),
    }


def evaluate_fading_memory_with_noise(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    sequences: Tensor,
    *,
    context_length: int,
    noise_std: float,
    trials: int,
    perturbations_per_lag: int,
    train_fraction: float,
    training_steps: int,
    learning_rate: float,
    train_windows: int,
    test_windows: int,
    seed: int,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray | float]:
    """Measure lag sensitivity by adding noise to one context element."""
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    if context_length < 1 or perturbations_per_lag < 1 or trials < 1:
        raise ValueError("Context, perturbation, and trial counts must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sequences = sequences.detach().cpu()
    lag_error_increases = []
    lag_prediction_shifts = []
    baseline_mses = []

    for trial in range(trials):
        rng = torch.Generator().manual_seed(seed + trial)
        order = torch.randperm(sequences.shape[0], generator=rng)
        split = round(train_fraction * sequences.shape[0])
        split = min(max(split, 1), sequences.shape[0] - 1)
        train_sequences = sequences[order[:split]]
        test_sequences = sequences[order[split:]]
        x_train, y_train = _sample_examples(
            train_sequences, context_length, train_windows, rng
        )
        x_test, y_test = _sample_examples(
            test_sequences, context_length, test_windows, rng
        )
        x_train = x_train.to(run_device)
        y_train = y_train.to(run_device)
        x_test_device = x_test.to(run_device)
        y_test = y_test.to(run_device)

        torch.manual_seed(seed + 100_000 + trial)
        model = model_class(**model_kwargs).to(run_device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        model.train()
        for _ in range(training_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x_train), y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            baseline_predictions = model(x_test_device)
            baseline_mse = F.mse_loss(baseline_predictions, y_test)
        baseline_mses.append(float(baseline_mse))
        trial_error_increases = []
        trial_prediction_shifts = []

        # Only the selected lag is perturbed; all other values remain unchanged.
        for lag in range(1, context_length + 1):
            error_increases = []
            prediction_shifts = []
            position = context_length - lag
            for _ in range(perturbations_per_lag):
                perturbed = x_test.clone()
                noise = noise_std * torch.randn(x_test.shape[0], generator=rng)
                perturbed[:, position, 0] = (
                    perturbed[:, position, 0] + noise
                ).clamp(-1.0, 1.0)
                with torch.no_grad():
                    predictions = model(perturbed.to(run_device))
                    perturbed_mse = F.mse_loss(predictions, y_test)
                error_increases.append(float(perturbed_mse - baseline_mse))
                prediction_shifts.append(
                    float((predictions - baseline_predictions).abs().mean())
                )
            trial_error_increases.append(float(np.mean(error_increases)))
            trial_prediction_shifts.append(float(np.mean(prediction_shifts)))

        lag_error_increases.append(trial_error_increases)
        lag_prediction_shifts.append(trial_prediction_shifts)

    error_array = np.asarray(lag_error_increases)
    shift_array = np.asarray(lag_prediction_shifts)
    return {
        "lags": np.arange(1, context_length + 1),
        "baseline_mse_mean": float(np.mean(baseline_mses)),
        "baseline_mse_std": float(np.std(baseline_mses)),
        "error_increase_mean": error_array.mean(axis=0),
        "error_increase_std": error_array.std(axis=0),
        "prediction_shift_mean": shift_array.mean(axis=0),
        "prediction_shift_std": shift_array.std(axis=0),
    }


def print_fading_memory_summary(summary: dict[str, np.ndarray | float]) -> None:
    """Print the baseline error and the most influential lags."""
    print(
        f"baseline MSE: {summary['baseline_mse_mean']:.6f} +- "
        f"{summary['baseline_mse_std']:.6f}"
    )
    shifts = np.asarray(summary["prediction_shift_mean"])
    lags = np.asarray(summary["lags"])
    ranking = np.argsort(shifts)[::-1][:10]
    print("most influential lags:")
    for index in ranking:
        print(f"  lag {lags[index]:3d}: prediction shift {shifts[index]:.6f}")
