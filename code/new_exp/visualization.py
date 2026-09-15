"""Plotting helpers for meta-training and set-valued datasets."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from meta_learning import MetaStepResult


def moving_average(values: Sequence[float], window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned indices and a moving average."""
    data = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be positive")
    if window == 1 or data.size < window:
        return np.arange(data.size), data
    kernel = np.ones(window, dtype=float) / window
    return np.arange(window - 1, data.size), np.convolve(data, kernel, mode="valid")


def plot_meta_history(history: Sequence[MetaStepResult], smoothing_window: int = 20):
    """Plot the final meta-loss and meta-test accuracy."""
    if not history:
        raise ValueError("history is empty")

    total_loss = [item.loss for item in history]
    accuracy = [item.accuracy for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(total_loss, color="tab:blue", alpha=0.2, linewidth=1, label="raw")
    x_axis, smoothed = moving_average(total_loss, smoothing_window)
    axes[0].plot(x_axis, smoothed, color="tab:blue", linewidth=2, label="moving average")
    axes[0].set(
        title=f"Final meta-loss (moving average, window={smoothing_window})",
        xlabel="Meta-iteration",
        ylabel="Loss",
    )
    axes[0].legend()

    x_axis, smoothed = moving_average(accuracy, smoothing_window)
    axes[1].plot(x_axis, smoothed, color="tab:green")
    axes[1].set(title="Meta-test accuracy", xlabel="Meta-iteration", ylabel="Accuracy", ylim=(0, 1.05))

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_set_dataset(
    x: Tensor,
    y: Tensor,
    *,
    samples_per_class: int = 6,
    limits: tuple[float, float] = (-1.1, 1.1),
):
    """Visualize two-dimensional sets as small scatter plots."""
    x = x.detach().cpu()
    y = y.detach().cpu()
    classes = torch.unique(y, sorted=True).tolist()
    cols = min(samples_per_class, min(int((y == class_id).sum()) for class_id in classes))
    fig, axes = plt.subplots(len(classes), cols, figsize=(2.2 * cols, 2.2 * len(classes)), squeeze=False)

    for row, class_id in enumerate(classes):
        indices = torch.where(y == class_id)[0][:cols]
        for col, index in enumerate(indices):
            points = x[index]
            point_order = np.arange(points.shape[0])
            axes[row, col].scatter(
                points[:, 0],
                points[:, 1],
                c=point_order,
                cmap="viridis",
                vmin=0,
                vmax=max(points.shape[0] - 1, 1),
                s=32,
                alpha=0.9,
            )
            axes[row, col].set(xlim=limits, ylim=limits, aspect="equal")
            axes[row, col].grid(alpha=0.2)
            if col == 0:
                axes[row, col].set_ylabel(f"Class {class_id}")

    fig.tight_layout()
    return fig, axes


def plot_permutation_audit(summary: dict[str, float]):
    """Compare accuracy before and after element permutations."""
    labels = ["Original", "Permuted"]
    means = [summary["original_accuracy_mean"], summary["permuted_accuracy_mean"]]
    errors = [summary["original_accuracy_std"], summary["permuted_accuracy_std"]]
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(labels, means, yerr=errors, capsize=5, color=["tab:blue", "tab:orange"])
    axis.set(ylabel="Accuracy", ylim=(0, 1.05), title="Permutation audit")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, axis


def plot_image_dataset(
    x: Tensor,
    y: Tensor,
    *,
    samples_per_class: int = 8,
):
    """Display generated grayscale images grouped by class."""
    x = x.detach().cpu()
    y = y.detach().cpu()
    classes = torch.unique(y, sorted=True).tolist()
    cols = min(samples_per_class, min(int((y == class_id).sum()) for class_id in classes))
    fig, axes = plt.subplots(
        len(classes), cols, figsize=(1.6 * cols, 1.6 * len(classes)), squeeze=False
    )
    for row, class_id in enumerate(classes):
        indices = torch.where(y == class_id)[0][:cols]
        for col, index in enumerate(indices):
            axes[row, col].imshow(x[index, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(f"Class {class_id}")
    fig.tight_layout()
    return fig, axes


def plot_noise_robustness(summary: dict[float, dict[str, float]]):
    """Plot accuracy and prediction changes against noise magnitude."""
    levels = sorted(summary)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specifications = (
        ("accuracy", "Accuracy"),
        ("flip_rate", "Prediction flip rate"),
        ("probability_shift", "Mean probability shift (L1)"),
    )
    for axis, (metric_name, title) in zip(axes, specifications):
        means = [summary[level][f"{metric_name}_mean"] for level in levels]
        errors = [summary[level][f"{metric_name}_std"] for level in levels]
        axis.errorbar(levels, means, yerr=errors, marker="o", capsize=4)
        axis.set(title=title, xlabel="Noise standard deviation")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    fig.tight_layout()
    return fig, axes


def plot_generated_sequences(
    sequences: Tensor,
    count: int = 6,
    predictions: Tensor | None = None,
    context_length: int | None = None,
):
    """Plot complete sequences and optionally aligned one-step predictions."""
    sequences = sequences.detach().cpu()
    if predictions is not None:
        predictions = predictions.detach().cpu()
        if context_length is None:
            raise ValueError("context_length is required when predictions are provided")
        expected_shape = (sequences.shape[0], sequences.shape[1] - context_length)
        if tuple(predictions.shape) != expected_shape:
            raise ValueError(f"Expected predictions with shape {expected_shape}")
    shown = min(count, sequences.shape[0])
    fig, axes = plt.subplots(shown, 1, figsize=(14, 2.2 * shown), squeeze=False)
    for index in range(shown):
        axes[index, 0].plot(
            sequences[index],
            linewidth=1,
            color="tab:blue",
            label="generated sequence" if index == 0 else None,
        )
        if predictions is not None:
            prediction_positions = np.arange(context_length, sequences.shape[1])
            axes[index, 0].plot(
                prediction_positions,
                predictions[index],
                color="tab:red",
                linewidth=1,
                alpha=0.85,
                label="RNN prediction" if index == 0 else None,
            )
        axes[index, 0].set_ylabel(f"Seq. {index}")
        axes[index, 0].grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time step")
    if predictions is not None:
        axes[0, 0].legend()
    fig.tight_layout()
    return fig, axes


def plot_sequence_meta_loss(history, smoothing_window: int = 20):
    """Plot the final sequence meta-loss with a moving average."""
    losses = [item.loss for item in history]
    x_axis, smoothed = moving_average(losses, smoothing_window)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(losses, color="tab:blue", alpha=0.2, linewidth=1, label="raw")
    axis.plot(x_axis, smoothed, color="tab:blue", linewidth=2, label="moving average")
    axis.set(
        title=f"Meta-loss (moving average, window={smoothing_window})",
        xlabel="Meta-iteration",
        ylabel="Loss",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return fig, axis


def plot_fading_memory(summary):
    """Plot prediction sensitivity and error increase against lag."""
    lags = np.asarray(summary["lags"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    specifications = (
        ("prediction_shift", "Mean absolute prediction shift"),
        ("error_increase", "Increase in MSE"),
    )
    for axis, (metric, title) in zip(axes, specifications):
        mean = np.asarray(summary[f"{metric}_mean"])
        std = np.asarray(summary[f"{metric}_std"])
        axis.plot(lags, mean, color="tab:blue")
        axis.fill_between(lags, mean - std, mean + std, color="tab:blue", alpha=0.2)
        axis.set(title=title, xlabel="Lag (1 = most recent)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig, axes
