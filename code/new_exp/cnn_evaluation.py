"""Post-hoc geometric generalization tests for image classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from evaluation import stratified_split


def embed_in_canvas(images: Tensor, canvas_size: int) -> Tensor:
    """Center a batch of images on a larger zero-valued square canvas."""
    if images.ndim != 4:
        raise ValueError("Expected images with shape [batch, channels, height, width]")
    height, width = images.shape[-2:]
    if height != width:
        raise ValueError("Only square images are supported")
    if canvas_size < height:
        raise ValueError("canvas_size must be at least the image size")

    margin = canvas_size - height
    left = margin // 2
    right = margin - left
    return F.pad(images, (left, right, left, right))


def random_integer_translation(
    images: Tensor,
    *,
    max_shift: int,
    rng: torch.Generator,
) -> Tensor:
    """Translate each image by random integer offsets with zero padding."""
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if max_shift == 0:
        return images.clone()

    translated = torch.zeros_like(images)
    shifts = torch.randint(
        -max_shift,
        max_shift + 1,
        (images.shape[0], 2),
        generator=rng,
    )
    height, width = images.shape[-2:]
    for index, (shift_y, shift_x) in enumerate(shifts.tolist()):
        source_y = slice(max(0, -shift_y), min(height, height - shift_y))
        source_x = slice(max(0, -shift_x), min(width, width - shift_x))
        target_y = slice(max(0, shift_y), min(height, height + shift_y))
        target_x = slice(max(0, shift_x), min(width, width + shift_x))
        translated[index, :, target_y, target_x] = images[
            index, :, source_y, source_x
        ]
    return translated


def rotate_images(images: Tensor, angle_degrees: float) -> Tensor:
    """Rotate every image by the same angle for a visual example."""
    batch = images.shape[0]
    angles = images.new_full((batch,), float(angle_degrees))
    radians = torch.deg2rad(angles)
    theta = images.new_zeros((batch, 2, 3))
    theta[:, 0, 0] = torch.cos(radians)
    theta[:, 0, 1] = -torch.sin(radians)
    theta[:, 1, 0] = torch.sin(radians)
    theta[:, 1, 1] = torch.cos(radians)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def geometric_transform(
    images: Tensor,
    *,
    max_shift: float = 0.0,
    max_rotation: float = 0.0,
    rng: torch.Generator,
) -> Tensor:
    """Apply independent random translations and rotations without resizing."""
    if max_shift < 0.0 or max_rotation < 0.0:
        raise ValueError("Transform magnitudes must be non-negative")

    batch, _, height, width = images.shape
    angles = (2.0 * torch.rand(batch, generator=rng) - 1.0) * max_rotation
    shifts_x = (2.0 * torch.rand(batch, generator=rng) - 1.0) * max_shift
    shifts_y = (2.0 * torch.rand(batch, generator=rng) - 1.0) * max_shift
    radians = torch.deg2rad(angles)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)

    theta = torch.zeros(batch, 2, 3, dtype=images.dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    theta[:, 0, 2] = 2.0 * shifts_x / max(width - 1, 1)
    theta[:, 1, 2] = 2.0 * shifts_y / max(height - 1, 1)

    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _train_image_model(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    x_train: Tensor,
    y_train: Tensor,
    *,
    num_classes: int,
    training_steps: int,
    learning_rate: float,
    augment_training: bool,
    train_max_shift: float,
    train_max_rotation: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Train a fresh classifier with optional on-the-fly augmentation."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    model = model_class(num_classes=num_classes, **model_kwargs).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(training_steps):
        batch = x_train
        if augment_training:
            batch = geometric_transform(
                batch,
                max_shift=train_max_shift,
                max_rotation=train_max_rotation,
                rng=rng,
            )
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch.to(device)), y_train.to(device))
        loss.backward()
        optimizer.step()
    return model.eval()


def evaluate_geometric_generalization(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    x: Tensor,
    y: Tensor,
    *,
    num_classes: int,
    shift_levels: list[float],
    rotation_levels: list[float],
    augment_training: bool,
    train_max_shift: float,
    train_max_rotation: float,
    trials: int,
    transforms_per_trial: int,
    train_fraction: float,
    training_steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device | str | None = None,
) -> dict[str, dict[float, dict[str, float]] | dict[str, float]]:
    """Train fresh models and test clean, translated, and rotated images."""
    if trials < 1 or transforms_per_trial < 1 or training_steps < 1:
        raise ValueError("Trial, transform, and step counts must be positive")
    if any(value < 0 for value in shift_levels + rotation_levels):
        raise ValueError("Test transform magnitudes must be non-negative")

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x = x.detach().cpu()
    y = y.detach().cpu().long()
    clean_scores: list[float] = []
    shift_scores = {float(level): [] for level in shift_levels}
    rotation_scores = {float(level): [] for level in rotation_levels}

    for trial in range(trials):
        split_rng = torch.Generator().manual_seed(seed + trial)
        transform_rng = torch.Generator().manual_seed(seed + 100_000 + trial)
        train_idx, test_idx = stratified_split(y, train_fraction, split_rng)
        model = _train_image_model(
            model_class,
            model_kwargs,
            x[train_idx],
            y[train_idx],
            num_classes=num_classes,
            training_steps=training_steps,
            learning_rate=learning_rate,
            augment_training=augment_training,
            train_max_shift=train_max_shift,
            train_max_rotation=train_max_rotation,
            seed=seed + 200_000 + trial,
            device=run_device,
        )
        x_test = x[test_idx]
        y_test = y[test_idx].to(run_device)
        with torch.no_grad():
            clean_scores.append(
                float((model(x_test.to(run_device)).argmax(dim=-1) == y_test).float().mean())
            )

        for level in shift_levels:
            scores = []
            for _ in range(transforms_per_trial):
                if not float(level).is_integer():
                    raise ValueError("Translation levels must be integer pixel counts")
                transformed = random_integer_translation(
                    x_test,
                    max_shift=int(level),
                    rng=transform_rng,
                )
                with torch.no_grad():
                    predictions = model(transformed.to(run_device)).argmax(dim=-1)
                scores.append(float((predictions == y_test).float().mean()))
            shift_scores[float(level)].append(float(np.mean(scores)))

        for level in rotation_levels:
            scores = []
            for _ in range(transforms_per_trial):
                transformed = geometric_transform(
                    x_test, max_rotation=level, rng=transform_rng
                )
                with torch.no_grad():
                    predictions = model(transformed.to(run_device)).argmax(dim=-1)
                scores.append(float((predictions == y_test).float().mean()))
            rotation_scores[float(level)].append(float(np.mean(scores)))

    def summarize(values: list[float]) -> dict[str, float]:
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    return {
        "clean": summarize(clean_scores),
        "translation": {level: summarize(values) for level, values in shift_scores.items()},
        "rotation": {level: summarize(values) for level, values in rotation_scores.items()},
    }


def evaluate_translation_generalization(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    x: Tensor,
    y: Tensor,
    *,
    num_classes: int,
    shift_levels: list[int],
    trials: int,
    transforms_per_trial: int,
    train_fraction: float,
    training_steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device | str | None = None,
):
    """Train on clean images and test random integer translations only."""
    return evaluate_geometric_generalization(
        model_class,
        model_kwargs,
        x,
        y,
        num_classes=num_classes,
        shift_levels=[float(level) for level in shift_levels],
        rotation_levels=[],
        augment_training=False,
        train_max_shift=0.0,
        train_max_rotation=0.0,
        trials=trials,
        transforms_per_trial=transforms_per_trial,
        train_fraction=train_fraction,
        training_steps=training_steps,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )


def evaluate_rotation_generalization(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    x: Tensor,
    y: Tensor,
    *,
    num_classes: int,
    rotation_levels: list[float],
    trials: int,
    transforms_per_trial: int,
    train_fraction: float,
    training_steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device | str | None = None,
):
    """Train on clean images and test random rotations only."""
    return evaluate_geometric_generalization(
        model_class,
        model_kwargs,
        x,
        y,
        num_classes=num_classes,
        shift_levels=[],
        rotation_levels=rotation_levels,
        augment_training=False,
        train_max_shift=0.0,
        train_max_rotation=0.0,
        trials=trials,
        transforms_per_trial=transforms_per_trial,
        train_fraction=train_fraction,
        training_steps=training_steps,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )


def print_geometric_audit(name: str, summary) -> None:
    """Print geometric-test accuracies as mean plus-or-minus std."""
    print(name)
    print(f"  clean: {summary['clean']['mean']:.4f} +- {summary['clean']['std']:.4f}")
    for condition in ("translation", "rotation"):
        print(f"  {condition}")
        for level, metrics in summary[condition].items():
            print(f"    {level:g}: {metrics['mean']:.4f} +- {metrics['std']:.4f}")


def print_transform_audit(name: str, summary, condition: str) -> None:
    """Print one transformation audit without unrelated empty sections."""
    if condition not in {"translation", "rotation"}:
        raise ValueError("condition must be 'translation' or 'rotation'")
    print(name)
    for level, metrics in summary[condition].items():
        print(f"  {level:g}: {metrics['mean']:.4f} +- {metrics['std']:.4f}")

