"""Visualization helpers shared by the toy experiments."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_sample_comparison(
    panels: Mapping[str, torch.Tensor],
    output_path: str | Path,
    losses: Sequence[float] | None = None,
    axis_limits: tuple[float, float] = (-6.0, 6.0),
) -> Path:
    """Save 2-D sample panels and, optionally, a training-loss panel."""
    if not panels:
        raise ValueError("panels must contain at least one sample tensor")

    panel_count = len(panels) + int(losses is not None)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(5 * panel_count, 4),
        squeeze=False,
    )
    axes = axes[0]

    for ax, (title, samples) in zip(
        axes[: len(panels)],
        panels.items(),
        strict=True,
    ):
        samples_cpu = samples.detach().cpu()
        if samples_cpu.ndim != 2 or samples_cpu.shape[1] != 2:
            raise ValueError(
                f"panel {title!r} must have shape (batch, 2), "
                f"got {tuple(samples_cpu.shape)}"
            )

        ax.scatter(samples_cpu[:, 0], samples_cpu[:, 1], s=3, alpha=0.5)
        ax.set_title(title)
        ax.set_xlim(*axis_limits)
        ax.set_ylim(*axis_limits)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    if losses is not None:
        loss_ax = axes[-1]
        loss_ax.plot(losses)
        loss_ax.set_title("CFM Training Loss")
        loss_ax.set_xlabel("Training step")
        loss_ax.set_ylabel("Squared error")
        loss_ax.grid(alpha=0.2)

    fig.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination
