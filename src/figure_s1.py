#!/usr/bin/env python3
"""Reproduce Supplementary Figure S1: simulated state trajectories."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from style import INFORMATION_NEED, INTENDED_ACTION, manuscript_style, save_pdf, style_axes

STATE = {"R": "Perceived urgency", "I": "Information need",
         "A": "Intended action", "U": "Uncertainty"}
DIR = {"lower": ("Lower recommendation", INTENDED_ACTION, "-"),
       "higher": ("Higher recommendation", INFORMATION_NEED, "--")}


def make_figure(data_path: Path, output_path: Path) -> None:
    d = pd.read_csv(data_path)
    with manuscript_style():
        fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.7), sharex=True, sharey=True)
        for ax, code in zip(axes.flat, STATE):
            title = STATE[code]
            for direction, (label, color, linestyle) in DIR.items():
                z = d[(d.state == code) & (d.direction == direction)].sort_values("progress")
                x, y = z.progress, z["mean"] * 100
                ax.plot(x, y, color=color, linestyle=linestyle, marker="o", markersize=4.5,
                        linewidth=1.55, markeredgecolor="white", markeredgewidth=.6, label=label)
                ax.fill_between(x, z.ci_low * 100, z.ci_high * 100, color=color, alpha=.13, linewidth=0)
            ax.set_title(title, fontweight="bold")
            ax.set_ylim(0, 100); ax.set_xticks([0, .25, .5, .75, 1])
            style_axes(ax, grid=True)
        axes[0, 0].legend(frameon=False, loc="best")
        for ax in axes[:, 0]: ax.set_ylabel("Self-reported state")
        for ax in axes[1, :]: ax.set_xlabel("Normalized dialogue progress")
        fig.tight_layout(w_pad=1.2, h_pad=1.0)
        save_pdf(fig, output_path); plt.close(fig)
