"""Generate Figure 6: categorical recommendation-by-confidence comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (
    CAP_SIZE,
    ERRORBAR_LINEWIDTH,
    MARKER_EDGEWIDTH,
    MARKER_SIZE,
    SOURCE_COLORS,
    SOURCE_MARKERS,
    WHITE,
    manuscript_style,
    save_pdf,
    style_axes,
)


def make_figure(data_path: Path, output_path: Path) -> None:
    data = pd.read_csv(data_path)
    cells = [("lower", "low"), ("lower", "high"), ("higher", "low"), ("higher", "high")]
    labels = ["Lower\nlow", "Lower\nhigh", "Higher\nlow", "Higher\nhigh"]
    outcomes = [("final_action", "Final action"), ("information_need_change", "Change in information need")]
    sources = ["Human", "Simulation"]

    with manuscript_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.20))
        x = np.arange(len(cells))
        for ax, (outcome, ylabel) in zip(axes, outcomes):
            for index, source in enumerate(sources):
                g = data[(data["source"] == source) & (data["outcome"] == outcome)].set_index(
                    ["direction", "confidence"]
                )
                values = [float(g.loc[cell, "mean"]) for cell in cells]
                sem = [float(g.loc[cell, "sem"]) for cell in cells]
                offset = -0.07 if index == 0 else 0.07
                ax.errorbar(
                    x + offset,
                    values,
                    yerr=sem,
                    color=SOURCE_COLORS[source],
                    marker=SOURCE_MARKERS[source],
                    markersize=MARKER_SIZE,
                    markeredgewidth=MARKER_EDGEWIDTH,
                    markeredgecolor=WHITE,
                    linestyle="none",
                    elinewidth=ERRORBAR_LINEWIDTH,
                    capsize=CAP_SIZE,
                    capthick=ERRORBAR_LINEWIDTH,
                    label=source,
                    zorder=3,
                )
            ax.set_xticks(x, labels)
            ax.set_ylabel(ylabel)
            style_axes(ax, grid=True)
        axes[0].legend(frameon=False)
        fig.tight_layout()
        save_pdf(fig, output_path)
        plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make_figure(root / "data" / "figure6.csv", root / "figures" / "figure6.pdf")

