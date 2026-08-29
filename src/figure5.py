"""Generate Figure 5: matched human and simulated state trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from style import (
    CAP_SIZE,
    DATA_LINEWIDTH,
    ERRORBAR_LINEWIDTH,
    INFORMATION_NEED,
    INTENDED_ACTION,
    MARKER_EDGEWIDTH,
    MARKER_SIZE,
    SOURCE_LINESTYLES,
    TEXT,
    TITLE_SIZE,
    WHITE,
    manuscript_style,
    save_pdf,
    style_axes,
)


CASE_COLORS = {"Self-care": INFORMATION_NEED, "Calling an ambulance": INTENDED_ACTION}
SOURCE_MARKERS = {"Human": "^", "Simulation": "s"}


def make_figure(data_path: Path, output_path: Path) -> None:
    data = pd.read_csv(data_path)
    states = [
        ("urgency", "Perceived urgency"),
        ("information_need", "Information need"),
        ("intended_action", "Intended action"),
        ("uncertainty", "Uncertainty"),
    ]
    sources = ["Human", "Simulation"]
    cases = ["Self-care", "Calling an ambulance"]

    with manuscript_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.10, 4.70), sharex=True, sharey=True)
        for ax, (state, title) in zip(axes.flat, states):
            for source in sources:
                for case in cases:
                    g = data[
                        (data["source"] == source)
                        & (data["case_reference"] == case)
                        & (data["state"] == state)
                    ].sort_values("turn_index")
                    ax.errorbar(
                        g["turn_index"],
                        g["mean"],
                        yerr=g["sem"],
                        color=CASE_COLORS[case],
                        linestyle=SOURCE_LINESTYLES[source],
                        marker=SOURCE_MARKERS[source],
                        markersize=MARKER_SIZE - 0.8,
                        markeredgewidth=MARKER_EDGEWIDTH,
                        markeredgecolor=WHITE,
                        linewidth=DATA_LINEWIDTH,
                        elinewidth=ERRORBAR_LINEWIDTH,
                        capsize=CAP_SIZE,
                        capthick=ERRORBAR_LINEWIDTH,
                        label=f"{source}: {case}",
                        zorder=3,
                    )
            ax.set_title(title, fontsize=TITLE_SIZE, color=TEXT)
            ax.set_ylim(0, 100)
            style_axes(ax, grid=True)

        for ax in axes[1, :]:
            ax.set_xlabel("Turn")
        axes[0, 0].set_ylabel("Reported state")
        axes[1, 0].set_ylabel("Reported state")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        save_pdf(fig, output_path)
        plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make_figure(root / "data" / "figure5.csv", root / "figures" / "figure5.pdf")
