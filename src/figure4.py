"""Generate Figure 4: domain-specific incremental observability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (
    CAP_SIZE,
    DOMAIN_COLORS,
    ERRORBAR_LINEWIDTH,
    MARKER_EDGEWIDTH,
    MARKER_SIZE,
    REFERENCE,
    REFERENCE_LINEWIDTH,
    WHITE,
    manuscript_style,
    save_pdf,
    style_axes,
)


def make_figure(data_path: Path, output_path: Path) -> None:
    data = pd.read_csv(data_path)
    states = ["urgency", "information_need", "intended_action", "uncertainty"]
    state_labels = ["Urgency", "Information need", "Intended action", "Uncertainty"]
    domains = [
        ("medical", "Medical", "o", -0.13),
        ("legal", "Legal", "s", 0.0),
        ("finance", "Finance", "^", 0.13),
    ]

    with manuscript_style():
        fig, ax = plt.subplots(figsize=(7.10, 2.75))
        x = np.arange(len(states))
        ax.axhline(0, color=REFERENCE, linewidth=REFERENCE_LINEWIDTH, zorder=1)
        for domain, label, marker, offset in domains:
            rows = data[data["domain"] == domain].set_index("state").loc[states]
            ax.errorbar(
                x + offset,
                rows["delta_r2"],
                yerr=rows["sem"],
                color=DOMAIN_COLORS[domain],
                marker=marker,
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGEWIDTH,
                markeredgecolor=WHITE,
                linestyle="none",
                elinewidth=ERRORBAR_LINEWIDTH,
                capsize=CAP_SIZE,
                capthick=ERRORBAR_LINEWIDTH,
                label=label,
                zorder=3,
            )
        ax.set_xticks(x, state_labels)
        ax.set_ylabel(r"$\Delta R^2$")
        ax.set_ylim(-0.025, 0.135)
        style_axes(ax, grid=True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
        save_pdf(fig, output_path)
        plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make_figure(root / "data" / "figure4.csv", root / "figures" / "figure4.pdf")
