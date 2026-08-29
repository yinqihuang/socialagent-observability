"""Generate Figure 3: manipulation responsiveness and pooled observability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (
    DATA_LINEWIDTH,
    INFORMATION_NEED,
    INTENDED_ACTION,
    MARKER_EDGEWIDTH,
    MARKER_SIZE,
    REFERENCE_LINEWIDTH,
    STATE_COLORS,
    TEXT,
    UNCERTAINTY,
    WHITE,
    add_panel_label,
    manuscript_style,
    save_pdf,
    style_axes,
)


def make_figure(data_path: Path, output_path: Path) -> None:
    data = pd.read_csv(data_path)
    panel_a = data[data["panel"] == "a"].copy()
    panel_b = data[data["panel"] == "b"].copy()

    with manuscript_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.25, 2.60),
            gridspec_kw={"wspace": 0.38, "width_ratios": [1.0, 1.28]},
        )

        order_a = ["intended_action", "uncertainty", "information_need"]
        colors_a = {
            "intended_action": INTENDED_ACTION,
            "uncertainty": UNCERTAINTY,
            "information_need": INFORMATION_NEED,
        }
        a = panel_a.set_index("state").loc[order_a]
        y = np.arange(len(a))
        axes[0].barh(
            y,
            a["value"],
            color=[colors_a[s] for s in order_a],
            height=0.58,
            edgecolor="none",
        )
        axes[0].set_yticks(y, a["display_label"])
        axes[0].invert_yaxis()
        axes[0].axvline(0, color=TEXT, linewidth=REFERENCE_LINEWIDTH, zorder=0)
        axes[0].set_xlabel("Standardized mean difference")
        axes[0].set_xlim(-0.90, 1.75)
        for yi, value in zip(y, a["value"]):
            axes[0].text(
                value + (0.04 if value >= 0 else -0.04),
                yi,
                f"{value:+.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                color=TEXT,
                fontweight="bold",
                fontsize=7.3,
            )

        order_b = ["urgency", "information_need", "intended_action", "uncertainty"]
        labels_b = {
            "urgency": "Urgency",
            "information_need": "Info. need",
            "intended_action": "Action",
            "uncertainty": "Uncertainty",
        }
        markers = {"urgency": "o", "information_need": "s", "intended_action": "^", "uncertainty": "D"}
        for state in order_b:
            g = panel_b[panel_b["state"] == state].set_index("model")
            base = float(g.loc["context_only", "value"])
            full = float(g.loc["context_plus_behavior", "value"])
            delta = float(g.loc["context_plus_behavior", "delta_r2"])
            axes[1].plot(
                [0, 1],
                [base, full],
                color=STATE_COLORS[state],
                linewidth=DATA_LINEWIDTH,
                marker=markers[state],
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGEWIDTH,
                markeredgecolor=WHITE,
                zorder=3,
            )
            axes[1].text(
                1.055,
                full,
                f"{labels_b[state]}  $\\Delta$ {delta:+.3f}",
                va="center",
                color=TEXT,
                fontsize=6.7,
            )
        axes[1].set_xlim(-0.12, 1.72)
        axes[1].set_ylim(0.49, 0.83)
        axes[1].set_xticks([0, 1], ["Context only", "Context + behavior"])
        axes[1].set_ylabel("Grouped-CV $R^2$")

        for label, ax in zip(("a", "b"), axes):
            style_axes(ax, grid=True)
            add_panel_label(ax, label)

        save_pdf(fig, output_path)
        plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make_figure(root / "data" / "figure3.csv", root / "figures" / "figure3.pdf")
