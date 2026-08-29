#!/usr/bin/env python3
"""Reproduce Supplementary Figure S2: AI-input transition coefficients."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from style import INFORMATION_NEED, INTENDED_ACTION, TEXT, UNCERTAINTY, manuscript_style, save_pdf, style_axes

OUTCOMES = ["dR", "dI", "dA", "dU", "delta_task_action_error", "delta_ai_action_error"]
LABELS = ["Δ urgency", "Δ information need", "Δ action", "Δ uncertainty",
          "Δ triage-reference distance", "Δ AI-advice distance"]
TERMS = [("direction_high", "Recommendation escalation", INTENDED_ACTION, "o"),
         ("confidence_high", "AI confidence", UNCERTAINTY, "s"),
         ("direction_x_confidence", "Escalation × confidence", INFORMATION_NEED, "^")]


def make_figure(data_path: Path, output_path: Path) -> None:
    d = pd.read_csv(data_path)
    with manuscript_style():
        fig, ax = plt.subplots(figsize=(6.6, 4.1))
        base = np.arange(len(OUTCOMES))
        for offset, (term, label, color, marker) in zip((-.18, 0, .18), TERMS):
            z = d[d.term == term].set_index("outcome").reindex(OUTCOMES)
            x = z.standardized_beta.to_numpy(); lo = z.ci_low.to_numpy(); hi = z.ci_high.to_numpy()
            ax.errorbar(x, base + offset, xerr=np.vstack([x-lo, hi-x]), fmt=marker,
                        color=color, ecolor=color, markersize=5.2, markeredgecolor="white",
                        markeredgewidth=.65, elinewidth=1.05, capsize=2.6, label=label)
        ax.axvline(0, color=TEXT, linewidth=.8, zorder=0)
        ax.set_yticks(base, LABELS); ax.invert_yaxis()
        ax.set_xlabel("Standardized coefficient (persona-clustered 95% CI)")
        ax.legend(frameon=False, loc="lower right")
        style_axes(ax, grid=True); ax.grid(axis="x", color="#D6DDD9", linewidth=.55, alpha=.88)
        fig.tight_layout(); save_pdf(fig, output_path); plt.close(fig)
