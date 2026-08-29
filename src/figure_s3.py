#!/usr/bin/env python3
"""Reproduce Supplementary Figure S3: policy and terminal alignment."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from style import HUMAN, INFORMATION_NEED, INTENDED_ACTION, TEXT, add_panel_label, manuscript_style, save_pdf, style_axes


def make_figure(data_path: Path, output_path: Path) -> None:
    d = pd.read_csv(data_path)
    p = d[d.table == "policy"].pivot(index="action_class", columns="feature", values="standardized_log_odds")
    features = ["R_t", "I_t", "A_t", "U_t", "direction_high", "confidence_high"]
    p = p.reindex(columns=features)
    g = d[d.table == "alignment"].copy()
    with manuscript_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={"width_ratios": [1.35, 1]})
        vmax = max(1, np.nanmax(np.abs(p.to_numpy(float))))
        im = axes[0].imshow(p.to_numpy(float), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[0].set_xticks(range(6), ["R", "I", "A", "U", "Escalation", "Confidence"], rotation=32, ha="right")
        axes[0].set_yticks(range(len(p)), [x.replace("_", " ").title() for x in p.index])
        add_panel_label(axes[0], "a")
        cb = fig.colorbar(im, ax=axes[0], shrink=.76); cb.set_label("Standardized log-odds coefficient")
        labels = [f"{r.recommendation_direction.title()}\n{r.recommendation_confidence.title()} conf." for _, r in g.iterrows()]
        x = np.arange(len(g)); w=.34
        axes[1].bar(x-w/2, g.ai_alignment, w, color=INTENDED_ACTION, label="AI advice")
        axes[1].bar(x+w/2, g.task_alignment, w, color=HUMAN, label="Triage reference")
        axes[1].set_xticks(x, labels); axes[1].set_ylim(0, 1); axes[1].set_ylabel("Final-decision alignment")
        axes[1].legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            columnspacing=0.9,
            handletextpad=0.4,
            borderaxespad=0,
        ); style_axes(axes[1], grid=True)
        add_panel_label(axes[1], "b")
        fig.tight_layout(w_pad=1.2); save_pdf(fig, output_path); plt.close(fig)
