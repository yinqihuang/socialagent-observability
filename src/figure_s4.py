#!/usr/bin/env python3
"""Reproduce Supplementary Figure S4: matched conversation/search channels."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from style import HUMAN, SIMULATION, SOURCE_MARKERS, manuscript_style, save_pdf, style_axes

METRICS = [("words_per_message", "Words per message"),
           ("information_actions", "Information actions"),
           ("evidence_acquired", "Canonical evidence acquired")]


def make_figure(data_path: Path, output_path: Path) -> None:
    d = pd.read_csv(data_path)
    colors = {"Human": HUMAN, "Simulation": SIMULATION}
    with manuscript_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
        for ax, (metric, label) in zip(axes, METRICS):
            z = d[d.metric == metric].set_index("source").loc[["Human", "Simulation"]]
            x = np.arange(2)
            for i, source in enumerate(z.index):
                ax.errorbar(i, z.loc[source, "mean"], yerr=z.loc[source, "sem"], fmt=SOURCE_MARKERS[source],
                            color=colors[source], ecolor=colors[source], markersize=5.2,
                            markeredgecolor="white", markeredgewidth=.65, elinewidth=1.05, capsize=2.6)
            ax.set_xticks(x, ["Human", "Simulation"], rotation=18)
            ax.set_ylabel(label); style_axes(ax, grid=True)
        fig.tight_layout(w_pad=1.0); save_pdf(fig, output_path); plt.close(fig)

