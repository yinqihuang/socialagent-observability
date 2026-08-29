"""Shared, deterministic visual system for manuscript Figures 3--6."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl


# Typography and geometry
FONT_FAMILY = "DejaVu Sans"
BASE_FONT_SIZE = 8.0
AXIS_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.4
LEGEND_SIZE = 7.0
PANEL_LABEL_SIZE = 10.0
TITLE_SIZE = 8.5

AXIS_LINEWIDTH = 0.8
DATA_LINEWIDTH = 1.55
ERRORBAR_LINEWIDTH = 1.05
MARKER_EDGEWIDTH = 0.65
MARKER_SIZE = 5.2
CAP_SIZE = 2.6
REFERENCE_LINEWIDTH = 0.8

RASTER_DPI = 300
GRID_LINEWIDTH = 0.55
GRID_ALPHA = 0.88

# Manuscript palette
TEXT = "#6F7A73"
URGENCY = "#6F7A73"
INFORMATION_NEED = "#A980C3"
UNCERTAINTY = "#8CCB98"
INTENDED_ACTION = "#E3C770"

HUMAN = "#A980C3"
SIMULATION = "#6F7A73"

MEDICAL = "#A980C3"
LEGAL = "#8CCB98"
FINANCE = "#E3C770"

REFERENCE = "#AEB8B2"
GRID = "#D6DDD9"
WHITE = "#FFFFFF"

STATE_COLORS = {
    "urgency": URGENCY,
    "information_need": INFORMATION_NEED,
    "intended_action": INTENDED_ACTION,
    "uncertainty": UNCERTAINTY,
}

DOMAIN_COLORS = {"medical": MEDICAL, "legal": LEGAL, "finance": FINANCE}
SOURCE_COLORS = {"Human": HUMAN, "Simulation": SIMULATION}
SOURCE_MARKERS = {"Human": "o", "Simulation": "s"}
SOURCE_LINESTYLES = {"Human": "-", "Simulation": "--"}


@contextmanager
def manuscript_style():
    """Apply local rc settings without leaking global state across scripts."""
    rc = {
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT,
        "axes.linewidth": AXIS_LINEWIDTH,
        "axes.titlesize": TITLE_SIZE,
        "axes.titlecolor": TEXT,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "legend.fontsize": LEGEND_SIZE,
        "text.color": TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
    }
    with mpl.rc_context(rc):
        yield


def style_axes(ax, *, grid: bool = False) -> None:
    """Apply the shared spine and optional Figure 6 grid treatment."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
        ax.spines[side].set_color(TEXT)
    ax.tick_params(width=AXIS_LINEWIDTH, length=3)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(
            axis="y",
            which="major",
            color=GRID,
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
            zorder=0,
        )
    else:
        ax.grid(False)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="bottom",
    )


def save_pdf(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"CreationDate": None, "ModDate": None},
    )
