#!/usr/bin/env python3
"""Generate and validate all public manuscript figure outputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from figure3 import make_figure as make_figure3
from figure4 import make_figure as make_figure4
from figure5 import make_figure as make_figure5
from figure6 import make_figure as make_figure6
from figure_s1 import make_figure as make_figure_s1
from figure_s2 import make_figure as make_figure_s2
from figure_s3 import make_figure as make_figure_s3
from figure_s4 import make_figure as make_figure_s4


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def main() -> None:
    make_figure3(DATA / "figure3.csv", FIGURES / "figure3.pdf")
    make_figure4(DATA / "figure4.csv", FIGURES / "figure4.pdf")
    make_figure5(DATA / "figure5.csv", FIGURES / "figure5.pdf")
    make_figure6(DATA / "figure6.csv", FIGURES / "figure6.pdf")
    make_figure_s1(DATA / "figure_s1.csv", FIGURES / "figure_s1.pdf")
    make_figure_s2(DATA / "figure_s2.csv", FIGURES / "figure_s2.pdf")
    make_figure_s3(DATA / "figure_s3.csv", FIGURES / "figure_s3.pdf")
    make_figure_s4(DATA / "figure_s4.csv", FIGURES / "figure_s4.pdf")
    subprocess.run([sys.executable, str(ROOT / "checks" / "validate_outputs.py")], check=True)
    print(f"Generated and validated Figures 3--6 and S1--S4 in {FIGURES}")


if __name__ == "__main__":
    main()
