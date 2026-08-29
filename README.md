# Anonymous reproducibility materials

This repository accompanies *Do Simulated Users Reveal Behaviorally Observable States During Human--LLM Joint Decision Making?* It contains analysis-ready simulated-user trajectories, aggregate/de-identified human summaries, deterministic plotting code, and validation checks for the manuscript figures.

## Scope and privacy

- `data/simulation/` contains simulated-agent trajectories and self-reported-state probes from the corrected four-persona run.
- `data/figure*.csv` contains the minimum derived values used in the figures. Human results are aggregate summaries only.
- Raw human conversations, participant identifiers, recruitment records, timestamps, credentials, and linkage files are excluded.

## Requirements

Python 3.12 is recommended. Install with either:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or `conda env create -f environment.yml`.

## Reproduce all manuscript figures

```bash
python src/make_all_figures.py
python checks/validate_outputs.py
```

The scripts write vector PDFs to `figures/`. Figures 3--6 and S1--S4 use one centralized visual system in `src/style.py`.

## Recompute the dynamic simulated-user analysis

```bash
python scripts/analysis/adapt_v8_to_matraix.py
```

Outputs are written to `results/dynamic_analysis/`. This analysis uses only the bundled simulated-agent tables and does not access human data.

## Figure mapping

| Manuscript figure | Script | Data |
|---|---|---|
| Figure 3 | `src/figure3.py` | `data/figure3.csv` |
| Figure 4 | `src/figure4.py` | `data/figure4.csv` |
| Figure 5 | `src/figure5.py` | `data/figure5.csv` |
| Figure 6 | `src/figure6.py` | `data/figure6.csv` |
| Figures S1--S4 | `src/figure_s1.py`--`figure_s4.py` | matching CSV files |

## Reproducibility notes

Plot generation is deterministic. The dynamic analysis uses seed `20260812` for bootstrap resampling. The public tables retain manuscript precision; `checks/validate_outputs.py` verifies focal values and the expected PDF outputs.
