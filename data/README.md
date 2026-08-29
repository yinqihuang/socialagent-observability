# Release data for Figures 3–6 and S1–S4

All files are tidy, deterministic, manuscript-facing tables. They are derived from the verified corrected simulation outputs and the completed human-data audit used by the current manuscript.

## `figure3.csv`

One row per plotted manipulation effect in panel **a**, and one row per state/model point in panel **b**.

- `panel`: `a` or `b`
- `state`: reported-state concept
- `display_label`: reader-facing panel-a label
- `manipulation`: analysis factor for panel a
- `model`: `context_only` or `context_plus_behavior` for panel b
- `value`: standardized mean difference (panel a) or grouped-CV R² (panel b)
- `delta_r2`: context-plus-behavior minus context-only R² (panel b)
- `n`: simulation trials for panel a or aligned turn observations for panel b

Values are aggregated/derived; no individual simulated-user records are included.

## `figure4.csv`

One row per domain × state estimate (4 states × 3 domains).

- `domain`: medical, legal, or finance
- `state`: urgency, information need, intended action, or uncertainty
- `delta_r2`: context-plus-behavior minus context-only grouped-CV R²
- `sem`: SEM of the increment across held-out persona groups
- `personas`, `trials`, `aligned_turns`: analysis counts

The figure shows `delta_r2 ± sem`. No pooled points are plotted.

## `figure5.csv`

One row per source × case reference × state × turn aggregate for the plotted turn range 0–6.

- `source`: Human or Simulation
- `case_reference`: Self-care or Calling an ambulance
- `state`: urgency, information need, intended action, or uncertainty
- `turn_index`: aligned interaction turn, beginning at 0
- `n`: observations contributing to the point
- `mean`: mean reported state on the 0–100 scale
- `sem`: standard error of the mean; blank when undefined (for example, `n=1`)

Human data are aggregated and de-identified. Later human turns remain outside this release because the manuscript figure intentionally aligns both sources through turn 6.

## `figure6.csv`

One row per source × recommendation direction × AI confidence × outcome aggregate for the matched viral-gastroenteritis factorial subset.

- `source`: Human or Simulation
- `direction`: lower or higher recommendation escalation
- `confidence`: low or high AI confidence
- `outcome`: final action or change in information need
- `mean`, `sem`, `n`: cell mean, standard error, and number of trajectories

The x-axis cells are categorical and are not connected by lines. Human and Simulation values are offset slightly within each cell.

## Filtering and SEM

The tables preserve the manuscript's existing completion/exclusion rules and actual delivered recommendation exposure. Figure 5 is limited to turns 0–6. Figure 6 uses only the case-matched viral-gastroenteritis factorial subset. Human SEMs are computed over the observations contributing to each plotted aggregate; Figure 4 SEMs summarize variation across held-out persona groups.

## Supplementary figures

- `figure_s1.csv`: reported state × recommendation level × normalized progress-bin means, bootstrap intervals, and counts from the verified V8 simulation analysis.
- `figure_s2.csv`: standardized transition/convergence coefficients with persona-clustered uncertainty for each experimental input and outcome.
- `figure_s3.csv`: dialogue-policy log-odds coefficients and terminal AI-advice/triage-reference alignment proportions, distinguished by `table`.
- `figure_s4.csv`: source × interaction-channel means and SEMs for 21 recommendation-exposed human and 16 simulated viral-gastroenteritis trajectories. It contains no identifiers or free text.
