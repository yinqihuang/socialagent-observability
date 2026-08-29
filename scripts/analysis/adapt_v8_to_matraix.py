#!/usr/bin/env python3
"""Adapt the latest optimal-control analysis (V8) to corrected MatrAIx Run 2.

This is a parallel, simulated-agent analysis. It does not alter the V8 human
analysis and it never writes into participant-data directories.
"""
from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "dynamic_analysis"
ANALYSIS = ROOT / "data" / "simulation"
FIG = OUT / "figures"
TABLE = OUT / "tables"
MODEL = OUT / "models"
for directory in (FIG, TABLE, MODEL):
    directory.mkdir(parents=True, exist_ok=True)

STATE_NAMES = {
    "R": "Perceived urgency",
    "I": "Information need",
    "A": "Intended action",
    "U": "Uncertainty",
}
INPUT_LABELS = {
    "direction_high": "Higher recommendation",
    "confidence_high": "High confidence",
    "direction_x_confidence": "Direction × confidence",
}


def standardize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if pd.notna(sd) and sd > 0 else s * 0


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    turns = pd.read_parquet(ANALYSIS / "run2_turn_trajectories.parquet")
    probes = pd.read_parquet(ANALYSIS / "run2_state_probes.parquet")
    manifest = pd.read_csv(ANALYSIS / "run2_trial_manifest.csv", dtype={"persona_id": str})
    # V8 needs a state after every intervention. Restrict to the prespecified
    # full-probe cohort; sparse/endpoint schedules are not interpolated.
    full_ids = set(manifest.loc[manifest.probe_schedule == "full", "trial_id"])
    turns = turns[turns.trial_id.isin(full_ids)].copy()
    probes = probes[probes.trial_id.isin(full_ids)].copy()
    manifest = manifest[manifest.trial_id.isin(full_ids)].copy()
    return turns, probes, manifest


def action_class(row: pd.Series) -> str:
    if int(row.get("continue_or_stop", 0)) == 1:
        return "stop"
    if bool(row.get("repeated_information_seeking", False)):
        return "repeat_information"
    if pd.notna(row.get("information_cluster_selected")) and str(row.get("information_cluster_selected")):
        return "seek_information"
    if int(row.get("asks_question", 0)):
        return "ask_without_reveal"
    return "continue_other"


def build_turn_and_transition(
    turns: pd.DataFrame, probes: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = probes.copy()
    p["R"] = p.perceived_urgency / 100
    p["I"] = p.information_needed / 100
    p["A"] = p.intended_action / 100
    p["U"] = 1 - p.confidence / 100
    p = p.sort_values(["trial_id", "turn_index", "state_index"])

    t = turns.copy()
    t["action_class"] = t.apply(action_class, axis=1)
    t["direction_high"] = (t.recommendation_direction == "higher").astype(int)
    t["confidence_high"] = (t.recommendation_confidence == "high").astype(int)
    t["direction_x_confidence"] = t.direction_high * t.confidence_high
    t["triage_target"] = t.reference_action_0_100 / 100
    t["urgency_target"] = np.where(t.reference_action_0_100 == 100, .90, .20)
    t["ai_target"] = t.ai_recommended_action / 100

    post = p[["trial_id", "turn_index", "R", "I", "A", "U"]].drop_duplicates(
        ["trial_id", "turn_index"], keep="last"
    )
    turn_level = t.merge(post, on=["trial_id", "turn_index"], how="inner")

    pre = p[["trial_id", "turn_index", "R", "I", "A", "U"]].copy()
    pre["turn_index"] = pre.turn_index + 1
    pre = pre.rename(columns={x: f"{x}_t" for x in STATE_NAMES})
    nxt = post.rename(columns={x: f"{x}_next" for x in STATE_NAMES})
    trans = t.merge(pre, on=["trial_id", "turn_index"], how="inner").merge(
        nxt, on=["trial_id", "turn_index"], how="inner"
    )
    for state in STATE_NAMES:
        trans[f"d{state}"] = trans[f"{state}_next"] - trans[f"{state}_t"]
    trans["triage_action_error_t"] = abs(trans.A_t - trans.triage_target)
    trans["triage_action_error_next"] = abs(trans.A_next - trans.triage_target)
    trans["delta_triage_action_error"] = trans.triage_action_error_next - trans.triage_action_error_t
    trans["ai_action_error_t"] = abs(trans.A_t - trans.ai_target)
    trans["ai_action_error_next"] = abs(trans.A_next - trans.ai_target)
    trans["delta_ai_action_error"] = trans.ai_action_error_next - trans.ai_action_error_t
    trans["urgency_error_t"] = abs(trans.R_t - trans.urgency_target)
    trans["urgency_error_next"] = abs(trans.R_next - trans.urgency_target)
    trans["delta_urgency_error"] = trans.urgency_error_next - trans.urgency_error_t
    trans["new_evidence"] = trans.number_of_evidence_items_acquired.groupby(trans.trial_id).diff().fillna(
        trans.number_of_evidence_items_acquired
    )
    trans["new_information_action"] = trans.number_of_information_actions.groupby(trans.trial_id).diff().fillna(
        trans.number_of_information_actions
    )
    return turn_level, trans


def case_table(turn: pd.DataFrame, probes: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    first = probes.sort_values("state_index").groupby("trial_id").first()
    last = probes.sort_values("state_index").groupby("trial_id").last()
    agg = turn.sort_values("turn_index").groupby("trial_id").agg(
        n_rounds=("turn_index", "max"),
        information_actions=("number_of_information_actions", "max"),
        evidence_acquired=("number_of_evidence_items_acquired", "max"),
        evidence_remaining=("evidence_remaining", "last"),
        stop_observed=("continue_or_stop", "max"),
        mean_latency_ms=("latency_ms", "mean"),
    )
    out = manifest.set_index("trial_id").join(agg)
    for source, dest in [
        ("perceived_urgency", "R"), ("information_needed", "I"),
        ("intended_action", "A"), ("confidence", "confidence")
    ]:
        out[f"start_{dest}"] = first[source] / 100
        out[f"final_{dest}"] = last[source] / 100
        out[f"total_d{dest}"] = out[f"final_{dest}"] - out[f"start_{dest}"]
    out["final_U"] = 1 - out.final_confidence
    out["start_U"] = 1 - out.start_confidence
    out["total_dU"] = out.final_U - out.start_U
    out["triage_target"] = out.reference_action_0_100 / 100
    out["ai_target"] = np.where(out.recommendation_direction == "higher", 1., 0.)
    out["triage_error_reduction"] = abs(out.start_A - out.triage_target) - abs(out.final_A - out.triage_target)
    out["ai_error_reduction"] = abs(out.start_A - out.ai_target) - abs(out.final_A - out.ai_target)
    out["final_decision_binary"] = (out.final_A >= .5).astype(int)
    out["triage_alignment"] = (out.final_decision_binary == (out.triage_target == 1)).astype(int)
    out["ai_alignment"] = (out.final_decision_binary == (out.ai_target == 1)).astype(int)
    return out.reset_index()


def design(df: pd.DataFrame, state_cols: list[str], include_action: bool = True) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for col in state_cols + [
        "direction_high", "confidence_high", "direction_x_confidence",
        "triage_target", "turn_index", "new_evidence", "repeated_information_seeking"
    ]:
        x[col] = pd.to_numeric(df[col], errors="coerce")
    if include_action:
        x = pd.concat([x, pd.get_dummies(df.action_class, prefix="action", drop_first=True, dtype=float)], axis=1)
    x = pd.concat([
        x,
        pd.get_dummies(df.domain, prefix="domain", drop_first=True, dtype=float),
        pd.get_dummies(df.persona_id, prefix="persona", drop_first=True, dtype=float),
    ], axis=1)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    return x.loc[:, x.nunique() > 1]


def fit_ols_models(trans: pd.DataFrame) -> pd.DataFrame:
    outcomes = ["dR", "dI", "dA", "dU", "delta_triage_action_error", "delta_ai_action_error", "delta_urgency_error"]
    rows = []
    for outcome in outcomes:
        state_cols = ["R_t", "I_t", "A_t", "U_t"]
        if outcome == "delta_triage_action_error": state_cols += ["triage_action_error_t"]
        if outcome == "delta_ai_action_error": state_cols += ["ai_action_error_t"]
        if outcome == "delta_urgency_error": state_cols += ["urgency_error_t"]
        x = design(trans, state_cols)
        # Standardized coefficients make human/agent effect directions and
        # magnitudes commensurable while retaining V8's first-order model.
        xs = x.apply(standardize)
        y = standardize(trans[outcome])
        fit = sm.OLS(y, sm.add_constant(xs, has_constant="add")).fit()
        robust = (fit.get_robustcov_results(cov_type="cluster", groups=trans.persona_id)
                  if trans.persona_id.nunique() >= 2 else fit.get_robustcov_results(cov_type="HC3"))
        names = ["const", *xs.columns]
        ci = robust.conf_int()
        for i, term in enumerate(names):
            rows.append({
                "outcome": outcome, "term": term, "standardized_beta": robust.params[i],
                "std_err": robust.bse[i], "p_value": robust.pvalues[i],
                "ci_low": ci[i, 0], "ci_high": ci[i, 1], "n": len(trans),
                "personas": trans.persona_id.nunique(), "r_squared": fit.rsquared,
                "se_type": (f"persona-clustered ({trans.persona_id.nunique()} clusters; interpret cautiously)"
                            if trans.persona_id.nunique() >= 2 else "HC3 heteroskedasticity-robust (one persona; descriptive only)"),
            })
    result = pd.DataFrame(rows)
    result.to_csv(MODEL / "transition_and_convergence_coefficients.csv", index=False)
    return result


def fit_policy(trans: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = ["R_t", "I_t", "A_t", "U_t", "direction_high", "confidence_high", "triage_target", "turn_index"]
    x = pd.concat([
        trans[features].astype(float),
        pd.get_dummies(trans.domain, prefix="domain", drop_first=True, dtype=float),
    ], axis=1)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    y = trans.action_class
    clf = LogisticRegression(max_iter=4000, class_weight="balanced").fit(xs, y)
    rows = []
    for cls, coefs in zip(clf.classes_, clf.coef_):
        for feature, coef in zip(x.columns, coefs):
            rows.append({"action_class": cls, "feature": feature, "standardized_log_odds": coef})
    result = pd.DataFrame(rows)
    result.to_csv(MODEL / "dialogue_policy_coefficients.csv", index=False)
    summary = {"n": len(y), "classes": list(clf.classes_), "training_accuracy": accuracy_score(y, clf.predict(xs))}
    (MODEL / "dialogue_policy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return result, summary


def bootstrap_trajectory(turn: pd.DataFrame, state: str, n_boot: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(20260812)
    d = turn.copy()
    d["progress"] = d.groupby("trial_id").turn_index.transform(lambda s: (s - 1) / max(1, s.max() - 1))
    d["bin"] = np.clip(np.round(d.progress * 4).astype(int), 0, 4)
    rows = []
    personas = d.persona_id.unique()
    for direction in ["lower", "higher"]:
        for b in range(5):
            sub = d[(d.recommendation_direction == direction) & (d.bin == b)]
            value = sub[state].mean()
            boots = []
            persona_sums = sub.groupby("persona_id")[state].agg(["sum", "count"])
            sums = persona_sums["sum"].reindex(personas, fill_value=0).to_numpy(float)
            counts = persona_sums["count"].reindex(personas, fill_value=0).to_numpy(float)
            for _ in range(n_boot):
                idx = rng.integers(0, len(personas), len(personas))
                denom = counts[idx].sum()
                boots.append(sums[idx].sum() / denom if denom else np.nan)
            lo, hi = np.nanquantile(boots, [.025, .975])
            rows.append({"state": state, "direction": direction, "progress": b / 4, "mean": value, "ci_low": lo, "ci_high": hi, "n": len(sub)})
    return pd.DataFrame(rows)


def make_figure1(turn: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3), sharex=True)
    summaries = []
    for ax, state in zip(axes.flat, ["R", "I", "A", "U"]):
        s = bootstrap_trajectory(turn, state)
        summaries.append(s)
        for direction, color in [("lower", "#2878B5"), ("higher", "#D9534F")]:
            z = s[s.direction == direction]
            ax.plot(z.progress, z["mean"] * 100, marker="o", color=color, label=direction.title())
            ax.fill_between(z.progress, z.ci_low * 100, z.ci_high * 100, color=color, alpha=.16)
        ax.set_title(STATE_NAMES[state])
        ax.set_ylim(0, 100); ax.grid(alpha=.2); ax.set_xlabel("Normalized trial progress"); ax.set_ylabel("Reported state (0–100)")
    axes[0, 0].legend(title="AI direction", frameon=False)
    fig.suptitle("V8-style reported decision-state trajectories in persona agents")
    fig.tight_layout()
    fig.savefig(FIG / "Figure1_state_trajectories.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "Figure1_state_trajectories.pdf", bbox_inches="tight")
    pd.concat(summaries).to_csv(TABLE / "Figure1_state_trajectory_data.csv", index=False)
    plt.close(fig)


def make_figure2(coefs: pd.DataFrame) -> None:
    outcomes = ["dR", "dI", "dA", "dU", "delta_triage_action_error", "delta_ai_action_error"]
    labels = ["Δ urgency", "Δ information need", "Δ action", "Δ uncertainty", "Δ triage-reference distance", "Δ AI-target distance"]
    inputs = list(INPUT_LABELS)
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for j, term in enumerate(inputs):
        z = coefs[(coefs.term == term) & coefs.outcome.isin(outcomes)].set_index("outcome").reindex(outcomes)
        y = np.arange(len(outcomes)) + (j - 1) * .18
        ax.errorbar(z.standardized_beta, y, xerr=[z.standardized_beta-z.ci_low, z.ci_high-z.standardized_beta], fmt=["o","s","^"][j], capsize=2.5, label=INPUT_LABELS[term])
    ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(np.arange(len(outcomes)), labels)
    se_label = ("persona-clustered 95% CI" if int(coefs.personas.max()) >= 2
                else "HC3 robust 95% CI (one persona)")
    ax.invert_yaxis(); ax.set_xlabel(f"Standardized coefficient ({se_label})")
    ax.set_title("V8-style AI-input associations with state change and convergence")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(FIG / "Figure2_AI_influence_forest.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / "Figure2_AI_influence_forest.pdf", bbox_inches="tight"); plt.close(fig)


def make_figure3(policy: pd.DataFrame, cases: pd.DataFrame) -> None:
    wanted = ["R_t", "I_t", "A_t", "U_t", "direction_high", "confidence_high"]
    p = policy[policy.feature.isin(wanted)].pivot(index="action_class", columns="feature", values="standardized_log_odds").reindex(columns=wanted)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.35, 1]})
    im = axes[0].imshow(p.values, cmap="coolwarm", vmin=-max(1, np.nanmax(abs(p.values))), vmax=max(1, np.nanmax(abs(p.values))), aspect="auto")
    axes[0].set_xticks(range(len(wanted)), ["R", "I", "A", "U", "Direction", "Confidence"], rotation=35, ha="right")
    axes[0].set_yticks(range(len(p)), [x.replace("_", " ").title() for x in p.index]); axes[0].set_title("A. Descriptive dialogue policy")
    fig.colorbar(im, ax=axes[0], shrink=.75, label="Standardized log-odds coefficient")
    g = cases.groupby(["recommendation_direction", "recommendation_confidence"])[["ai_alignment", "triage_alignment"]].mean().reset_index()
    x = np.arange(len(g)); w=.36
    axes[1].bar(x-w/2, g.ai_alignment, w, label="Aligned with AI", color="#D9534F")
    axes[1].bar(x+w/2, g.triage_alignment, w, label="Aligned with triage reference", color="#2878B5")
    axes[1].set_xticks(x, [f"{d.title()}\n{c.title()} conf." for d,c in zip(g.recommendation_direction,g.recommendation_confidence)])
    axes[1].set_ylim(0,1); axes[1].set_ylabel("Proportion"); axes[1].set_title("B. Final decision alignment"); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Persona-agent policy and terminal decisions")
    fig.tight_layout(); fig.savefig(FIG / "Figure3_policy_and_final_decision.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / "Figure3_policy_and_final_decision.pdf", bbox_inches="tight"); plt.close(fig)
    g.to_csv(TABLE / "final_alignment_by_condition.csv", index=False)


def write_report(turn: pd.DataFrame, trans: pd.DataFrame, cases: pd.DataFrame, coefs: pd.DataFrame, policy_summary: dict) -> None:
    focal = coefs[coefs.term.isin(INPUT_LABELS)].copy()
    def beta(outcome: str, term: str) -> str:
        z=focal[(focal.outcome==outcome)&(focal.term==term)].iloc[0]
        return f"{z.standardized_beta:+.3f} [{z.ci_low:+.3f}, {z.ci_high:+.3f}]"
    n_personas = cases.persona_id.nunique()
    run_label = "corrected four-persona simulation"
    ci_label = (f"persona-clustered 95% confidence intervals ({n_personas} clusters)"
                if n_personas >= 2 else "HC3 heteroskedasticity-robust 95% confidence intervals (one persona; descriptive only)")
    persona_limit = ("It cannot estimate between-person persona effects when only one persona is used."
                     if n_personas < 2 else
                     f"With only {n_personas} persona clusters, between-person and moderator estimates remain descriptive and unstable.")
    text = f"""# V8 optimal-control analysis adapted to MatrAIx {run_label}

## Scope and provenance

This analysis applies the latest V8 framework at commit `f72129177ab20ba8575d5f41f7881e33e925df8f` to the {run_label} persona-agent data. It uses only full-probe trials, is labeled `matraix_simulated_agent`, and does not modify or mix with human-participant records.

## Mapping to V8

- State: `X_t = [R_t, I_t, A_t, U_t]`, with `U_t = 1 - confidence_t`.
- AI input: randomized recommendation Direction, randomized Confidence, and their interaction. These replace V8's observational text-coded `q/s/m` inputs because {run_label} experimentally controls direction/confidence but does not randomize separate information-request or sensemaking intensities.
- Transition timing: baseline/post-turn state `X_(t-1)` → persona information action + AI response → post-turn state `X_t`.
- Task convergence: absolute intended-action distance to the fixed study reference action.
- AI convergence: absolute intended-action distance to the randomized AI recommendation. This is reported separately because AI advice can conflict with the case reference.
- Policy: next/current observable action class from the pre-action state and experimental input.

## Analysis sample

- Personas: {cases.persona_id.nunique()}
- Full-probe trials: {cases.trial_id.nunique()}
- Domains: {cases.domain.nunique()}
- Cases: {cases.case_id.nunique()}
- Post-turn observations: {len(turn)}
- Temporally aligned transitions (including baseline → turn 1): {len(trans)}

## Selected transition estimates

Standardized coefficients with {ci_label}:

| Outcome | Higher recommendation | High confidence | Interaction |
|---|---:|---:|---:|
| Δ urgency | {beta('dR','direction_high')} | {beta('dR','confidence_high')} | {beta('dR','direction_x_confidence')} |
| Δ information need | {beta('dI','direction_high')} | {beta('dI','confidence_high')} | {beta('dI','direction_x_confidence')} |
| Δ intended action | {beta('dA','direction_high')} | {beta('dA','confidence_high')} | {beta('dA','direction_x_confidence')} |
| Δ uncertainty | {beta('dU','direction_high')} | {beta('dU','confidence_high')} | {beta('dU','direction_x_confidence')} |
| Δ triage-reference distance | {beta('delta_triage_action_error','direction_high')} | {beta('delta_triage_action_error','confidence_high')} | {beta('delta_triage_action_error','direction_x_confidence')} |
| Δ AI-target distance | {beta('delta_ai_action_error','direction_high')} | {beta('delta_ai_action_error','confidence_high')} | {beta('delta_ai_action_error','direction_x_confidence')} |

Negative distance coefficients indicate movement toward the relevant target. These intervals must be treated as descriptive stress-test uncertainty, not human inference.

## Policy and terminal outcomes

The descriptive regularized action-policy model used {policy_summary['n']} transitions, {len(policy_summary['classes'])} action classes, and had in-sample accuracy {policy_summary['training_accuracy']:.3f}. In-sample accuracy is not a generalization estimate. Final triage-reference alignment was {cases.triage_alignment.mean():.3f}; final AI-direction alignment was {cases.ai_alignment.mean():.3f}.

## Can this fit the paper storyline?

Yes—as a complementary dynamic validity analysis, not as a replacement for the paper's main incremental-observability result. The current paper asks whether reported state is expressed in behavior beyond task context. V8 asks a different but adjacent question: how reported state changes after controlled AI inputs, how the state predicts the next observable information action, and whether action moves toward the task or AI target.

The clean human comparison is prospective and side-by-side:

1. Fit the same timing-aligned state-transition model separately in agents and humans.
2. Compare standardized Direction/Confidence coefficients, transition signs, trajectory shapes, and target-distance changes; do not pool turns as independent observations.
3. Test transport with a dataset indicator × input interaction in a harmonized model, while clustering on persona for agents and participant for humans.
4. Retain behavioral observability (context-only versus context+behavior ΔR²) as the primary simulation-validity criterion; use V8 as the dynamic response/convergence analysis.

This supports a coherent paper arc: simulated users generate a selective state–behavior mapping and a set of dynamic input–state predictions; the prospective human study tests both behavioral observability and whether those transition patterns transport.

## Important limits before comparison

- {run_label} contains {n_personas} persona profile(s) and one backbone, repeatedly crossed with cases and conditions. {persona_limit}
- The simulated AI response combines recommendation wording, evidence disclosure, and a next-question prompt; separate V8 `q/s/m` causal effects cannot be recovered from this design.
- Agent state probes are evaluation labels, not neural or true latent states.
- Domains share an abstract 0–100 action axis but not identical substantive meanings; human/agent comparison should be stratified by domain/case before any pooled summary.
- Trial stopping is censored by a six-turn cap for many runs, so a definitive stopping-hazard model is not supported here.
- V8's human care scale contains an intermediate same-day-care category; {run_label} uses a binary endpoint encoded continuously on 0–100. Compare normalized action and target distance, not raw category coefficients.

## Recommendation for the manuscript

Do not replace the current main figure. If human data are collected in time, add a compact agent-versus-human transition/convergence panel or a supplementary V8 analysis. Before human data, the safest manuscript addition is one sentence in the prospective-analysis plan specifying that the same V8 transition and target-distance models will test transport. The generated agent-only plots are useful for analysis rehearsal and preregistration, but eight-persona confidence intervals are too fragile to become another central result.
"""
    (OUT / "optimal_control_v8_simulated_report.md").write_text(text)


def main() -> None:
    turns, probes, manifest = load_tables()
    turn, trans = build_turn_and_transition(turns, probes, manifest)
    cases = case_table(turn, probes, manifest)
    turn.to_csv(TABLE / "turn_level.csv", index=False)
    trans.to_csv(TABLE / "transition_level.csv", index=False)
    cases.to_csv(TABLE / "case_level.csv", index=False)
    coefs = fit_ols_models(trans)
    policy, policy_summary = fit_policy(trans)
    make_figure1(turn)
    make_figure2(coefs)
    make_figure3(policy, cases)
    write_report(turn, trans, cases, coefs, policy_summary)
    print(f"Wrote V8-adapted outputs to {OUT}")


if __name__ == "__main__":
    main()
