#!/usr/bin/env python3
"""Validate public figure tables against manuscript values and check PDFs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
ATOL = 1e-9


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=ATOL, rtol=0):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def validate_figure3() -> None:
    data = pd.read_csv(DATA / "figure3.csv")
    a = data[data.panel == "a"].set_index("state")
    expected_effects = {
        "intended_action": 1.2520788194998402,
        "uncertainty": 1.4788721179065332,
        "information_need": -0.6083446897424913,
    }
    for state, expected in expected_effects.items():
        assert_close(float(a.loc[state, "value"]), expected, f"Figure 3a {state}")

    b = data[data.panel == "b"].set_index(["state", "model"])
    expected = {
        ("urgency", "context_only"): 0.6788133170470771,
        ("urgency", "context_plus_behavior"): 0.6781635208130985,
        ("information_need", "context_only"): 0.5386093350331431,
        ("information_need", "context_plus_behavior"): 0.6072916731363089,
        ("intended_action", "context_only"): 0.7165172151022724,
        ("intended_action", "context_plus_behavior"): 0.7197282635008488,
        ("uncertainty", "context_only"): 0.7256753446345166,
        ("uncertainty", "context_plus_behavior"): 0.7440727566231993,
    }
    for key, value in expected.items():
        assert_close(float(b.loc[key, "value"]), value, f"Figure 3b {key}")


def validate_figure4() -> None:
    data = pd.read_csv(DATA / "figure4.csv").set_index(["domain", "state"])
    expected_information_need = {
        "medical": 0.0892172090515386,
        "legal": 0.0713785954461423,
        "finance": 0.0587683077785168,
    }
    for domain, value in expected_information_need.items():
        assert_close(float(data.loc[(domain, "information_need"), "delta_r2"]), value, f"Figure 4 {domain}")
    if len(data) != 12:
        raise AssertionError(f"Figure 4 must contain 4 states x 3 domains; found {len(data)} rows")


def validate_figure5() -> None:
    data = pd.read_csv(DATA / "figure5.csv")
    expected_states = {"urgency", "information_need", "intended_action", "uncertainty"}
    if set(data.state) != expected_states:
        raise AssertionError(f"Figure 5 states differ: {sorted(set(data.state))}")
    if set(data.source) != {"Human", "Simulation"}:
        raise AssertionError("Figure 5 must include Human and Simulation")
    if set(data.case_reference) != {"Self-care", "Calling an ambulance"}:
        raise AssertionError("Figure 5 case references differ from the manuscript")
    if data.turn_index.min() != 0 or data.turn_index.max() != 6:
        raise AssertionError("Figure 5 release must contain the plotted turn range 0--6")
    if (data["n"] < 1).any() or not data["mean"].between(0, 100).all():
        raise AssertionError("Figure 5 has invalid n or state values")


def validate_figure6() -> None:
    data = pd.read_csv(DATA / "figure6.csv").set_index(["source", "direction", "confidence", "outcome"])
    expected = {
        ("Human", "higher", "high", "final_action"): 81.25,
        ("Human", "higher", "low", "final_action"): 40.0,
        ("Human", "lower", "high", "final_action"): 11.11111111111111,
        ("Human", "lower", "low", "final_action"): 7.6923076923076925,
        ("Simulation", "higher", "high", "final_action"): 90.0,
        ("Simulation", "higher", "low", "final_action"): 45.0,
        ("Simulation", "lower", "high", "final_action"): 10.0,
        ("Simulation", "lower", "low", "final_action"): 20.0,
        ("Human", "higher", "high", "information_need_change"): -16.625,
        ("Human", "higher", "low", "information_need_change"): -19.3,
        ("Human", "lower", "high", "information_need_change"): -53.111111111111114,
        ("Human", "lower", "low", "information_need_change"): -38.23076923076923,
        ("Simulation", "higher", "high", "information_need_change"): -57.5,
        ("Simulation", "higher", "low", "information_need_change"): -42.5,
        ("Simulation", "lower", "high", "information_need_change"): -55.0,
        ("Simulation", "lower", "low", "information_need_change"): -30.0,
    }
    for key, value in expected.items():
        assert_close(float(data.loc[key, "mean"]), value, f"Figure 6 {key}")


def validate_supplement() -> None:
    s1 = pd.read_csv(DATA / "figure_s1.csv")
    if set(s1.state) != {"R", "I", "A", "U"} or set(s1.direction) != {"lower", "higher"}:
        raise AssertionError("Figure S1 state/direction coverage is incomplete")
    assert_close(float(s1.query("state == 'R' and direction == 'lower' and progress == 0")['mean'].iloc[0]),
                 0.5570833333333335, "Figure S1 baseline")
    s2 = pd.read_csv(DATA / "figure_s2.csv").set_index(["outcome", "term"])
    assert_close(float(s2.loc[("dA", "direction_high"), "standardized_beta"]),
                 0.5102955075640254, "Figure S2 advice-action transition")
    s3 = pd.read_csv(DATA / "figure_s3.csv")
    align = s3[s3.table == "alignment"].set_index(["recommendation_direction", "recommendation_confidence"])
    assert_close(float(align.loc[("higher", "high"), "ai_alignment"]), 0.9166666666666666,
                 "Figure S3 higher/high AI alignment")
    s4 = pd.read_csv(DATA / "figure_s4.csv").set_index(["source", "metric"])
    assert_close(float(s4.loc[("Human", "words_per_message"), "mean"]),
                 12.644255952380952, "Figure S4 human words/message")
    assert_close(float(s4.loc[("Simulation", "evidence_acquired"), "mean"]), 7.625,
                 "Figure S4 simulated evidence")


def validate_pdfs() -> None:
    names = [*(f"figure{number}" for number in range(3, 7)), *(f"figure_s{number}" for number in range(1, 5))]
    for name in names:
        path = FIGURES / f"{name}.pdf"
        if not path.exists() or path.stat().st_size < 2_000:
            raise AssertionError(f"Missing or unexpectedly small PDF: {path}")
        if path.read_bytes()[:4] != b"%PDF":
            raise AssertionError(f"Not a PDF: {path}")


def main() -> None:
    validate_figure3()
    validate_figure4()
    validate_figure5()
    validate_figure6()
    validate_supplement()
    validate_pdfs()
    print("Validation passed: main and supplementary release tables match manuscript values and all PDFs are present.")


if __name__ == "__main__":
    main()
