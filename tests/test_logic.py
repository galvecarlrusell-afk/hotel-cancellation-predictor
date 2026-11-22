"""Tests for the helper functions, the saved models and the saved result files.

Run:  python -m pytest
(The model and result tests are skipped until you have run python train.py.)
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import (explain_booking, invalid_rows, predicted_label, revenue_at_risk,  # noqa: E402
                   risk_level, validate_booking)


# ---------- helper functions ----------
def test_risk_levels():
    assert risk_level(0.10)[0] == "LOW"
    assert risk_level(0.35)[0] == "MEDIUM"
    assert risk_level(0.59)[0] == "MEDIUM"
    assert risk_level(0.60)[0] == "HIGH"
    assert risk_level(0.50, medium=0.2, high=0.4)[0] == "HIGH"


def test_predicted_label():
    assert predicted_label(0.7) == "Likely to Cancel"
    assert predicted_label(0.2) == "Likely to Stay"


def test_validate_booking():
    assert validate_booking(2, 1, 2, 100) == []
    assert len(validate_booking(0, 1, 2, 100)) == 1
    assert len(validate_booking(2, 0, 0, 100)) == 1
    assert len(validate_booking(2, 1, 2, 0)) == 1
    assert len(validate_booking(0, 0, 0, 0)) == 3


def test_revenue_at_risk():
    assert revenue_at_risk(100, 3, 0.5) == 150


def test_invalid_rows():
    good = {"no_of_adults": 2, "no_of_weekend_nights": 1, "no_of_week_nights": 2,
            "avg_price_per_room": 100.0, "arrival_month": 6}
    df = pd.DataFrame([good,
                       {**good, "no_of_adults": 0},
                       {**good, "no_of_weekend_nights": 0, "no_of_week_nights": 0},
                       {**good, "avg_price_per_room": 0},
                       {**good, "arrival_month": 13},
                       {**good, "arrival_month": None}])
    assert invalid_rows(df).tolist() == [False, True, True, True, True, True]


def test_explain_booking_points_to_the_input_that_changes_the_risk():
    row = pd.DataFrame([{"lead_time": 200, "price": 100}])
    reference = {"lead_time": 50, "price": 100}

    def fake_model(df):            # risk depends only on lead time
        return (df["lead_time"] / 400).to_numpy()

    out = explain_booking(fake_model, row, reference)
    assert out["feature"].tolist() == ["lead_time"]
    assert out["effect"].iloc[0] == pytest.approx(0.375)


# ---------- saved models and results (need python train.py first) ----------
INFO = ROOT / "models" / "info.json"
needs_training = pytest.mark.skipif(not INFO.exists(), reason="run python train.py first")


@needs_training
def test_saved_models_predict_a_valid_probability():
    import joblib
    info = json.loads(INFO.read_text())
    row = pd.DataFrame([info["reference"]])[info["columns"]]
    assert len(info["models"]) >= 3
    for m in info["models"]:
        p = joblib.load(ROOT / m["file"]).predict_proba(row)[0, 1]
        assert 0.0 <= p <= 1.0


@needs_training
def test_result_files_exist():
    info = json.loads(INFO.read_text())
    for name in ["model_comparison.csv", "cv_results.csv", "feature_importance.csv",
                 "insights_lead_time.csv", "insights_segment.csv", "insights_requests.csv",
                 "insights_month.csv", "roc_curves.png", "pr_curves.png"]:
        assert (ROOT / "results" / name).exists(), name
    for m in info["models"]:
        assert (ROOT / "results" / f"confusion_{m['slug']}.png").exists()


@needs_training
def test_baseline_is_beaten_on_f1():
    comp = pd.read_csv(ROOT / "results" / "model_comparison.csv").set_index("Model")
    best = json.loads(INFO.read_text())["best_model"]
    assert comp.loc[best, "F1"] > comp.loc["Baseline (always Not Canceled)", "F1"]