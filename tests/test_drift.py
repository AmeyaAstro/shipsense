"""PSI maths and the drift alert rule, on synthetic data (no CSV needed)."""
import math

import numpy as np
import pandas as pd
import pytest
from test_pipeline import make_items

from shipsense.baselines import gradient_boosting
from shipsense.config import CATEGORICAL, NUMERIC, TARGET
from shipsense.data import build_database, load_order_features
from shipsense.drift import drift_summary, psi_categorical, psi_from_shares, psi_numeric, simulate_shift, status


def test_psi_matches_hand_calculation():
    # ref 50/50 -> cur 25/75:  (0.25-0.5)*ln(0.25/0.5) + (0.75-0.5)*ln(0.75/0.5)
    expected = -0.25 * math.log(0.5) + 0.25 * math.log(1.5)
    assert psi_from_shares(np.array([0.5, 0.5]), np.array([0.25, 0.75])) == pytest.approx(expected)


def test_identical_samples_have_zero_psi():
    x = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi_numeric(x, x) == pytest.approx(0, abs=1e-9)
    c = pd.Series(["a", "b", "b", "c"] * 100)
    assert psi_categorical(c, c) == pytest.approx(0, abs=1e-9)


def test_shifted_numbers_and_unseen_categories_raise_psi():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.1   # same distribution, new sample
    assert psi_numeric(ref, pd.Series(rng.normal(1, 1, 5000))) > 0.25  # mean moved by 1 sd
    cats = pd.Series(["a", "b"] * 500)
    assert psi_categorical(cats, pd.Series(["a", "b", "z", "z"] * 250)) > 0.25  # 'z' never seen -> "other"


def test_status_thresholds():
    assert [status(v) for v in (0.05, 0.1, 0.2, 0.3)] == ["stable", "stable", "watch", "act"]


@pytest.fixture(scope="module")
def orders_and_model(tmp_path_factory):
    db = tmp_path_factory.mktemp("drift") / "t.db"
    build_database(make_items(n_orders=600), db_path=db)
    df = load_order_features(db_path=db)
    return df, gradient_boosting().fit(df[CATEGORICAL + NUMERIC], df[TARGET])


def test_no_change_is_stable_and_simulated_shift_alerts(orders_and_model):
    df, model = orders_and_model
    same = drift_summary(df, df, model)
    assert same["overall"] == "stable" and same["prediction_psi"] == pytest.approx(0, abs=1e-6)
    shifted = drift_summary(df, simulate_shift(df, share=1.0), model)  # every Standard order -> First Class
    psi = shifted["features"].set_index("feature")["psi"]
    assert psi["shipping_mode"] > 0.25 and shifted["overall"] in ("watch", "act")
