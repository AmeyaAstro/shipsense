"""Drift monitoring: compare a new batch of orders with the training data, feature by feature.

Why: the model learned 2015-2016. If new orders look different (a new market, a shift toward
First Class), its scores silently get worse. PSI (Population Stability Index) flags that early.

PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%).  Rule of thumb (credit-risk convention):
  < 0.10 stable · 0.10-0.25 watch · > 0.25 act.
Bins: deciles of the TRAINING data for numbers, one bin per category for text (unseen -> "other").

Run:  make drift MONTH=2018-01            (a real month of orders)
      make drift MONTH=2018-01 ARGS=--simulate-shift   (same month with a synthetic First Class surge)
      make drift ARGS=--summary            (every month after training in one table)
Writes reports/drift/drift_<label>.md (or summary.md).

Finding on DataCo: real months drift hugely on market and basket features (markets rotate in monthly
blocks; from Oct 2017 orders become single-item), yet batch AUC stays ~0.78, because lateness is driven
by shipping mode, whose mix barely moves. Feature drift alone is not proof the model is broken.

So the OVERALL status alerts on what changes decisions: prediction PSI, and batch AUC more than
AUC_DROP below the test AUC once labels arrive. Feature PSI is the diagnosis (why), not the alarm.
(A first version alerted on any feature and fired every month while the model was fine: alert fatigue.)
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from shipsense.config import CATEGORICAL, MODELS_DIR, NUMERIC, REPORTS_DIR, SEED, TARGET, TEST_START, VAL_START
from shipsense.data import load_order_features, time_split

EPS = 1e-4                       # avoids log(0) when a bin is empty in one of the samples
WATCH, ACT = 0.10, 0.25
BASELINE_AUC, AUC_DROP = 0.782, 0.05   # test-set ROC-AUC from Week 1; alert if a batch falls 0.05 below
CALENDAR = {"order_month"}       # one month vs a full year always "drifts": reported, never alerted
DRIFT_DIR = REPORTS_DIR / "drift"


def psi_from_shares(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(ref: pd.Series, cur: pd.Series, n_bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))  # unique: ties can merge deciles
    edges[0], edges[-1] = -np.inf, np.inf                                # catch values outside train range
    ref_share = np.histogram(ref, edges)[0] / len(ref)
    cur_share = np.histogram(cur, edges)[0] / len(cur)
    return psi_from_shares(ref_share, cur_share)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = list(ref.value_counts().index)
    cur = cur.where(cur.isin(cats), "other")                             # unseen categories pooled
    ref_share = ref.value_counts(normalize=True).reindex(cats + ["other"], fill_value=0).to_numpy()
    cur_share = cur.value_counts(normalize=True).reindex(cats + ["other"], fill_value=0).to_numpy()
    return psi_from_shares(ref_share, cur_share)


def status(value: float) -> str:
    return "act" if value > ACT else "watch" if value > WATCH else "stable"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL + NUMERIC:
        value = psi_categorical(ref[col], cur[col]) if col in CATEGORICAL else psi_numeric(ref[col], cur[col])
        rows.append({"feature": col, "psi": round(value, 4),
                     "status": "calendar (not alerted)" if col in CALENDAR else status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def simulate_shift(df: pd.DataFrame, share: float = 0.4) -> pd.DataFrame:
    """Synthetic scenario for demos/tests: move `share` of Standard Class orders to First Class."""
    out = df.copy()
    rng = np.random.default_rng(SEED)
    std = out.index[out["shipping_mode"] == "Standard Class"]
    moved = rng.choice(std, size=int(len(std) * share), replace=False)
    out.loc[moved, ["shipping_mode", "days_scheduled"]] = ["First Class", 1]
    return out


def drift_summary(ref: pd.DataFrame, cur: pd.DataFrame, model) -> dict:
    """Feature PSI, prediction PSI, and (if labels exist) AUC / late rate vs the reference."""
    cols = CATEGORICAL + NUMERIC
    p_ref, p_cur = model.predict_proba(ref[cols])[:, 1], model.predict_proba(cur[cols])[:, 1]
    alerts = feature_drift(ref, cur)
    summary = {
        "n_ref": len(ref), "n_cur": len(cur), "features": alerts,
        "prediction_psi": round(psi_numeric(pd.Series(p_ref), pd.Series(p_cur)), 4),
        "mean_score_ref": round(float(p_ref.mean()), 4), "mean_score_cur": round(float(p_cur.mean()), 4),
    }
    if TARGET in cur and cur[TARGET].nunique() == 2:   # labels arrive after delivery, so may be missing
        summary["late_rate_ref"] = round(float(ref[TARGET].mean()), 4)
        summary["late_rate_cur"] = round(float(cur[TARGET].mean()), 4)
        summary["auc_cur"] = round(float(roc_auc_score(cur[TARGET], p_cur)), 4)
    overall = status(summary["prediction_psi"])
    if summary.get("auc_cur", BASELINE_AUC) < BASELINE_AUC - AUC_DROP:
        overall = "act"
    summary["overall"] = overall
    return summary


def write_markdown(summary: dict, label: str, note: str = "") -> str:
    f = summary["features"]
    lines = [f"# Drift report: {label}", "", note, "",
             f"Reference: training orders (2015-2016, n={summary['n_ref']:,}) · current batch n={summary['n_cur']:,}",
             "", f"**Overall: {summary['overall'].upper()}**  (from prediction PSI: < {WATCH} stable · "
             f"{WATCH}-{ACT} watch · > {ACT} act; forced to act if batch AUC < {BASELINE_AUC - AUC_DROP:.3f}). "
             "Feature PSI below explains why.", "", "## Predictions", "",
             "| Prediction PSI | Mean score (train) | Mean score (batch) |", "|---|---|---|",
             f"| {summary['prediction_psi']} | {summary['mean_score_ref']} | {summary['mean_score_cur']} |", ""]
    if "auc_cur" in summary:
        lines += ["## Outcomes (labels available)", "", "| Late rate (train) | Late rate (batch) | ROC-AUC (batch) |",
                  "|---|---|---|", f"| {summary['late_rate_ref']} | {summary['late_rate_cur']} | "
                  f"{summary['auc_cur']} |", "", f"Model test-set ROC-AUC for comparison: {BASELINE_AUC}.", ""]
    lines += ["## Features (highest PSI first)", "", "| Feature | PSI | Status |", "|---|---|---|",
              *[f"| {r.feature} | {r.psi} | {r.status} |" for r in f.itertuples()], ""]
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    path = DRIFT_DIR / f"drift_{label}.md"
    path.write_text("\n".join(lines))
    return str(path.relative_to(REPORTS_DIR.parent))


def monthly_summary(df: pd.DataFrame, train: pd.DataFrame, model) -> str:
    """One row per month after training: overall status, top drifting feature, prediction PSI, AUC."""
    lines = ["# Drift summary: every month after training vs training (2015-2016)", "",
             "Real orders. Overall = prediction PSI (< 0.10 stable · 0.10-0.25 watch · > 0.25 act), forced to "
             f"act if AUC < {BASELINE_AUC - AUC_DROP:.3f}. AUC uses each month's actual labels (in production "
             f"they arrive days later). Model test-set AUC: {BASELINE_AUC}.", "",
             "| Month | Orders | Overall | Top drifting feature (PSI) | shipping_mode PSI | Prediction PSI "
             "| Late rate | ROC-AUC |", "|---|---|---|---|---|---|---|---|"]
    months = sorted(df.loc[df["order_ts"] >= VAL_START, "order_ts"].dt.strftime("%Y-%m").unique())
    for month in months:
        batch = df[df["order_ts"].dt.strftime("%Y-%m") == month]
        s = drift_summary(train, batch, model)
        f = s["features"]
        top = f[~f["feature"].isin(CALENDAR)].iloc[0]
        mode_psi = f.loc[f["feature"] == "shipping_mode", "psi"].iloc[0]
        lines.append(f"| {month} | {s['n_cur']:,} | {s['overall']} | {top.feature} ({top.psi}) | {mode_psi} "
                     f"| {s['prediction_psi']} | {s['late_rate_cur']} | {s['auc_cur']} |")
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    (DRIFT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    return "reports/drift/summary.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="PSI drift report for one month of orders vs training")
    parser.add_argument("--month", default="2018-01", help="YYYY-MM to treat as the new batch")
    parser.add_argument("--simulate-shift", action="store_true", help="add a synthetic First Class surge")
    parser.add_argument("--summary", action="store_true", help="table of every month after training")
    args = parser.parse_args()

    df = load_order_features()
    train, _, _ = time_split(df, VAL_START, TEST_START)
    if args.summary:
        print("wrote", monthly_summary(df, train, joblib.load(MODELS_DIR / "gradient_boosting.joblib")))
        return
    batch = df[df["order_ts"].dt.strftime("%Y-%m") == args.month]
    label, note = args.month, "Real orders from this month."
    if args.simulate_shift:
        batch, label = simulate_shift(batch), f"{args.month}_simulated_shift"
        note = ("**SYNTHETIC SCENARIO:** 40% of this month's Standard Class orders were relabelled as "
                "First Class to show what the monitor does when the mix shifts. Not real data.")
    summary = drift_summary(train, batch, joblib.load(MODELS_DIR / "gradient_boosting.joblib"))
    path = write_markdown(summary, label, note)
    print(f"{label}: overall {summary['overall']}, prediction PSI {summary['prediction_psi']} -> {path}")


if __name__ == "__main__":
    main()
