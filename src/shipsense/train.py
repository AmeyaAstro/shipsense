"""Train every model on the same time split and write an honest comparison.

Run:  python -m shipsense.train
Outputs: reports/metrics.json, reports/model_comparison.md, models/*.joblib|.pt,
         and one MLflow parent run (+ a child run per model) in mlruns/.
"""
from __future__ import annotations

import json
import time

import joblib
import mlflow
import torch

from shipsense import tracking
from shipsense.baselines import ShippingModeRule, gradient_boosting, logistic_regression
from shipsense.config import (
    CATEGORICAL,
    MODELS_DIR,
    NUMERIC,
    REPORTS_DIR,
    TARGET,
    TEST_START,
    VAL_START,
)
from shipsense.data import DB_PATH, build_database, load_order_features, load_raw, time_split
from shipsense.evaluate import best_threshold, score
from shipsense.torch_model import TorchClassifier


def main() -> dict:
    if not DB_PATH.exists():
        build_database(load_raw())
    df = load_order_features()
    train, val, test = time_split(df, VAL_START, TEST_START)
    cols = CATEGORICAL + NUMERIC
    Xtr, ytr = train[cols], train[TARGET]
    Xva, yva = val[cols], val[TARGET]
    Xte, yte = test[cols], test[TARGET]
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    models = {
        "shipping_mode_rule": ShippingModeRule(),
        "logistic_regression": logistic_regression(),
        "gradient_boosting": gradient_boosting(),
        "pytorch_mlp": TorchClassifier(),
    }
    results = {}
    MODELS_DIR.mkdir(exist_ok=True)
    tracking.setup()
    parent = mlflow.start_run(run_name="train_all")
    mlflow.log_params({"n_train": len(train), "n_val": len(val), "n_test": len(test),
                       "val_start": VAL_START, "test_start": TEST_START,
                       "n_features": len(cols), "features": ",".join(cols)})
    for name, model in models.items():
        t0 = time.time()
        if name == "pytorch_mlp":
            model.fit(Xtr, ytr, Xva, yva)
        else:
            model.fit(Xtr, ytr)
        secs = round(time.time() - t0, 1)
        thr = best_threshold(yva.to_numpy(), model.predict_proba(Xva)[:, 1])
        results[name] = {
            "val": score(yva.to_numpy(), model.predict_proba(Xva)[:, 1], thr),
            "test": score(yte.to_numpy(), model.predict_proba(Xte)[:, 1], thr),
            "train_seconds": secs,
        }
        print(name, results[name]["test"])
        if name == "pytorch_mlp":
            model_file = MODELS_DIR / "pytorch_mlp.pt"
            torch.save({"state_dict": model.model.state_dict(), "encoder": model.encoder,
                        "history": model.history}, model_file)
        else:
            model_file = MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_file)
        tracking.log_model_run(name, model, results[name], model_file)

    REPORTS_DIR.mkdir(exist_ok=True)
    summary = {"split": {"train": len(train), "val": len(val), "test": len(test),
                         "val_start": VAL_START, "test_start": TEST_START,
                         "test_late_rate": round(float(yte.mean()), 4)},
               "models": results}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary)
    mlflow.log_artifacts(str(REPORTS_DIR), artifact_path="reports")
    mlflow.end_run()
    print(f"MLflow run {parent.info.run_id} logged to {tracking.TRACKING_URI}")
    return summary


def write_markdown(summary: dict) -> None:
    lines = ["| Model | ROC-AUC | PR-AUC | Balanced acc. | Accuracy | F1 | Brier | Train time (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in summary["models"].items():
        t = r["test"]
        lines.append(f"| {name} | {t['roc_auc']} | {t['pr_auc']} | {t['balanced_accuracy']} | {t['accuracy']} "
                     f"| {t['f1']} | {t['brier']} | {r['train_seconds']} |")
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
