"""MLflow experiment tracking, kept in one place so train.py stays readable.

Why: metrics.json is overwritten on every run, so there's no history. MLflow keeps
every run (params, metrics, model files) and lets us compare runs in a web UI.

Storage is local and gitignored: run metadata in mlruns/mlflow.db (SQLite; MLflow 3
deprecated the plain-folder store) and files under mlruns/artifacts/.
View with:  make mlflow-ui   (then open http://127.0.0.1:5000)
"""
from __future__ import annotations

from pathlib import Path

import mlflow

from shipsense.config import ROOT

MLRUNS_DIR = ROOT / "mlruns"
TRACKING_URI = f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}"
EXPERIMENT = "shipsense-late-delivery"


def setup() -> None:
    """Point MLflow at the local store and select (or create) our experiment."""
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(MLRUNS_DIR / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def model_params(model) -> dict:
    """Hyperparameters worth recording for each kind of model we train."""
    if hasattr(model, "named_steps"):  # sklearn Pipeline: log the final estimator's settings
        clf = model.named_steps["clf"]
        params = {k: v for k, v in clf.get_params().items() if isinstance(v, (int, float, str, bool))}
        return {"estimator": type(clf).__name__, **params}
    if hasattr(model, "lr"):  # TorchClassifier
        return {"estimator": "TorchClassifier", "epochs": model.epochs, "batch_size": model.batch_size,
                "lr": model.lr, "weight_decay": model.weight_decay, "patience": model.patience}
    return {"estimator": type(model).__name__}


def log_model_run(name: str, model, result: dict, model_file: Path) -> None:
    """Log one model as a child run: params, val/test metrics, and its saved file."""
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params(model_params(model))
        mlflow.log_metric("train_seconds", result["train_seconds"])
        for split in ("val", "test"):
            mlflow.log_metrics({f"{split}_{k}": v for k, v in result[split].items()})
        for row in getattr(model, "history", []):  # per-epoch loss curve (PyTorch only)
            mlflow.log_metrics({"train_loss": row["train_loss"], "val_loss": row["val_loss"]}, step=row["epoch"])
        mlflow.log_artifact(str(model_file), artifact_path="model")
