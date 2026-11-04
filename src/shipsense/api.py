"""FastAPI service that scores late-delivery risk for orders.

Why: a model is only useful once other systems can call it. The API loads the
production model (HistGradientBoosting pipeline) and the threshold tuned on the
validation set, validates every request with Pydantic, and builds features with
the same code as training.

POST /ask (Week 3) answers ops questions with the RAG + tools copilot. It is built lazily on the
first question, so /predict works without the copilot's extra dependencies (e.g. in Docker).

Run:  make serve   (then open http://127.0.0.1:8000/docs for interactive docs)
Paths can be overridden with env vars SHIPSENSE_MODEL_PATH / SHIPSENSE_METRICS_PATH.
On AWS (Week 4) set SHIPSENSE_MODEL_S3_URI=s3://bucket/key: the model is downloaded to /tmp at startup,
so a new model is an S3 upload, not a new container image.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from shipsense.config import CATEGORICAL, DAYS_SCHEDULED_BY_MODE, DB_PATH, MODELS_DIR, NUMERIC, REPORTS_DIR
from shipsense.data import add_calendar_features

MODEL_NAME = "gradient_boosting"
DEFAULT_MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.joblib"
DEFAULT_METRICS_PATH = REPORTS_DIR / "metrics.json"


class Order(BaseModel):
    """One order's fields, all known at the moment the order is placed."""

    order_ts: datetime = Field(examples=["2017-09-14T10:30:00"])
    shipping_mode: Literal["Standard Class", "Second Class", "First Class", "Same Day"]
    payment_type: Literal["DEBIT", "TRANSFER", "PAYMENT", "CASH"]
    market: Literal["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"]
    order_region: str = Field(min_length=1, examples=["Central America"])
    order_country: str = Field(min_length=1, examples=["México"])
    customer_segment: Literal["Consumer", "Corporate", "Home Office"]
    main_department: str = Field(min_length=1, examples=["Fan Shop"])
    n_items: int = Field(ge=1, le=100)
    total_quantity: int = Field(ge=1, le=1000)
    total_sales: float = Field(gt=0)
    avg_discount_rate: float = Field(ge=0, le=1)
    n_departments: int = Field(ge=1)

    @model_validator(mode="after")
    def check_consistent(self) -> Order:
        if self.total_quantity < self.n_items:
            raise ValueError("total_quantity must be >= n_items (each item has quantity >= 1)")
        if self.n_departments > self.n_items:
            raise ValueError("n_departments cannot exceed n_items")
        return self


class BatchRequest(BaseModel):
    orders: list[Order] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    late_probability: float
    is_late_risk: bool
    threshold: float
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000,
                          examples=["How much credit is owed if an order is 2 days late?"])


class ToolCallOut(BaseModel):
    name: str
    arguments: dict


class AskResponse(BaseModel):
    answer: str
    sources: list[str]          # policy passages retrieved (doc § section)
    invalid_citations: list[str]  # cited in the answer but NOT retrieved, i.e. made up by the LLM
    tool_calls: list[ToolCallOut]
    steps: int
    llm_model: str


def to_features(orders: list[Order]) -> pd.DataFrame:
    """Turn validated orders into the exact feature columns the model was trained on."""
    df = pd.DataFrame([o.model_dump() for o in orders])
    df["days_scheduled"] = df["shipping_mode"].map(DAYS_SCHEDULED_BY_MODE)
    df = add_calendar_features(df)
    return df[CATEGORICAL + NUMERIC]


def file_version(path: Path) -> str:
    """Model version = name + short hash of the file, so a prediction traces to an exact binary."""
    return f"{MODEL_NAME}-{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


def download_from_s3(uri: str, dest_dir: Path = Path("/tmp")) -> Path:
    """Fetch s3://bucket/key to dest_dir. boto3 is imported lazily: only the AWS image needs it."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not uri.startswith("s3://") or not bucket or not key:  # fail fast on bad config, before importing boto3
        raise ValueError(f"expected s3://bucket/key, got {uri!r}")
    import boto3

    dest = dest_dir / Path(key).name
    boto3.client("s3").download_file(bucket, key, str(dest))
    return dest


def build_default_agent(state: dict):
    """Real copilot: MiniLM + FAISS index, SQLite tools, LLM from env (Ollama by default)."""
    from shipsense.agent import Agent  # lazy: avoids a circular import and heavy deps at startup
    from shipsense.llm import get_llm
    from shipsense.rag import INDEX_DIR, PolicyIndex, SentenceTransformerEmbedder
    from shipsense.tools import ShipSenseTools

    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError("policy index missing: run `make copilot-data`")
    index = PolicyIndex.load(SentenceTransformerEmbedder(), INDEX_DIR)
    tools = ShipSenseTools(DB_PATH, state["model"], state["threshold"], state["version"], index)
    return Agent(get_llm(), tools)


def create_app(model_path: Path | None = None, metrics_path: Path | None = None,
               agent_factory: Callable[[dict], object] | None = None) -> FastAPI:
    """Build the app. Tests pass small model/metrics files and a fake agent; production uses the defaults."""
    model_path = Path(model_path or os.getenv("SHIPSENSE_MODEL_PATH", DEFAULT_MODEL_PATH))
    metrics_path = Path(metrics_path or os.getenv("SHIPSENSE_METRICS_PATH", DEFAULT_METRICS_PATH))
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once at startup (not per request). joblib uses pickle, so only load our own files.
        s3_uri = os.getenv("SHIPSENSE_MODEL_S3_URI")
        model_file = download_from_s3(s3_uri) if s3_uri else model_path
        state["model"] = joblib.load(model_file)
        metrics = json.loads(metrics_path.read_text())
        state["threshold"] = metrics["models"][MODEL_NAME]["val"]["threshold"]
        state["version"] = file_version(model_file)
        yield
        state.clear()

    app = FastAPI(title="ShipSense", version="0.3.0", lifespan=lifespan,
                  description="Late-delivery risk scoring and an ops copilot (fictional policies).")

    def predict_many(orders: list[Order]) -> list[Prediction]:
        proba = state["model"].predict_proba(to_features(orders))[:, 1]
        thr = state["threshold"]
        return [Prediction(late_probability=round(float(p), 4), is_late_risk=bool(p >= thr),
                           threshold=thr, model_version=state["version"]) for p in proba]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": state["version"]}

    @app.post("/predict", response_model=Prediction)
    def predict(order: Order) -> Prediction:
        return predict_many([order])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest) -> BatchResponse:
        return BatchResponse(predictions=predict_many(req.orders))

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        from shipsense.llm import LLMUnavailable

        try:
            if "agent" not in state:
                state["agent"] = (agent_factory or build_default_agent)(state)
            result = state["agent"].run(req.question)
        except (FileNotFoundError, LLMUnavailable) as e:
            raise HTTPException(status_code=503, detail=f"copilot unavailable: {e}") from e
        return AskResponse(answer=result.answer, sources=result.sources, steps=result.steps,
                           invalid_citations=result.invalid_citations,
                           tool_calls=[ToolCallOut(name=t.name, arguments=t.arguments) for t in result.tool_calls],
                           llm_model=state["agent"].llm.model)

    return app


app = create_app()
