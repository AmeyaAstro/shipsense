"""The copilot's three tools, plus their JSON-schema descriptions for the LLM.

Why tools: an LLM can't query a database or run our model by itself, and it shouldn't guess.
It ASKS for a tool by name with arguments; our code runs it and returns JSON. Errors are returned
as data ({"error": ...}) instead of raised, so the model can read them and recover.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from shipsense.api import Order, to_features
from shipsense.config import CATEGORICAL, NUMERIC
from shipsense.data import add_calendar_features
from shipsense.rag import PolicyIndex

TOOL_SPECS = [
    {
        "name": "search_policy",
        "description": "Search ShipSense's (fictional) shipping policies and SOPs. Use for any question about "
                       "rules, credits, refunds, escalation, SLAs, customs, upgrades or procedures.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "what to look up, in plain words"}}},
    },
    {
        "name": "lookup_shipment",
        "description": "Look up what actually happened to an existing order: shipping mode, scheduled vs real "
                       "shipping days, delivery status, order status, destination.",
        "parameters": {"type": "object", "required": ["order_id"], "properties": {
            "order_id": {"type": "integer"}}},
    },
    {
        "name": "predict_risk",
        "description": "Predict the late-delivery risk of an order with the ML model. Pass order_id for an "
                       "existing order, or `order` with the order's fields for a new one.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "integer"},
            "order": {"type": "object", "description": "fields of a new order (same schema as POST /predict)"}}},
    },
]


class ShipSenseTools:
    def __init__(self, db_path: Path, model, threshold: float, model_version: str, index: PolicyIndex):
        self.db_path, self.model, self.threshold = db_path, model, threshold
        self.model_version, self.index = model_version, index

    def execute(self, name: str, arguments: dict) -> dict:
        """Run a tool by name. Unknown tools and bad arguments come back as {"error": ...}."""
        fn = {"search_policy": self.search_policy, "lookup_shipment": self.lookup_shipment,
              "predict_risk": self.predict_risk}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**arguments)
        except (TypeError, ValueError) as e:  # wrong/missing arguments from the model
            return {"error": f"bad arguments for {name}: {e}"}

    def search_policy(self, query: str, k: int = 4) -> dict:
        hits = self.index.search(query, k=k)
        return {"results": [{"citation": h.chunk.citation, "score": round(h.score, 3), "text": h.chunk.text}
                            for h in hits]}

    def lookup_shipment(self, order_id: int) -> dict:
        with sqlite3.connect(self.db_path) as con:
            df = pd.read_sql("SELECT * FROM shipments WHERE order_id = ?", con, params=[int(order_id)])
        if df.empty:
            return {"error": f"order {order_id} not found"}
        return json.loads(df.iloc[0].to_json())  # native ints/floats, not numpy types

    def predict_risk(self, order_id: int | None = None, order: dict | None = None) -> dict:
        if order is not None:
            try:
                X = to_features([Order(**order)])
            except ValidationError as e:
                return {"error": f"invalid order fields: {e.errors(include_url=False)}"}
        elif order_id is not None:
            with sqlite3.connect(self.db_path) as con:
                df = pd.read_sql("SELECT * FROM order_features WHERE order_id = ?", con, params=[int(order_id)])
            if df.empty:  # order_features only holds shipped orders; say WHY there is no score
                status = self.lookup_shipment(order_id).get("order_status")
                if status:
                    return {"order_id": order_id, "order_status": status,
                            "note": f"order exists but has status {status}; it never shipped, so the "
                                    "late-delivery model does not score it"}
                return {"error": f"order {order_id} not found"}
            X = add_calendar_features(df)[CATEGORICAL + NUMERIC]
        else:
            return {"error": "pass order_id or order"}
        p = float(self.model.predict_proba(X)[0, 1])
        return {"order_id": order_id, "late_probability": round(p, 4), "is_late_risk": p >= self.threshold,
                "threshold": self.threshold, "model_version": self.model_version}


def to_json(result: dict) -> str:
    """Tool results go back to the LLM as JSON text (default=str handles numpy/pandas scalars)."""
    return json.dumps(result, default=str)
