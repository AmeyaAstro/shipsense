"""Tools, agent loop and POST /ask, using a scripted fake LLM (no Ollama, no model download)."""
import json

import joblib
import pytest
from fastapi.testclient import TestClient
from test_api import VALID
from test_pipeline import make_items

from shipsense.agent import Agent, prefetch_calls
from shipsense.api import MODEL_NAME, create_app
from shipsense.baselines import gradient_boosting
from shipsense.config import CATEGORICAL, NUMERIC, TARGET
from shipsense.data import build_database, load_order_features
from shipsense.llm import LLMResponse, LLMUnavailable, ToolCall
from shipsense.rag import HashingEmbedder, PolicyIndex, load_knowledge
from shipsense.shipments import build_shipments_table
from shipsense.tools import ShipSenseTools


class ScriptedLLM:
    """Plays back canned replies in order and records what it was sent."""

    model = "scripted"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic DB (order_features + shipments), tiny GBM, and a hashing-embedder policy index."""
    tmp = tmp_path_factory.mktemp("agent")
    db = tmp / "test.db"
    items = make_items()
    build_database(items, db_path=db)
    ship = items.assign(shipping_ts=items["order_ts"], order_city="Springfield",
                        days_real=items["days_scheduled"] + items["late_delivery_risk"],
                        delivery_status=items["late_delivery_risk"].map({1: "Late delivery", 0: "Shipping on time"}))
    build_shipments_table(ship, db_path=db)
    df = load_order_features(db_path=db)
    model = gradient_boosting().fit(df[CATEGORICAL + NUMERIC], df[TARGET])
    index = PolicyIndex.build(load_knowledge(), HashingEmbedder())
    tools = ShipSenseTools(db, model, 0.5, "test-version", index)
    shipped_id = int(df["order_id"].iloc[0])
    canceled_id = int(items.loc[items["order_status"] == "CANCELED", "order_id"].iloc[0])
    return {"tools": tools, "model": model, "df": df, "shipped_id": shipped_id,
            "canceled_id": canceled_id, "tmp": tmp}


def test_search_policy_returns_citations(env):
    out = env["tools"].search_policy("no carrier scan for 10 business days lost")
    assert out["results"][0]["citation"].startswith("09_damaged_and_lost_shipments.md § ")


def test_lookup_shipment(env):
    row = env["tools"].lookup_shipment(env["canceled_id"])
    assert row["order_status"] == "CANCELED"   # lookup covers all orders, not just shipped ones
    assert isinstance(row["days_real"], int)    # native types, not numpy
    assert "error" in env["tools"].lookup_shipment(999_999)


def test_predict_risk_by_id_matches_model(env):
    out = env["tools"].predict_risk(order_id=env["shipped_id"])
    row = env["df"][env["df"]["order_id"] == env["shipped_id"]]
    expected = env["model"].predict_proba(row[CATEGORICAL + NUMERIC])[0, 1]
    assert out["late_probability"] == pytest.approx(expected, abs=1e-4)
    assert out["is_late_risk"] == (expected >= 0.5)


def test_predict_risk_errors_are_returned_not_raised(env):
    t = env["tools"]
    assert "not found among shipped" in t.predict_risk(order_id=env["canceled_id"])["error"]
    assert "invalid order fields" in t.predict_risk(order={**VALID, "shipping_mode": "Rocket"})["error"]
    assert 0 <= t.predict_risk(order=VALID)["late_probability"] <= 1
    assert "unknown tool" in t.execute("drop_tables", {})["error"]
    assert "bad arguments" in t.execute("lookup_shipment", {"wrong": 1})["error"]


def test_agent_runs_tool_then_answers_with_sources(env):
    llm = ScriptedLLM([
        LLMResponse("", [ToolCall("c1", "search_policy", {"query": "when is a shipment lost"})]),
        LLMResponse("After 10 business days with no scan [09_damaged_and_lost_shipments.md § ...]."),
    ])
    result = Agent(llm, env["tools"], prefetch=False).run("When is a shipment lost?")
    assert result.answer.startswith("After 10 business days")
    assert result.steps == 2 and [t.name for t in result.tool_calls] == ["search_policy"]
    assert result.sources and result.sources[0].startswith("09_damaged_and_lost_shipments.md")
    tool_msg = llm.seen[1][-1]                  # the tool result was fed back to the LLM
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"])["results"]


def test_prefetch_retrieves_before_first_llm_call(env):
    llm = ScriptedLLM([LLMResponse("Answer.")])
    result = Agent(llm, env["tools"]).run("When is a shipment declared lost?")
    first_call = llm.seen[0]  # what the LLM saw on its very first turn
    assert [m["role"] for m in first_call] == ["system", "user", "assistant", "tool"]
    assert result.tool_calls[0].name == "search_policy" and result.sources


def test_prefetch_adds_order_tools_when_an_order_id_is_mentioned():
    names = [(c.name, c.arguments) for c in prefetch_calls("Is order #75939 late? And Order ID 12?")]
    assert names == [("search_policy", {"query": "Is order #75939 late? And Order ID 12?"}),
                     ("lookup_shipment", {"order_id": 75939}), ("predict_risk", {"order_id": 75939}),
                     ("lookup_shipment", {"order_id": 12}), ("predict_risk", {"order_id": 12})]
    assert [c.name for c in prefetch_calls("What is the refund policy?")] == ["search_policy"]


def test_made_up_citations_are_flagged(env):
    query = "lost shipment carrier scan"
    real = env["tools"].search_policy(query)["results"][0]["citation"]  # a source the search will return
    llm = ScriptedLLM([LLMResponse("", [ToolCall("c1", "search_policy", {"query": query})]),
                       LLMResponse(f"Real: [{real}]. Fake: [99_made_up.md § Nowhere].")])
    result = Agent(llm, env["tools"], prefetch=False).run("When is a shipment lost?")
    assert result.invalid_citations == ["99_made_up.md § Nowhere"]  # the real one passes, the fake is flagged


def test_agent_stops_at_step_limit(env):
    looping = [LLMResponse("", [ToolCall(f"c{i}", "lookup_shipment", {"order_id": 1})]) for i in range(3)]
    result = Agent(ScriptedLLM(looping), env["tools"], max_steps=3).run("loop forever")
    assert result.steps == 3 and "step limit" in result.answer


@pytest.fixture(scope="module")
def ask_client(env):
    model_path, metrics_path = env["tmp"] / "m.joblib", env["tmp"] / "metrics.json"
    joblib.dump(env["model"], model_path)
    metrics_path.write_text(json.dumps({"models": {MODEL_NAME: {"val": {"threshold": 0.5}}}}))
    replies = [LLMResponse("", [ToolCall("c1", "lookup_shipment", {"order_id": env["shipped_id"]})]),
               LLMResponse("It was delivered.")]
    app = create_app(model_path, metrics_path, agent_factory=lambda state: Agent(ScriptedLLM(replies), env["tools"]))
    with TestClient(app) as c:
        yield c


def test_ask_endpoint(ask_client):
    r = ask_client.post("/ask", json={"question": "What happened to my order?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "It was delivered." and body["llm_model"] == "scripted"
    assert [t["name"] for t in body["tool_calls"]] == ["search_policy", "lookup_shipment"]  # prefetch, then LLM
    assert body["invalid_citations"] == []


def test_ask_rejects_tiny_question(ask_client):
    assert ask_client.post("/ask", json={"question": "?"}).status_code == 422


def test_ask_returns_503_when_llm_down(env, tmp_path):
    class DownLLM:
        model = "down"

        def chat(self, messages, tools=None):
            raise LLMUnavailable("Ollama not reachable")

    model_path, metrics_path = tmp_path / "m.joblib", tmp_path / "metrics.json"
    joblib.dump(env["model"], model_path)
    metrics_path.write_text(json.dumps({"models": {MODEL_NAME: {"val": {"threshold": 0.5}}}}))
    app = create_app(model_path, metrics_path, agent_factory=lambda state: Agent(DownLLM(), env["tools"]))
    with TestClient(app) as c:
        r = c.post("/ask", json={"question": "anything at all"})
    assert r.status_code == 503 and "copilot unavailable" in r.json()["detail"]
