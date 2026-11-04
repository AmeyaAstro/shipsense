"""LLM adapter tests: httpx.post is faked, so no model server or API key is needed."""
import httpx
import pytest

from shipsense import llm
from shipsense.llm import AnthropicLLM, LLMUnavailable, OllamaLLM, ToolCall, get_llm

TOOL = {"name": "predict_risk", "description": "risk for an order",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "integer"}}}}


class FakePost:
    """Records the request and returns a canned JSON response."""

    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def __call__(self, url, json, timeout, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return httpx.Response(200, json=self.payload, request=httpx.Request("POST", url))


def test_ollama_parses_tool_calls_and_sends_tools(monkeypatch):
    fake = FakePost({"message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "function": {"name": "predict_risk", "arguments": {"order_id": 7}}}]}})
    monkeypatch.setattr(llm.httpx, "post", fake)
    resp = OllamaLLM().chat([{"role": "user", "content": "risk of order 7?"}], tools=[TOOL])
    assert resp.tool_calls == [ToolCall("call_1", "predict_risk", {"order_id": 7})]
    sent = fake.calls[0]["json"]
    assert sent["tools"][0] == {"type": "function", "function": TOOL}
    assert sent["options"]["temperature"] == 0 and sent["stream"] is False


def test_ollama_converts_tool_messages(monkeypatch):
    fake = FakePost({"message": {"role": "assistant", "content": "done"}})
    monkeypatch.setattr(llm.httpx, "post", fake)
    call = ToolCall("call_1", "predict_risk", {"order_id": 7})
    OllamaLLM().chat([{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "", "tool_calls": [call]},
                      {"role": "tool", "tool_call_id": "call_1", "name": "predict_risk", "content": "{}"}])
    msgs = fake.calls[0]["json"]["messages"]
    assert msgs[1]["tool_calls"] == [{"function": {"name": "predict_risk", "arguments": {"order_id": 7}}}]
    assert msgs[2] == {"role": "tool", "content": "{}", "tool_name": "predict_risk"}


def test_ollama_down_raises_llm_unavailable(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(llm.httpx, "post", refuse)
    with pytest.raises(LLMUnavailable):
        OllamaLLM().chat([{"role": "user", "content": "hi"}])


def test_anthropic_message_translation(monkeypatch):
    fake = FakePost({"content": [{"type": "text", "text": "Checking. "},
                                 {"type": "tool_use", "id": "tu_1", "name": "predict_risk", "input": {"order_id": 7}}]})
    monkeypatch.setattr(llm.httpx, "post", fake)
    calls = [ToolCall("a", "predict_risk", {"order_id": 1}), ToolCall("b", "predict_risk", {"order_id": 2})]
    resp = AnthropicLLM(api_key="test-key").chat([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "a", "name": "predict_risk", "content": "r1"},
        {"role": "tool", "tool_call_id": "b", "name": "predict_risk", "content": "r2"},
    ], tools=[TOOL])
    sent = fake.calls[0]["json"]
    assert sent["system"] == "be brief"
    assert sent["tools"][0]["input_schema"] == TOOL["parameters"]
    assert [b["type"] for b in sent["messages"][1]["content"]] == ["tool_use", "tool_use"]
    assert sent["messages"][2] == {"role": "user", "content": [   # both results merged into one user turn
        {"type": "tool_result", "tool_use_id": "a", "content": "r1"},
        {"type": "tool_result", "tool_use_id": "b", "content": "r2"}]}
    assert fake.calls[0]["headers"]["x-api-key"] == "test-key"
    assert resp.content == "Checking. " and resp.tool_calls == [ToolCall("tu_1", "predict_risk", {"order_id": 7})]


def test_get_llm_reads_env(monkeypatch):
    monkeypatch.delenv("SHIPSENSE_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("SHIPSENSE_LLM_MODEL", "llama3.1:8b")
    assert isinstance(get_llm(), OllamaLLM) and get_llm().model == "llama3.1:8b"
    monkeypatch.setenv("SHIPSENSE_LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        get_llm()
