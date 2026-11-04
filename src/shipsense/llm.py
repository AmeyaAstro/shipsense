"""Provider-agnostic LLM client with tool calling.

Why: the agent should not care which model it talks to. Every provider implements one method,
`chat(messages, tools) -> LLMResponse`, using ONE internal message format (the OpenAI/Ollama
style: role + content, assistant `tool_calls`, and `tool` result messages).

Default: Ollama on this Mac (free, local).  Optional: Anthropic, if ANTHROPIC_API_KEY is set.
Choose with env vars: SHIPSENSE_LLM_PROVIDER=ollama|anthropic, SHIPSENSE_LLM_MODEL=<model name>.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from shipsense.config import SEED

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


class LLMUnavailable(RuntimeError):
    """The model server can't be reached (e.g. Ollama not running). The API maps this to 503."""


class OllamaLLM:
    """Talks to Ollama's /api/chat. temperature=0 + fixed seed so answers are repeatable."""

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, host: str | None = None, timeout: float = 180):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        body = {"model": self.model, "messages": [self._to_ollama(m) for m in messages], "stream": False,
                "options": {"temperature": 0, "seed": SEED}}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        try:
            r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        except httpx.ConnectError as e:
            raise LLMUnavailable(f"Ollama not reachable at {self.host} (is it running?)") from e
        r.raise_for_status()
        msg = r.json()["message"]
        calls = [ToolCall(c.get("id") or f"call_{i}", c["function"]["name"], c["function"].get("arguments") or {})
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        return LLMResponse(msg.get("content") or "", calls)

    @staticmethod
    def _to_ollama(m: dict) -> dict:
        if m["role"] == "assistant" and m.get("tool_calls"):
            return {"role": "assistant", "content": m.get("content", ""),
                    "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in m["tool_calls"]]}
        if m["role"] == "tool":
            return {"role": "tool", "content": m["content"], "tool_name": m["name"]}
        return {"role": m["role"], "content": m["content"]}


class AnthropicLLM:
    """Claude via the Messages API. Optional; needs ANTHROPIC_API_KEY (never committed)."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, api_key: str | None = None, timeout: float = 120):
        self.model = model
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        body = {"model": self.model, "max_tokens": 1024, "temperature": 0, "system": system,
                "messages": self._to_anthropic([m for m in messages if m["role"] != "system"])}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                             for t in tools]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        r = httpx.post(self.URL, json=body, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        blocks = r.json()["content"]
        text = "".join(b["text"] for b in blocks if b["type"] == "text")
        calls = [ToolCall(b["id"], b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"]
        return LLMResponse(text, calls)

    @staticmethod
    def _to_anthropic(messages: list[dict]) -> list[dict]:
        """Assistant tool calls become tool_use blocks; tool results become a user turn of tool_result blocks."""
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)  # several results from one turn share one user message
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out


def get_llm() -> LLM:
    """Pick the provider from env vars; default is local Ollama."""
    provider = os.getenv("SHIPSENSE_LLM_PROVIDER", "ollama").lower()
    model = os.getenv("SHIPSENSE_LLM_MODEL")
    if provider == "anthropic":
        return AnthropicLLM(model or DEFAULT_ANTHROPIC_MODEL)
    if provider == "ollama":
        return OllamaLLM(model or DEFAULT_OLLAMA_MODEL)
    raise ValueError(f"unknown SHIPSENSE_LLM_PROVIDER={provider!r} (use 'ollama' or 'anthropic')")
