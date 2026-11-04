"""The ops copilot: a small, readable tool-calling loop.

Loop: send question + tool list -> if the LLM asks for tools, run them and send results back ->
repeat until it answers in plain text (or MAX_STEPS is hit, so it can never loop forever).
Sources are collected from search_policy results, so every answer carries its citations.

Two guardrails, added after testing with qwen2.5:7b showed it sometimes skipped retrieval and
invented policies and citations:
- prefetch: before the first LLM call, gather what we can detect deterministically: policy passages
  for the question, and if it mentions an order number, that order's shipment record and model risk.
  The model still writes the answer and may call more tools. Switchable; the eval measures both.
- citation check: every [doc § section] in the answer must be one of the retrieved passages;
  any that isn't is reported in `invalid_citations` instead of being passed off as a real source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from shipsense.llm import LLM, ToolCall
from shipsense.tools import TOOL_SPECS, ShipSenseTools, to_json

MAX_STEPS = 6

SYSTEM_PROMPT = """You are the ShipSense Ops Copilot for ShipSense Logistics (a fictional company).
Answer operations questions using your tools; never invent facts.
- Policy or procedure questions: answer ONLY from search_policy results (call it again with a better query
  if the results don't cover the question). Cite each policy fact with its citation in square brackets,
  copied exactly, e.g. [03_refund_and_credit_policy.md § Credit amounts]. Never invent a source.
- Questions about a specific order: call lookup_shipment for what happened and predict_risk for the
  model's late-delivery risk. Report numbers exactly as the tools return them; don't cite them.
- If the tools don't contain the answer, say you don't know. Keep answers short and concrete."""

CITATION = re.compile(r"\[([^\[\]]+?\.md § [^\[\]]+?)\]")
ORDER_ID = re.compile(r"\border\s*(?:id\s*)?#?\s*(\d{1,7})\b", re.IGNORECASE)


@dataclass
class ToolTrace:
    name: str
    arguments: dict
    result: dict


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[ToolTrace] = field(default_factory=list)
    steps: int = 0
    invalid_citations: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LLM, tools: ShipSenseTools, max_steps: int = MAX_STEPS, prefetch: bool = True):
        self.llm, self.tools, self.max_steps, self.prefetch = llm, tools, max_steps, prefetch

    def run(self, question: str) -> AgentResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        result = AgentResult(answer="")
        if self.prefetch:  # shown to the model as if it had called these tools itself
            calls = prefetch_calls(question)
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            for call in calls:
                self._run_tool(call, messages, result)
        for step in range(1, self.max_steps + 1):
            result.steps = step
            reply = self.llm.chat(messages, tools=TOOL_SPECS)
            if not reply.tool_calls:  # plain text = final answer
                result.answer = reply.content.strip()
                result.invalid_citations = [c for c in CITATION.findall(result.answer) if c not in result.sources]
                return result
            messages.append({"role": "assistant", "content": reply.content, "tool_calls": reply.tool_calls})
            for call in reply.tool_calls:
                self._run_tool(call, messages, result)
        result.answer = "Sorry, I couldn't finish answering within the step limit."
        return result

    def _run_tool(self, call: ToolCall, messages: list[dict], result: AgentResult) -> None:
        output = self.tools.execute(call.name, call.arguments)
        result.tool_calls.append(ToolTrace(call.name, call.arguments, output))
        for hit in output.get("results", []):  # search_policy hits -> sources
            if hit["citation"] not in result.sources:
                result.sources.append(hit["citation"])
        messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": to_json(output)})


def prefetch_calls(question: str) -> list[ToolCall]:
    """Tool calls we can decide on without the LLM: always a policy search, plus order tools per order id."""
    calls = [ToolCall("prefetch_policy", "search_policy", {"query": question})]
    for order_id in dict.fromkeys(int(m) for m in ORDER_ID.findall(question)):  # unique, in order
        calls.append(ToolCall(f"prefetch_ship_{order_id}", "lookup_shipment", {"order_id": order_id}))
        calls.append(ToolCall(f"prefetch_risk_{order_id}", "predict_risk", {"order_id": order_id}))
    return calls
