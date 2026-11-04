"""Evaluate the ops copilot on eval/questions.jsonl and write reports/rag_eval.md.

Run:  make rag-eval    (needs `make data train copilot-data` and Ollama running; ~15-25 min)

Scoring is deliberately simple and transparent (no LLM-as-judge):
- retrieval hit@1 / hit@4: the expected doc is the top / a top-4 search result (no LLM involved)
- correct: every `must_include` group has at least one of its phrases in the answer (case-insensitive)
- cites expected doc: a VALID citation (retrieved, not invented) from the expected doc is in the answer
- invalid citations: [doc § section] in the answer that were never retrieved
- required tools: the order tools the question needs appear in the trace
Both agent settings are run on the same questions: prefetch on (default) and off (model decides).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import joblib

from shipsense.agent import CITATION, Agent
from shipsense.api import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, MODEL_NAME, file_version
from shipsense.config import DB_PATH, ROOT
from shipsense.llm import OllamaLLM
from shipsense.rag import INDEX_DIR, PolicyIndex, SentenceTransformerEmbedder
from shipsense.tools import ShipSenseTools

QUESTIONS = ROOT / "eval" / "questions.jsonl"

# Kept in the report so regenerating it never hides how the numbers were reached.
HISTORY = """## Run history (read this before quoting numbers)

| Run | Change | Prefetch ON correct | Prefetch OFF correct | Invented citations ON / OFF |
|---|---|---|---|---|
| 1 | first full run | 23/25 (92%) | 23/25 (92%) | 1 / 3 |
| 2 | `predict_risk` message for never-shipped orders fixed (was "not found") | 24/25 (96%) | 23/25 (92%) | 1 / 3 |

Run 1's o04 failure (a canceled order) was caused by a misleading tool message, which was fixed. That fix
was found **using this question set**, so run 2 is not a fully unseen test: treat 96% as optimistic. No other
change was made to rescue a failing question (u01 is left failing on purpose). A fresh held-out question set
would be the honest next step. Run 1 raw answers: `reports/rag_eval_results_run1.json`."""
REPORT = ROOT / "reports" / "rag_eval.md"
RAW = ROOT / "reports" / "rag_eval_results.json"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(q: dict, answer: str, sources: list[str], tool_names: list[str], invalid: list[str]) -> dict:
    text = answer.lower()
    valid_cites = [c for c in CITATION.findall(answer) if c in sources]
    return {
        "correct": all(any(p.lower() in text for p in group) for group in q["must_include"]),
        "cites_expected_doc": bool(q["expected_doc"]) and any(c.startswith(q["expected_doc"]) for c in valid_cites),
        "invalid_citations": len(invalid),
        "required_tools_used": all(t in tool_names for t in q["required_tools"]),
    }


def retrieval_metrics(index: PolicyIndex, questions: list[dict]) -> dict:
    rows = [q for q in questions if q["expected_doc"]]
    ranks = []
    for q in rows:
        docs = [h.chunk.doc for h in index.search(q["question"], k=4)]
        ranks.append(docs.index(q["expected_doc"]) + 1 if q["expected_doc"] in docs else None)
    return {"n": len(rows), "hit@1": sum(r == 1 for r in ranks) / len(rows),
            "hit@4": sum(r is not None for r in ranks) / len(rows),
            "mrr@4": sum(1 / r for r in ranks if r) / len(rows),
            "misses": [q["id"] for q, r in zip(rows, ranks, strict=True) if r != 1]}


def run_agent(agent: Agent, questions: list[dict]) -> list[dict]:
    out = []
    for q in questions:
        t0 = time.perf_counter()
        res = agent.run(q["question"])
        secs = time.perf_counter() - t0
        names = [t.name for t in res.tool_calls]
        row = {"id": q["id"], "category": q["category"], "question": q["question"], "answer": res.answer,
               "tools": names, "sources": res.sources, "invalid": res.invalid_citations,
               "steps": res.steps, "seconds": round(secs, 1),
               **score(q, res.answer, res.sources, names, res.invalid_citations)}
        out.append(row)
        print(f"  {q['id']} {'OK ' if row['correct'] else 'BAD'} {secs:5.1f}s  tools={names}", flush=True)
    return out


def summarise(rows: list[dict]) -> dict:
    def rate(rs, key):
        return sum(r[key] for r in rs) / len(rs) if rs else float("nan")
    cats = sorted({r["category"] for r in rows})
    with_doc = [r for r in rows if r["category"] in ("policy", "mixed")]
    needs_tools = [r for r in rows if r["category"] in ("order", "mixed")]
    return {"correct": rate(rows, "correct"), "by_category": {c: rate([r for r in rows if r["category"] == c],
                                                                       "correct") for c in cats},
            "cites_expected_doc": rate(with_doc, "cites_expected_doc"),
            "answers_with_invalid_citation": sum(r["invalid_citations"] > 0 for r in rows),
            "invalid_citations": sum(r["invalid_citations"] for r in rows),
            "required_tools_used": rate(needs_tools, "required_tools_used"),
            "median_seconds": statistics.median(r["seconds"] for r in rows),
            "max_seconds": max(r["seconds"] for r in rows)}


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def write_report(retrieval: dict, results: dict, llm_model: str, n: int) -> None:
    on, off = summarise(results["prefetch_on"]), summarise(results["prefetch_off"])
    cats = list(on["by_category"])
    counts = {c: sum(r["category"] == c for r in results["prefetch_on"]) for c in cats}
    lines = [
        "# Ops copilot evaluation",
        "",
        f"{n} questions in `eval/questions.jsonl` · LLM `{llm_model}` via Ollama (temperature 0, seed 42) · "
        "embeddings `all-MiniLM-L6-v2` + FAISS · scored by `eval/run_eval.py` · Apple M4, 16 GB.",
        "Knowledge base is **fictional** (`knowledge/`). Order questions use real DataCo orders.",
        "",
        "## Retrieval (no LLM)",
        "",
        f"Questions with an expected source document: {retrieval['n']}.",
        "",
        "| hit@1 | hit@4 | MRR@4 |",
        "|---|---|---|",
        f"| {pct(retrieval['hit@1'])} | {pct(retrieval['hit@4'])} | {retrieval['mrr@4']:.2f} |",
        "",
        f"Not ranked first: {', '.join(retrieval['misses']) or 'none'}.",
        "",
        "## End-to-end answers",
        "",
        "| Metric | Prefetch ON (default) | Prefetch OFF (model decides) |",
        "|---|---|---|",
        f"| Answer correct (all {n}) | {pct(on['correct'])} | {pct(off['correct'])} |",
        *[f"| · {c} ({counts[c]}) | {pct(on['by_category'][c])} | {pct(off['by_category'][c])} |" for c in cats],
        f"| Cites the expected doc (policy + mixed) | {pct(on['cites_expected_doc'])} "
        f"| {pct(off['cites_expected_doc'])} |",
        f"| Answers containing an invented citation | {on['answers_with_invalid_citation']} "
        f"| {off['answers_with_invalid_citation']} |",
        f"| Required order tools used (order + mixed) | {pct(on['required_tools_used'])} "
        f"| {pct(off['required_tools_used'])} |",
        f"| Median / max latency per question | {on['median_seconds']:.0f} s / {on['max_seconds']:.0f} s "
        f"| {off['median_seconds']:.0f} s / {off['max_seconds']:.0f} s |",
        "",
        "With prefetch ON the order tools are called by the agent's code whenever an order id is mentioned, "
        "so 'required tools used' measures the design there, not the model's judgement.",
        "",
        HISTORY,
        "",
        "## How to read this",
        "",
        "- **Correct** = every required fact (e.g. `25%`, `2 business hours`) appears in the answer. It is strict "
        "about facts and blind to wording; a right answer phrased unusually can be marked wrong, and an answer "
        "could contain the fact while also saying something wrong. Spot-check the raw answers.",
        "- **Unanswerable** questions are 'correct' when the answer says it doesn't know / isn't covered.",
        "- 25 questions is a small set: one question is 4 percentage points. Treat differences of 1-2 "
        "questions as noise.",
        "",
        "## Failures (prefetch ON)",
        "",
    ]
    bad = [r for r in results["prefetch_on"] if not r["correct"] or r["invalid_citations"]]
    for r in bad or []:
        lines += [f"- **{r['id']}** {r['question']}  ", f"  tools: {r['tools']} · invalid citations: {r['invalid']}  ",
                  f"  answer: {r['answer'][:400].replace(chr(10), ' ')}"]
    if not bad:
        lines.append("None.")
    lines += ["", "Raw answers for both settings: `reports/rag_eval_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    questions = load_questions()
    model = joblib.load(DEFAULT_MODEL_PATH)
    threshold = json.loads(DEFAULT_METRICS_PATH.read_text())["models"][MODEL_NAME]["val"]["threshold"]
    index = PolicyIndex.load(SentenceTransformerEmbedder(), INDEX_DIR)
    tools = ShipSenseTools(DB_PATH, model, threshold, file_version(DEFAULT_MODEL_PATH), index)
    llm = OllamaLLM()

    retrieval = retrieval_metrics(index, questions)
    print(f"retrieval: hit@1 {retrieval['hit@1']:.2f}  hit@4 {retrieval['hit@4']:.2f}", flush=True)
    results = {}
    for name, prefetch in (("prefetch_on", True), ("prefetch_off", False)):
        print(f"{name}:", flush=True)
        results[name] = run_agent(Agent(llm, tools, prefetch=prefetch), questions)
    RAW.write_text(json.dumps({"retrieval": retrieval, **results}, indent=1, ensure_ascii=False))
    write_report(retrieval, results, llm.model, len(questions))
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
