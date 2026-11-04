# Ops copilot evaluation

25 questions in `eval/questions.jsonl` · LLM `qwen2.5:7b` via Ollama (temperature 0, seed 42) · embeddings `all-MiniLM-L6-v2` + FAISS · scored by `eval/run_eval.py` · Apple M4, 16 GB.
Knowledge base is **fictional** (`knowledge/`). Order questions use real DataCo orders.

## Retrieval (no LLM)

Questions with an expected source document: 17.

| hit@1 | hit@4 | MRR@4 |
|---|---|---|
| 82% | 100% | 0.90 |

Not ranked first: p02, p06, p13.

## End-to-end answers

| Metric | Prefetch ON (default) | Prefetch OFF (model decides) |
|---|---|---|
| Answer correct (all 25) | 96% | 92% |
| · mixed (2) | 100% | 50% |
| · order (5) | 100% | 100% |
| · policy (15) | 100% | 93% |
| · unanswerable (3) | 67% | 100% |
| Cites the expected doc (policy + mixed) | 88% | 82% |
| Answers containing an invented citation | 1 | 3 |
| Required order tools used (order + mixed) | 100% | 100% |
| Median / max latency per question | 12 s / 24 s | 13 s / 38 s |

With prefetch ON the order tools are called by the agent's code whenever an order id is mentioned, so 'required tools used' measures the design there, not the model's judgement.

## Run history (read this before quoting numbers)

| Run | Change | Prefetch ON correct | Prefetch OFF correct | Invented citations ON / OFF |
|---|---|---|---|---|
| 1 | first full run | 23/25 (92%) | 23/25 (92%) | 1 / 3 |
| 2 | `predict_risk` message for never-shipped orders fixed (was "not found") | 24/25 (96%) | 23/25 (92%) | 1 / 3 |

Run 1's o04 failure (a canceled order) was caused by a misleading tool message, which was fixed. That fix
was found **using this question set**, so run 2 is not a fully unseen test: treat 96% as optimistic. No other
change was made to rescue a failing question (u01 is left failing on purpose). A fresh held-out question set
would be the honest next step. Run 1 raw answers: `reports/rag_eval_results_run1.json`.

## How to read this

- **Correct** = every required fact (e.g. `25%`, `2 business hours`) appears in the answer. It is strict about facts and blind to wording; a right answer phrased unusually can be marked wrong, and an answer could contain the fact while also saying something wrong. Spot-check the raw answers.
- **Unanswerable** questions are 'correct' when the answer says it doesn't know / isn't covered.
- 25 questions is a small set: one question is 4 percentage points. Treat differences of 1-2 questions as noise.

## Failures (prefetch ON)

- **u01** What is the policy on shipping live animals?  
  tools: ['search_policy'] · invalid citations: ['07_shipping_mode_upgrade_policy.md § Approval limits, When upgrades are allowed']  
  answer: According to ShipSense's policies, shipping live animals is considered a high-risk order. Here are the key points:  1. **Proactive Notice:** Email the customer a realistic delivery window within 2 business hours of the order being placed. 2. **Upgrade Check:** If the order value is above $500 and it ships Standard Class, the Fulfilment Lead may upgrade it to Second Class at our cost (see Shipping 

Raw answers for both settings: `reports/rag_eval_results.json`.
