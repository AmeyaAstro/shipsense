# CLAUDE.md — ShipSense project brief (read fully before doing anything)

## Who this is for
Ameya Joshi — MS CS (Data Science concentration), UNC Charlotte, graduating May 2027, F-1/OPT.
This repo is a **portfolio project built to close skill gaps seen in real job applications**
(DS / MLE / DE / Python SWE roles): PyTorch, SQL, Spark, MLOps (MLflow, Docker, CI/CD),
LLMs/RAG/agents, AWS, monitoring. Every finished milestone becomes resume bullets, so:

- **Truthful only.** Never claim a result you didn't measure. Report metrics exactly as produced.
- **Ameya must be able to defend every line in interviews.** He self-reports weak coding
  fundamentals. After each milestone, write `docs/weekN_walkthrough.md`: what each file does,
  why each design choice, how to run it, 3 truthful resume bullets, and ~10 interview questions.
  Explain in plain language as you work; prefer simple, readable code over clever code.
- Ask before: pushing to GitHub, creating cloud resources, anything that costs money, adding
  secrets. Never commit data, `.env`, API keys, or model binaries.

## Repo / environment
- GitHub: https://github.com/AmeyaAstro/shipsense (not pushed yet — see `docs/push_to_github.md`)
- Local: `~/Desktop/Claude Job Hunt/Skill gap project shipsense/shipsense_week1/` (this folder is the repo root)
- Dataset: DataCo Smart Supply Chain (Kaggle), at `../archive/DataCoSupplyChainDataset.csv`
  (Latin-1 encoded, 180,519 item rows). `config.py` finds it automatically.
- Mac (Apple Silicon). Python venv in `.venv`. Setup: `make setup`, then `make all`.

## Week 1 — DONE (do not redo; build on it)
- Model at **order level** (62,897 shipped orders, 57% late). Canceled/SUSPECTED_FRAUD dropped.
- **Leakage excluded**: `Days for shipping (real)`, `Delivery Status`, shipping date, `Order Status`,
  profit/benefit fields. Label == real days > scheduled days. A test enforces this.
- Features in SQL: `sql/order_features.sql` (SQLite), loaded by `src/shipsense/data.py`.
- Time split: train 2015–16, val Jan–Jun 2017, test Jul 2017–Jan 2018. Threshold tuned on val (Youden's J).
- Test ROC-AUC: shipping-mode rule 0.746 · logistic 0.756 · **HistGradientBoosting 0.782 (production model)** ·
  PyTorch embeddings MLP 0.780 (~4x slower to train: 8.8–9.4 s vs 2.1–2.4 s on Apple M4; an earlier "20x" was wrong).
- Known data property: within a shipping mode, actual days are ~uniform 2–6, so ~0.78 AUC is near
  the ceiling. Don't chase accuracy with leaky features.

## Week 2 — DONE (MLflow, FastAPI, Docker, CI, PySpark parity; 25 tests; see docs/week2_walkthrough.md)
1. **MLflow**: log params/metrics/artifacts for all 4 models in `train.py` (local `mlruns/`, gitignored).
2. **FastAPI** `src/shipsense/api.py`: `GET /health`, `POST /predict` (Pydantic schema for one order's
   order-time fields → `{late_probability, is_late_risk, threshold, model_version}`), `POST /predict/batch`.
   Load the GBM pipeline + threshold from `models/` and `reports/metrics.json`.
3. **Tests**: API tests with `fastapi.testclient` (valid input, invalid input → 422, batch).
   CI must not need the 96 MB CSV — train a tiny model on synthetic data in a fixture (see `tests/test_pipeline.py::make_items`).
4. **Docker**: slim Python image, non-root user, `uvicorn` entrypoint; `docker-compose.yml` optional.
5. **GitHub Actions** `.github/workflows/ci.yml`: ruff lint + pytest on push/PR (Python 3.11).
6. **PySpark** `spark/order_features_spark.py`: same order-level features as the SQL, run locally;
   a test or script asserting it matches the SQL output on a sample. (Requires Java on the Mac — check `java -version`; if missing, tell Ameya how to install, e.g. `brew install openjdk@17`.)
7. Update README (architecture diagram in Mermaid, run instructions) + `docs/week2_walkthrough.md`.

## Week 3 — RAG + LLM agent
1. Write ~8–12 short **synthetic** shipping SOP/policy markdown docs in `knowledge/` (clearly labelled
   fictional: late-shipment escalation, carrier SLAs by mode, refund/credit policy, customs delays, etc.).
2. RAG: chunk → embed with a Hugging Face sentence-transformer (`all-MiniLM-L6-v2`) → vector store
   (Chroma or FAISS) → retrieval with citations.
3. LLM: provider-agnostic wrapper. Default to **Ollama** locally (Ameya has it installed; e.g. `llama3.1:8b`
   or `qwen2.5:7b`) so it's free; optional OpenAI/Anthropic via env var key.
4. Agent with tools: `predict_risk(order)`, `lookup_shipment(order_id)` (queries SQLite),
   `search_policy(query)`. Simple, readable tool-calling loop (or LangGraph if it stays small).
   Expose via `POST /ask` in FastAPI.
5. Evaluation: `eval/questions.jsonl` (~25 Q&A) measuring retrieval hit-rate and answer correctness
   (exact-match / keyword or LLM-as-judge); write results to `reports/rag_eval.md`. Report honestly.
6. `docs/week3_walkthrough.md`.

## Week 4 — deploy + monitor + showcase
1. AWS free tier (ask before creating anything; give Ameya console steps where he must click):
   model artifact in S3; container on ECR → App Runner or ECS Fargate (or Lambda container if cheaper).
   Include teardown instructions and a cost note.
2. Drift monitoring: script comparing a new batch vs training distribution (PSI per feature +
   prediction drift), e.g. with Evidently or hand-rolled PSI; save an HTML/markdown report; a scheduled GitHub Action is a plus.
3. Simple Streamlit demo UI (risk prediction + ask-the-copilot).
4. Final README polish (results, architecture, screenshots placeholder), demo-video script,
   LinkedIn post draft, final resume bullets → `docs/week4_walkthrough.md` and `docs/resume_bullets.md`.

## Conventions
- Code in `src/shipsense/`, tests in `tests/`, `pyproject.toml` packaging, pinned-floor deps in `requirements.txt`.
- Keep every module small with a docstring explaining *why*. Type hints. Seed everything (SEED=42).
- Run `pytest -q` after every change; keep CI green.
- Commit in small logical steps with clear messages (when Ameya has approved pushing).
