# ShipSense — resume bullets

Every number below was measured in this repo; the **Evidence** line under each bullet says where. Pick
3–5 bullets per application and tailor them to the role. Don't round the numbers up.

**Project line:**
> **ShipSense — Late-Delivery Risk Prediction & LLM Ops Copilot** · Python, SQL, PyTorch, scikit-learn,
> PySpark, FastAPI, Docker, AWS (Lambda, S3, ECR), MLflow, GitHub Actions, FAISS, Ollama ·
> github.com/AmeyaAstro/shipsense

## Core bullets (use these by default)

1. **Built a leakage-safe late-delivery risk model** on 180K supply-chain records (62,897 orders).
   Engineered order-level features in SQL with a time-based split, and improved ROC-AUC from 0.746
   (rule baseline) to 0.782 with gradient boosting.
   - *Evidence:* `reports/metrics.json`, `docs/week1_walkthrough.md`.
2. **Deployed the model as a FastAPI service on AWS Lambda** (arm64 container in ECR, model versioned in
   S3, least-privilege IAM, HTTPS Function URL, scripted deploy/teardown). Cut cold-start latency from
   ~26 s to 7.7 s by diagnosing an init timeout in CloudWatch logs; warm p95 73 ms.
   - *Evidence:* `docs/aws_deploy.md` "Measured on the real deploy".
3. **Built a RAG + tool-calling LLM copilot** (MiniLM + FAISS retrieval with citations, local
   `qwen2.5:7b`, tools for SQLite lookups and the risk model) with a 25-question eval: retrieval hit@4
   100%, and 24/25 answers correct with deterministic tool prefetch vs 23/25 without.
   - *Evidence:* `reports/rag_eval.md`.
   - *If asked:* the 24/25 run came after a fix found with the same questions, so it's optimistic.
4. **Set up MLOps:**
   - MLflow tracking for 4 models
   - Docker (slim, non-root, health-checked)
   - GitHub Actions CI with ruff and 61 offline pytest tests, including API contract and train/serve
     consistency checks
   - PSI drift monitoring.
   - *Evidence:* `.github/workflows/ci.yml`, `tests/`, `docs/week2_walkthrough.md`.

## Role-specific alternatives

**Data Scientist**
- **Benchmarked a PyTorch entity-embedding network against scikit-learn models.** It tied gradient
  boosting (0.780 vs 0.782 AUC) but trained ~4x slower, so I chose gradient boosting for production on
  the evidence. *Evidence:* `reports/model_comparison.md`.
- **Identified label-leaking post-delivery fields** that inflate accuracy to ~98% on this dataset,
  excluded them, and enforced the exclusion with a unit test.
  *Evidence:* `notebooks/01_eda.ipynb`, `tests/test_pipeline.py`.

**ML Engineer / MLOps**
- **Built PSI-based drift monitoring** for features, predictions and performance. Real monthly batches
  drifted heavily (feature PSI 4.5–12) while AUC held at 0.754–0.792, so I redesigned alerting around
  prediction drift and AUC drop to avoid alert fatigue.
  *Evidence:* `reports/drift/summary.md`, `src/shipsense/drift.py`.
- **Served predictions with Pydantic-validated single and batch endpoints:** 8.2 ms median per request
  and ~29 ms per 1,000-order batch locally in Docker.
  *Evidence:* `docs/week2_walkthrough.md` §3.

**Data Engineer**
- **Re-implemented the SQL feature pipeline in PySpark** (aggregations + window functions) and verified
  exact parity with SQL on all 62,897 orders, enforced by a CI test.
  *Evidence:* `spark/order_features_spark.py`, `tests/test_spark_features.py`.

**LLM / GenAI**
- **Diagnosed LLM agent failure modes in live testing** (skipped tool calls, hallucinated policies and
  citations) and added guardrails: deterministic tool prefetch, and verifying citations against the
  sources actually retrieved. The agent is covered by offline tests with a scripted fake LLM.
  *Evidence:* `docs/week3_walkthrough.md` §6, `reports/rag_eval.md`.

**Python SWE**
- **Designed an LLM client** behind one interface (Ollama default, optional Anthropic), with dependency
  injection so 28 LLM/RAG tests run offline in CI without a model server or downloads.
  *Evidence:* `src/shipsense/llm.py`, `tests/test_llm.py`, `tests/test_agent.py`.

## Don't claim
- **"Production" traffic or users.** It's a portfolio deployment, measured with test requests.
- **"Deployed the copilot to AWS."** Only the prediction API was deployed; the copilot runs locally.
- **"Claude/Anthropic integration tested."** That adapter is unit-tested but was never run against the
  live API.
- **"20x faster".** The measured training-time ratio is ~4x.
- **"Real company policies".** The `knowledge/` docs are fictional.
