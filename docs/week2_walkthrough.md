# Week 2 walkthrough — from "a model on my laptop" to a tested, containerised service

Budget about 90 minutes. Open each file in VS Code as you read, and run the commands yourself.

## 1. The big picture
Week 1 ended with a good model saved in a file. Nobody can use a file. Week 2 answers four
questions every ML team asks:

| Question | Tool | File |
|---|---|---|
| "Which experiment produced this model, with what settings?" | MLflow | `src/shipsense/tracking.py` |
| "How do other systems get a prediction?" | FastAPI + Pydantic | `src/shipsense/api.py` |
| "Will it run the same on another machine?" | Docker | `Dockerfile`, `requirements-serve.txt` |
| "How do we know a change didn't break anything?" | pytest + GitHub Actions | `tests/`, `.github/workflows/ci.yml` |

Plus one data-engineering question: "What if the data outgrows one machine?" → PySpark
(`spark/order_features_spark.py`).

## 2. MLflow experiment tracking — `tracking.py`, `train.py`
- **Problem:** `reports/metrics.json` is overwritten on each run, so there was no history.
- **What happens now:** each `make train` creates one **parent run** (`train_all`: split sizes,
  feature list, the reports) with four **child runs**, one per model. Each child run records:
  - params (hyperparameters)
  - metrics (val + test ROC-AUC, PR-AUC, etc., threshold, train seconds)
  - artifacts (the saved model file)
  - for PyTorch only: train/val loss per epoch, so MLflow draws the learning curve.
- **Storage:** a local SQLite file `mlruns/mlflow.db` plus `mlruns/artifacts/`. MLflow 3 deprecated
  the older plain-folder store. The folder is gitignored because runs and model binaries are not code.
- **Try it:** `make mlflow-ui` → http://127.0.0.1:5000 → open `pytorch_mlp` → `val_loss` chart.

## 3. The API — `api.py`
Read it top to bottom; it's about 120 lines.

1. **`Order` (Pydantic model):** the request schema. It checks:
   - **Types:** `n_items` must be an int.
   - **Allowed values:** `shipping_mode` must be one of 4.
   - **Ranges:** e.g. `0 <= avg_discount_rate <= 1`.
   - **Cross-field rules** in `check_consistent`: `total_quantity >= n_items`, `n_departments <= n_items`.

   Anything else gets a **422** with a message saying exactly which field is wrong, and the model
   never sees bad data.
2. **`days_scheduled` is not in the request.** In DataCo it is fixed per shipping mode (verified:
   0 exceptions in 62,897 orders), so the server derives it from `DAYS_SCHEDULED_BY_MODE`. A client
   can't send a contradiction like "Same Day, 4 days".
3. **`to_features`** turns orders into the exact columns the model was trained on. It calls
   `add_calendar_features` from `data.py`, **the same function training uses**. Two copies of
   feature logic that drift apart is called **train/serve skew**, a classic production bug.
4. **`create_app` + `lifespan`:** the model is loaded **once at startup**, not per request. Paths
   come from arguments (tests), env vars (`SHIPSENSE_MODEL_PATH`), or defaults. Importing the module
   never loads a model, which is why CI can import it without one.
5. **`file_version`:** model version = name + first 12 chars of the file's SHA-256 hash. Every
   prediction can be traced to the exact binary that made it.
6. **Threshold:** read from `reports/metrics.json` (0.595, tuned on validation in Week 1).
   `is_late_risk = late_probability >= threshold`, the same rule as `evaluate.score`.
7. **Endpoints:**
   - `GET /health` (is it alive, which model?)
   - `POST /predict` (one order)
   - `POST /predict/batch` (1–1000 orders, scored in one vectorised call)

**Measured (Docker on Apple M4, localhost):** `/predict` median 8.2 ms, p95 14.3 ms (500 sequential
requests); a 1,000-order batch takes a median of 29 ms, about 280x cheaper per order than single calls.

## 4. Tests — `tests/test_api.py`, `tests/test_spark_features.py`
- CI must not need the 96 MB CSV, so the `artifacts` fixture trains a **tiny GBM on synthetic
  orders** (`make_items` from Week 1) and writes a fake `metrics.json`.
- `TestClient` calls the app in-process. Using it with `with` runs startup, so the model loads.
- Most important test: **`test_api_matches_model_called_directly`**. The API score must equal
  `pipeline.predict_proba` on the same order. That test is what would catch train/serve skew.
- 7 invalid-input cases (via `parametrize`) must return 422. Batch results must equal single results.
- Total: **25 tests** (11 from Week 1, 12 API, 2 Spark).

## 5. Docker — `Dockerfile`, `.dockerignore`, `docker-compose.yml`
Read the Dockerfile line by line; every line has a comment. Key ideas:
- **`python:3.11-slim`:** small Debian base with Python.
- **`requirements-serve.txt`:** only what the API imports (no torch, MLflow, Jupyter). It is
  pinned to the **exact** versions the model was trained with, because a pickled sklearn model is
  only guaranteed to load and behave identically under the same version.
- **Requirements are copied before the code:** Docker caches each step, so a code change rebuilds
  in seconds without reinstalling libraries.
- **Non-root user (`appuser`, uid 10001):** if the app were compromised, it can't change the image.
  We verified that it cannot overwrite the model file.
- **`HEALTHCHECK`:** Docker pings `/health` every 30 s and marks the container healthy or unhealthy.
- **Networking:** `--host 0.0.0.0` inside the container accepts outside connections. `-p 127.0.0.1:8000:8000`
  exposes it only to this Mac, not to the Wi-Fi network.
- **The model is baked into the image:** one self-contained artifact to deploy (Week 4). The
  trade-off is that a new model means a new image.
- **Image size:** 738 MB, mostly numpy, scipy and scikit-learn.

## 6. CI — `.github/workflows/ci.yml`
On every push to `main` and every PR, a fresh Ubuntu machine on GitHub runs these steps in order:
1. Check out the code.
2. Set up Python 3.11 (with a pip cache).
3. Set up Java 17.
4. Install dependencies.
5. `ruff check .`
6. `pytest -q`.

Details:
- **CPU-only torch:** the default Linux torch wheel bundles ~2.5 GB of GPU (CUDA) libraries that a
  CI runner can't use.
- **`concurrency`:** a new push cancels the older, still-running check.
- **Ruff rules are explicit in `pyproject.toml`:** E/F (errors, unused names), I (import order),
  UP (modern syntax), B (likely bugs). The generated notebook is excluded, and its generator script
  is linted instead.
- **First run:** passed in 1 min 44 s. The badge at the top of the README shows the latest status.

## 7. PySpark — `spark/order_features_spark.py`
- Same features as `sql/order_features.sql`, written with Spark's DataFrame API:
  - `GROUP BY order_id` → `groupBy("order_id").agg(F.min(...), F.sum(...), F.countDistinct(...))`
  - "department of the most expensive item" (SQL correlated subquery) → a **window function**:
    `row_number()` over each order's items sorted by sales desc, department asc; keep row 1.
- `local[*]` = Spark runs on this Mac's cores. The same code runs on a cluster (EMR, Databricks).
- Session timezone pinned to UTC, so timestamps aren't shifted by daylight-saving rules.
- **Parity, two ways:**
  - `make spark-features ARGS=--compare` checks Spark vs SQLite on the real data. Result: **all
    62,897 orders match on every column.**
  - The CI test runs both pipelines on the same synthetic items. A second test changes a value on
    purpose and asserts that the comparison catches it, because a check that can't fail proves nothing.
- Decimals are compared with a 1e-6 relative tolerance: Spark sums numbers in a different order than
  SQLite, and float addition order changes the last digits. Integers and text must match exactly.
- Needs Java 17 (installed user-level at `~/.local/jdk`; the Makefile sets `JAVA_HOME`).

## 8. Things that went wrong (great interview stories — they're real)
1. **`ModuleNotFoundError: shipsense` after `make serve`.**
   - Cause: the Desktop is iCloud-synced, and iCloud set macOS's "hidden" flag on the venv's
     editable-install `.pth` file. Python (for security) skips hidden `.pth` files.
   - Why only some commands broke: tests still passed because pytest adds `src/` itself.
   - Fix: the Makefile exports `PYTHONPATH=src`. Lesson: understand *how* imports resolve.
2. **Container crashed with `PermissionError: config.py`.**
   - Cause: that file was owner-only on the Mac, and `COPY` keeps file modes, so the non-root user
     couldn't read it.
   - Fix: `chmod -R a+rX,go-w` in the Dockerfile, so the image no longer depends on laptop permissions.
     Running as non-root surfaced a hidden assumption.
3. **A wrong number in our own docs.**
   - Week 1 said PyTorch trains "~20x slower". Re-measuring gave ~4x (2.1–2.4 s vs 8.8–9.4 s).
     README and the Week 1 walkthrough were corrected.
   - Lesson: put only measured numbers on a resume.

## 9. How to run everything
```bash
make setup                 # once
make all                   # features -> train (logs to MLflow) -> 25 tests
make mlflow-ui             # http://127.0.0.1:5000
make serve                 # http://127.0.0.1:8000/docs  (Ctrl+C to stop)
make docker-build && make docker-run
make spark-features ARGS=--compare
.venv/bin/ruff check .     # same lint CI runs
```

## 10. Resume bullets this week unlocks (truthful, measured)
- Deployed the gradient-boosting late-delivery model as a FastAPI service with Pydantic-validated
  single and batch endpoints, containerised with Docker (slim, non-root, health check); measured
  8.2 ms median / 14.3 ms p95 latency per request and ~29 ms per 1,000-order batch locally.
- Built the MLOps workflow: MLflow tracking of parameters, metrics and model artifacts for 4 models,
  and GitHub Actions CI (ruff + 25 pytest tests, including API contract and train/serve consistency
  checks) that runs on synthetic data without the dataset.
- Re-implemented the SQL feature pipeline in PySpark (aggregations + window functions) and verified
  exact parity with SQL on all 62,897 orders, enforced by a CI test.

## 11. Interview questions — answer out loud, then check the hint
1. **What problem does MLflow solve that `metrics.json` didn't?**
   *Hint:* history and reproducibility. Every run's params, metrics and model are kept and comparable.
2. **Why load the model at startup instead of inside `/predict`?**
   *Hint:* loading takes time and memory. Do it once; per-request work should be just inference.
3. **What is train/serve skew, and how did you prevent it?**
   *Hint:* features computed differently in training vs serving. We used one shared function plus a
   test that the API score equals the pipeline score.
4. **What does a 422 mean, and where does it come from here?**
   *Hint:* "Unprocessable Entity". Pydantic rejected the body (wrong type, value or range) before
   our code ran.
5. **Why is batch prediction so much faster per order?**
   *Hint:* one HTTP round-trip and one vectorised `predict_proba` call instead of 1,000 of each.
6. **Why pin exact versions in `requirements-serve.txt` but floors in `requirements.txt`?**
   *Hint:* the container must load a pickled model identically (exact pins). The dev environment
   just needs compatible versions.
7. **Why run the container as a non-root user? What broke when you did?**
   *Hint:* least privilege. It exposed an owner-only file that root would have silently read.
8. **Why does `0.0.0.0` appear in the Dockerfile but `127.0.0.1` in `docker run -p`?**
   *Hint:* inside, listen on all interfaces so traffic from outside the container arrives; on the
   host, publish only to localhost.
9. **How does CI test the API without the 96 MB dataset or a trained model?**
   *Hint:* a fixture trains a tiny model on synthetic orders and writes its own `metrics.json`.
10. **How do you know the Spark and SQL features are the same? Why compare floats with a tolerance?**
    *Hint:* a full-data row-by-row comparison, plus a CI test with a negative case. Float addition
    isn't associative, so summation order changes the last digits.
11. **What does the window function in the Spark job do?**
    *Hint:* ranks items within each order by sales (desc), department (asc); keeping rank 1 gives
    the most expensive item's department.
12. **When would you actually need Spark here instead of SQLite?**
    *Hint:* when the data no longer fits or is too slow on one machine, e.g. billions of rows
    across many files. At 180k rows, SQLite is the right tool, and saying so is a good answer.

## 12. Your to-do (≈30 min)
1. `make all`, then `make mlflow-ui` and click through the runs.
2. `make serve`, and try valid and invalid orders at `/docs`.
3. `make docker-build && make docker-run`, then repeat one request against the container.
4. Open the Actions tab on GitHub and look at a passing run's logs.
5. Answer the 12 questions out loud; bring the shaky ones to me.
6. Update your resume's Week 1 bullet from "20x" to "~4x faster training".
