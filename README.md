# ShipSense — late-delivery risk + logistics operations copilot

Predict, **at the moment an order is placed**, whether it will arrive late, then (in later
milestones) serve the model as an API and wrap it in an LLM agent that answers operations
questions using the model plus company shipping policies.

> Status: **Week 1 of 4 complete** — data pipeline, leakage-safe features in SQL,
> three baselines and a PyTorch model compared on a time-based split.

## Results (test set: Jul 2017 – Jan 2018, 13,071 orders, 57% late)

| Model | ROC-AUC | PR-AUC | Balanced acc. | Accuracy | Brier ↓ | Train time |
|---|---|---|---|---|---|---|
| Shipping-mode rule (no ML) | 0.746 | 0.788 | 0.724 | 0.696 | 0.188 | 0 s |
| Logistic regression | 0.756 | 0.844 | 0.735 | 0.711 | 0.187 | 0.4 s |
| **Gradient boosting** | **0.782** | **0.862** | **0.750** | **0.726** | **0.175** | 0.7 s |
| PyTorch MLP (entity embeddings) | 0.780 | 0.860 | 0.749 | 0.725 | 0.176 | 14 s |

**Honest read:** gradient boosting and the PyTorch network are tied (within 0.003 AUC), and both
beat the no-ML rule by ~3.5 AUC points. The deep model gives no accuracy gain on this small tabular
problem and trains ~20x slower, so **gradient boosting is the production choice**; the PyTorch model
stays as the benchmark and as the base for adding text/sequence inputs later.

Why not 99%? The EDA shows actual shipping days are spread almost evenly within each shipping mode,
so most of the remaining lateness is not predictable from order-time data. Versions of this project
that report ~98% accuracy are using `Days for shipping (real)` or `Delivery Status`, which **are** the
label (see `notebooks/01_eda.ipynb`).

## Key design decisions

1. **Order-level, not item-level.** The CSV has 180k item rows but only 65k orders, and all items in
   an order share the label. Item-level random splits put the same order in train and test.
2. **No leakage.** Only fields known at order time. Real shipping days, delivery status, shipping date,
   order status and profit fields are excluded (a unit test enforces this).
3. **Canceled / suspected-fraud orders removed:** they never ship, so they are always labelled "not late".
4. **Time-based split:** train 2015–2016, validate Jan–Jun 2017, test Jul 2017–Jan 2018.
5. **Threshold tuned on validation only** (Youden's J), then applied unchanged to test.

## Project layout

```
sql/order_features.sql      feature engineering in SQL (SQLite now, Postgres-compatible)
src/shipsense/config.py     paths, split dates, feature lists
src/shipsense/data.py       CSV -> SQLite -> order-level features, time split
src/shipsense/baselines.py  shipping-mode rule, logistic regression, gradient boosting
src/shipsense/torch_model.py  PyTorch embeddings + MLP with early stopping
src/shipsense/evaluate.py   shared metrics + threshold selection
src/shipsense/train.py      trains everything, writes reports/
notebooks/01_eda.ipynb      EDA and leakage analysis
tests/                      11 pytest unit tests on synthetic data
reports/                    metrics.json, model_comparison.md
```

## Run it

```bash
# 1. put DataCoSupplyChainDataset.csv in archive/ or data/raw/  (Kaggle download)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m shipsense.data     # builds data/processed/shipsense.db
python -m shipsense.train    # trains 4 models, writes reports/
pytest -q
```

## Roadmap

- [x] **Week 1** — EDA, SQL features, baselines, PyTorch model, honest comparison
- [ ] **Week 2** — FastAPI service, Docker, GitHub Actions CI, MLflow tracking, PySpark feature job
- [ ] **Week 3** — RAG over shipping SOPs + LLM agent with tools (`predict_risk`, `lookup_shipment`, `search_policy`), eval set
- [ ] **Week 4** — AWS deployment, drift monitoring, demo video

Data: [DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (Kaggle). Not redistributed in this repo.
