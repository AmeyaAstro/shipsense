# Week 1 walkthrough — read this before you put ShipSense on a resume

Budget about 60–90 minutes. Open each file in VS Code as you read.

## 1. The problem in one sentence
"When an order is placed, predict whether it will arrive later than its scheduled delivery
date, so operations can intervene (upgrade shipping, warn the customer)."

## 2. The data — `notebooks/01_eda.ipynb`
- 180,519 rows, but each row is an **order item**. There are 65,752 orders.
- Label `Late_delivery_risk` = 1 when actual shipping days > scheduled days.
- **Leakage trap:** `Days for shipping (real)` and `Delivery Status` are only known *after*
  delivery and basically equal the label. Using them gives ~98% accuracy and a useless model.
- Canceled / suspected-fraud orders never ship, so they are always 0 → removed.
- After cleaning: **62,897 shipped orders**, 57% late.
- Strongest signal: shipping mode. First Class is scheduled for 1 day but always takes 2, so it
  is always late.

## 3. The SQL layer — `sql/order_features.sql`, `src/shipsense/data.py`
- `data.py` loads the CSV into SQLite as `order_items`.
- The SQL `GROUP BY order_id` collapses items into one row per order: counts, sums, averages, and
  a correlated subquery for the department of the most expensive item.
- Why SQL instead of pandas? Real companies keep data in warehouses (Postgres, Snowflake,
  BigQuery); feature logic in SQL moves there unchanged.

## 4. The split — `time_split()` in `data.py`
- Train: 2015–2016 · Validation: Jan–Jun 2017 · Test: Jul 2017–Jan 2018.
- Time split = "train on the past, predict the future", which is how the model is used in production.

## 5. The models
| File | Model | What to know |
|---|---|---|
| `baselines.py` | Shipping-mode rule | Predicts the historical late rate for each mode. The bar any ML must clear. |
| `baselines.py` | Logistic regression | One-hot categoricals + scaled numerics. Linear, interpretable. |
| `baselines.py` | HistGradientBoosting | Tree ensemble with native categorical support. Usually best on tabular data. |
| `torch_model.py` | PyTorch MLP | Entity embeddings per categorical → 2 hidden layers (BatchNorm, ReLU, Dropout) → logit. AdamW, BCEWithLogitsLoss, early stopping on validation loss. |

PyTorch training loop (`TorchClassifier.fit`), line by line:
1. `TabularEncoder.fit` builds category→id maps **on train only** (id 0 = unseen).
2. `DataLoader` shuffles mini-batches of 512.
3. For each batch: `zero_grad` → forward → loss → `backward` (compute gradients) → `step` (update weights).
4. After each epoch, compute validation loss; keep the best weights; stop after 4 epochs with no improvement.

## 6. Evaluation — `evaluate.py`
- **ROC-AUC**: how well the model ranks late orders above on-time ones (threshold-free).
- **PR-AUC**: precision/recall trade-off for the "late" class.
- **Brier score**: how calibrated the probabilities are (lower is better).
- **Threshold** chosen on validation with Youden's J, never on test. We did not maximise F1:
  with 57% positives, F1 is maximised by flagging nearly everything.

## 7. Results and the honest conclusion
GBM 0.782 AUC ≈ PyTorch 0.780 > logistic 0.756 > rule 0.746. The deep model ties GBM but trains
~20x slower, so GBM is the production pick. That is a *good* interview answer: you tested the
fancy model, measured it, and chose the simpler one on evidence.

## 8. Resume bullets this week unlocks (truthful)
- Built a leakage-safe late-delivery risk model on 180K supply-chain records (DataCo); engineered
  order-level features in SQL and used a time-based split, improving ROC-AUC from 0.746 (rule
  baseline) to 0.782 with gradient boosting.
- Implemented a PyTorch tabular network (entity embeddings, BatchNorm, dropout, early stopping) and
  benchmarked it against scikit-learn models; selected gradient boosting for production on equal
  accuracy and 20x faster training.
- Identified and removed label-leaking post-delivery fields that inflate accuracy to ~98%;
  covered the pipeline with 11 pytest unit tests.

## 9. Interview questions — answer these out loud
1. What is data leakage? Give two examples from this dataset.
2. Why model at the order level instead of the item level?
3. Why a time-based split instead of `train_test_split`?
4. Why does your PyTorch model use embeddings instead of one-hot encoding?
5. Explain `loss.backward()` and `optimizer.step()` in one sentence each.
6. What does early stopping protect against?
7. Your deep model didn't beat gradient boosting. Why might that be, and what would you try next?
8. Why choose the threshold on validation instead of test?
9. ROC-AUC vs PR-AUC: when do you prefer each?
10. Why is 0.78 AUC a reasonable result here and not a failure?

## 10. Your to-do (≈20 min)
1. Open the folder in VS Code.
2. In the VS Code terminal: run the commands in README "Run it" and confirm `pytest` shows 11 passed.
3. Push to GitHub (commands in `docs/push_to_github.md`).
4. Read this doc and answer the 10 questions; bring the ones you're unsure about to me.
