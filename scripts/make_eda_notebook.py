"""Generates notebooks/01_eda.ipynb (so the notebook stays reproducible)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# 01 - Exploratory data analysis: DataCo late deliveries\n\n"
       "**Question:** at the moment an order is placed, can we predict whether it will arrive late?\n\n"
       "This notebook checks (1) what the label really means, (2) which columns would *leak* the "
       "answer, and (3) which order-time signals look predictive."),
    code("import pandas as pd\nimport matplotlib.pyplot as plt\n"
         "from shipsense.config import RAW_CSV\n"
         "raw = pd.read_csv(RAW_CSV, encoding='latin1')\nprint(raw.shape)\nraw.head(3).T.head(20)"),
    md("## 1. One row = one order *item*, not one order"),
    code("print('rows:', len(raw), '| unique orders:', raw['Order Id'].nunique())\n"
         "print('orders whose items disagree on the label:',\n"
         "      (raw.groupby('Order Id')['Late_delivery_risk'].nunique() > 1).sum())"),
    md("Every item in an order shares the label. If we split rows randomly, items from the *same* order "
       "would sit in both train and test, inflating scores. **Decision: model at order level.**"),
    md("## 2. What defines the label? (leakage check)"),
    code("late_by_rule = (raw['Days for shipping (real)'] > raw['Days for shipment (scheduled)']).astype(int)\n"
         "print('label == (real days > scheduled days):', "
         "round((late_by_rule == raw['Late_delivery_risk']).mean(), 4))\n"
         "pd.crosstab(raw['Delivery Status'], raw['Late_delivery_risk'])"),
    md("`Days for shipping (real)` and `Delivery Status` basically *are* the label, and both are only known "
       "after delivery. Including them would give ~98-100% accuracy and a useless model. "
       "Shipping date and profit fields are also post-fulfilment. **All are excluded.**"),
    code("pd.crosstab(raw['Order Status'], raw['Late_delivery_risk'])"),
    md("Canceled and suspected-fraud orders are *always* 0 because they never ship. "
       "**Decision: drop them**; the model answers 'given it ships, will it be late?'."),
    md("## 3. Order-time signals"),
    code("fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
         "for ax, col in zip(axes, ['Shipping Mode', 'Type', 'Market']):\n"
         "    raw.groupby(col)['Late_delivery_risk'].mean().sort_values().plot.barh(ax=ax, color='#4C72B0')\n"
         "    ax.set_title(f'Late rate by {col}'); ax.set_xlim(0, 1); ax.set_xlabel('share late')\n"
         "plt.tight_layout()"),
    md("**Shipping mode dominates:** First Class is late ~95% of the time (scheduled 1 day), "
       "Standard Class ~38% (scheduled 4 days). Payment type matters a little (TRANSFER is lower). "
       "Market barely matters (all ~55%)."),
    code("pd.crosstab(raw['Shipping Mode'], raw['Days for shipping (real)'], normalize='index').round(2)"),
    md("Striking pattern: First Class *always* takes 2 days (scheduled 1, so it is always late), and "
       "Second/Standard Class are spread evenly over 2-6 days. Within a mode, actual days look close to "
       "random, which caps how well *any* model can do. A realistic goal is beating the shipping-mode rule, "
       "not 99% accuracy. (This is a known property of the DataCo dataset - worth saying in interviews.)"),
    code("ts = pd.to_datetime(raw['order date (DateOrders)'])\n"
         "raw.groupby(ts.dt.to_period('M'))['Late_delivery_risk'].mean()"
         ".plot(figsize=(12, 3), title='Monthly late rate')\n"
         "plt.ylim(0.4, 0.7); plt.ylabel('share late')"),
    md("The late rate is stable over time (~54-57%), so a **time-based split** "
       "(train 2015-16, validate H1 2017, test Jul 2017-Jan 2018) is fair and mirrors production."),
    md("## 4. Feature table built in SQL\nSee `sql/order_features.sql`. Load it:"),
    code("from shipsense.data import build_database, load_raw, load_order_features\n"
         "build_database(load_raw())\nfeats = load_order_features()\nprint(feats.shape)\nfeats.head()"),
    md("## Takeaways\n"
       "1. Model at the order level (62,897 shipped orders).\n"
       "2. Exclude post-delivery columns (they leak the label).\n"
       "3. Shipping mode is the strongest signal; any model must beat a shipping-mode-only rule.\n"
       "4. Use a time-based split.\n\nNext: `python -m shipsense.train` compares the models."),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
