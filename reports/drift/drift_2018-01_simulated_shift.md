# Drift report: 2018-01_simulated_shift

**SYNTHETIC SCENARIO:** 40% of this month's Standard Class orders were relabelled as First Class to show what the monitor does when the mix shifts. Not real data.

Reference: training orders (2015-2016, n=39,950) · current batch n=2,037

**Overall: ACT**  (from prediction PSI: < 0.1 stable · 0.1-0.25 watch · > 0.25 act; forced to act if batch AUC < 0.732). Feature PSI below explains why.

## Predictions

| Prediction PSI | Mean score (train) | Mean score (batch) |
|---|---|---|
| 0.3719 | 0.5736 | 0.7216 |

## Outcomes (labels available)

| Late rate (train) | Late rate (batch) | ROC-AUC (batch) |
|---|---|---|
| 0.5739 | 0.5866 | 0.6366 |

Model test-set ROC-AUC for comparison: 0.782.

## Features (highest PSI first)

| Feature | PSI | Status |
|---|---|---|
| order_month | 8.6227 | calendar (not alerted) |
| n_items | 7.6518 | act |
| main_department | 7.2101 | act |
| total_quantity | 7.1087 | act |
| n_departments | 6.8512 | act |
| market | 6.3726 | act |
| order_region | 5.9268 | act |
| order_country | 5.6593 | act |
| total_sales | 5.079 | act |
| avg_discount_rate | 0.9777 | act |
| shipping_mode | 0.3551 | act |
| days_scheduled | 0.3022 | act |
| order_dow | 0.0142 | stable |
| customer_segment | 0.0049 | stable |
| order_hour | 0.0012 | stable |
| payment_type | 0.0005 | stable |
