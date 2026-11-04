# Drift summary: every month after training vs training (2015-2016)

Real orders. Overall = prediction PSI (< 0.10 stable · 0.10-0.25 watch · > 0.25 act), forced to act if AUC < 0.732. AUC uses each month's actual labels (in production they arrive days later). Model test-set AUC: 0.782.

| Month | Orders | Overall | Top drifting feature (PSI) | shipping_mode PSI | Prediction PSI | Late rate | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 2017-01 | 1,685 | stable | order_country (4.5384) | 0.0053 | 0.0708 | 0.5579 | 0.7556 |
| 2017-02 | 1,538 | stable | market (7.3503) | 0.0015 | 0.0082 | 0.5715 | 0.7782 |
| 2017-03 | 1,710 | stable | market (7.3503) | 0.0011 | 0.0098 | 0.5754 | 0.7768 |
| 2017-04 | 1,666 | stable | market (7.3503) | 0.0027 | 0.0121 | 0.5504 | 0.7569 |
| 2017-05 | 1,672 | stable | market (7.3503) | 0.0026 | 0.0106 | 0.5772 | 0.7815 |
| 2017-06 | 1,605 | stable | market (4.8067) | 0.0024 | 0.0152 | 0.5676 | 0.7542 |
| 2017-07 | 1,693 | stable | market (7.0091) | 0.0038 | 0.0063 | 0.5706 | 0.7813 |
| 2017-08 | 1,695 | stable | market (7.0091) | 0.0009 | 0.0158 | 0.5788 | 0.7864 |
| 2017-09 | 1,640 | stable | market (7.0091) | 0.0008 | 0.0083 | 0.5738 | 0.7702 |
| 2017-10 | 2,019 | watch | main_department (8.728) | 0.0023 | 0.2357 | 0.5632 | 0.7923 |
| 2017-11 | 1,956 | stable | main_department (12.3057) | 0.0005 | 0.0713 | 0.5787 | 0.7773 |
| 2017-12 | 2,031 | stable | n_items (7.6518) | 0.0021 | 0.0268 | 0.5771 | 0.7804 |
| 2018-01 | 2,037 | stable | n_items (7.6518) | 0.0017 | 0.0625 | 0.5866 | 0.7828 |
