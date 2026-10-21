# Fifteen-model prediction research

Prediction evaluation only: no model portfolio-return backtest or calibrated-return prediction metrics.

- Historical window: 2015–2024, rolling out-of-sample predictions, already viewed during research.
- New window: 2025-01-02 through 2026-10-15, 430 valid sessions. Predictions extend to 2026-10-19; final labels are not yet complete.
- Each quarter selects exactly 100 factors using only prior data. All 15 models share feature selection, training/validation dates and evaluation labels. Computational budgets are not identical across models.
- R² is a decimal: 0.0027 is approximately 0.27%, not 27%. Equal-weight scores have a different scale from the rank target; native R² alone is not a fair model-ranking criterion.

![RankIC comparison](figures/rankic_comparison.png)

## Historical results

| universe   | model          |       IC |   RankIC |   rank_R2 |
|:-----------|:---------------|---------:|---------:|----------:|
| csi300     | catboost       | 0.037847 | 0.049156 |  0.002707 |
| csi300     | elastic_net    | 0.037145 | 0.048812 |  0.002263 |
| csi300     | lasso          | 0.036518 | 0.048705 |  0.002233 |
| csi300     | pca_ridge      | 0.035832 | 0.047965 |  0.001912 |
| csi300     | xgboost        | 0.03767  | 0.047661 |  0.001514 |
| csi300     | random_forest  | 0.036304 | 0.047541 |  0.002299 |
| csi300     | ft_transformer | 0.035414 | 0.047488 | -0.004464 |
| csi300     | ridge          | 0.03656  | 0.047106 |  0.000372 |
| csi300     | pls            | 0.035451 | 0.047084 |  0.000671 |
| csi300     | lightgbm       | 0.035792 | 0.045676 |  0.001422 |
| csi300     | moe            | 0.031806 | 0.038433 | -0.022892 |
| csi300     | mlp            | 0.030432 | 0.036009 | -0.020502 |
| csi300     | equal_weight   | 0.02627  | 0.035058 | -0.598332 |
| csi300     | resnet         | 0.02925  | 0.035035 | -0.019663 |
| csi300     | tabnet         | 0.019818 | 0.028268 |  0.000198 |
| csi500     | catboost       | 0.048745 | 0.06728  |  0.005527 |
| csi500     | random_forest  | 0.044341 | 0.065047 |  0.004998 |
| csi500     | xgboost        | 0.046695 | 0.064566 |  0.00459  |
| csi500     | lightgbm       | 0.045968 | 0.063811 |  0.004625 |
| csi500     | elastic_net    | 0.043362 | 0.06228  |  0.004134 |
| csi500     | ft_transformer | 0.043764 | 0.062081 |  0.000469 |
| csi500     | lasso          | 0.041889 | 0.061753 |  0.00422  |
| csi500     | pca_ridge      | 0.040648 | 0.059867 |  0.003623 |
| csi500     | ridge          | 0.042914 | 0.059783 |  0.002829 |
| csi500     | pls            | 0.042731 | 0.059697 |  0.002863 |
| csi500     | mlp            | 0.044287 | 0.053387 | -0.010209 |
| csi500     | resnet         | 0.041941 | 0.051894 | -0.008401 |
| csi500     | moe            | 0.043994 | 0.051591 | -0.013498 |
| csi500     | equal_weight   | 0.028959 | 0.047153 | -0.624821 |
| csi500     | tabnet         | 0.025123 | 0.044633 |  0.001514 |

## New-period results

| universe   | model          |        IC |   RankIC |   rank_R2 |
|:-----------|:---------------|----------:|---------:|----------:|
| csi300     | catboost       | -0.003874 | 0.028923 | -0.000648 |
| csi300     | random_forest  | -0.004233 | 0.028267 | -0.00069  |
| csi300     | ft_transformer |  0.00039  | 0.027295 | -0.008088 |
| csi300     | lightgbm       | -0.00482  | 0.025617 | -0.001306 |
| csi300     | xgboost        | -0.005737 | 0.023646 | -0.002937 |
| csi300     | equal_weight   | -0.010832 | 0.021898 | -0.659483 |
| csi300     | pca_ridge      | -0.005515 | 0.021838 | -0.000971 |
| csi300     | elastic_net    | -0.008856 | 0.021712 | -0.000271 |
| csi300     | pls            | -0.005785 | 0.020413 | -0.002973 |
| csi300     | ridge          | -0.005527 | 0.020004 | -0.003334 |
| csi300     | lasso          | -0.010479 | 0.019496 | -6.3e-05  |
| csi300     | tabnet         |  8.9e-05  | 0.018483 | -0.00092  |
| csi300     | resnet         | -0.00579  | 0.01442  | -0.021169 |
| csi300     | mlp            | -0.000362 | 0.010782 | -0.024272 |
| csi300     | moe            | -0.008042 | 0.006623 | -0.032351 |
| csi500     | random_forest  |  0.006113 | 0.043524 |  0.001821 |
| csi500     | catboost       |  0.003553 | 0.042071 |  0.001506 |
| csi500     | xgboost        |  0.006301 | 0.039961 |  0.000828 |
| csi500     | ft_transformer |  0.005573 | 0.039955 | -0.001695 |
| csi500     | equal_weight   |  0.00166  | 0.039922 | -0.690152 |
| csi500     | lightgbm       |  0.003248 | 0.038181 |  0.000644 |
| csi500     | pca_ridge      |  0.003717 | 0.03747  |  0.001076 |
| csi500     | elastic_net    | -0.001602 | 0.034958 |  0.000883 |
| csi500     | lasso          | -0.004694 | 0.032066 |  0.000978 |
| csi500     | pls            |  0.003874 | 0.031409 | -0.000603 |
| csi500     | ridge          |  0.003178 | 0.030488 | -0.001421 |
| csi500     | tabnet         |  0.003239 | 0.029724 | -0.000298 |
| csi500     | resnet         | -0.001723 | 0.021261 | -0.020828 |
| csi500     | mlp            | -0.001593 | 0.01583  | -0.020776 |
| csi500     | moe            | -0.002927 | 0.013205 | -0.030079 |

CatBoost leads historical mean IC and RankIC in both universes. In the new period, CSI300 CatBoost has negative mean IC, and every model has negative native rank R². CSI500 Random Forest leads RankIC and rank R², but its RankIC improvement over equal weight is small. These point estimates do not establish statistical superiority or executable profits. New-period deterioration is retained in full.

Annual/period IC summaries and daily aggregate IC series are in `results`; these are not stock-level predictions. Native rank R² is provided as a full-period summary only; calibrated-return R² is excluded.
