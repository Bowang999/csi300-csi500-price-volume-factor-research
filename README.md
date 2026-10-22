# CSI300 & CSI500 Price–Volume Factor Research

Research on dynamic CSI300 and CSI500 constituents: price–volume factor construction, single-factor backtests, 50-model signal combination, and new-period out-of-sample prediction evaluation.

**857 candidate factor fields · 1706 factor–universe backtests · 50 configurations · 2 universes.** Window variants are not independent signals. This is auditable historical research, not a promise of live returns.

## Representative single-factor IC and RankIC

Five examples cover momentum (`PV_MOM_20`), volatility (`PV_VOL_20`), turnover variability (`PV_TURN_STD_30`), illiquidity (`PV_AMIHUD_20`), and a report-formula implementation (`HT101_001`). They illustrate different formula families and were **not selected as the highest-IC factors**.

**Window: 2015-01-05–2024-12-27.** Values are arithmetic means of daily cross-sectional Pearson IC and Spearman RankIC against one-session forward VWAP returns, after library preprocessing (MAD, industry/size neutralization, standardization). The **original factor sign is retained**, including negative correlations; no full-period sign flip is applied. Each row has 2429 valid days except CSI500 `HT101_001` (2428).

| Factor | CSI300 IC | CSI300 RankIC | CSI500 IC | CSI500 RankIC |
|---|---:|---:|---:|---:|
| PV_MOM_20 | -0.01032 | -0.02009 | -0.01372 | -0.02730 |
| PV_VOL_20 | -0.00418 | -0.02277 | -0.00935 | -0.03364 |
| PV_TURN_STD_30 | -0.01155 | -0.01933 | -0.01706 | -0.03354 |
| PV_AMIHUD_20 | 0.00329 | 0.00683 | 0.00330 | 0.01100 |
| HT101_001 | -0.00947 | -0.01814 | -0.00981 | -0.01940 |

[Summary and coverage](backtests/results/representative_single_factor_ic.csv) · [Daily aggregate IC/RankIC](backtests/results/representative_single_factor_daily_ic.csv) · [Formulas](factors/catalog.json)

A negative IC means higher factor values were associated with lower subsequent returns. It does not establish that reversing the factor will generate profitable trades after costs.

## Fifty-model composite IC and RankIC

All 50 configurations have completed both universes and both evaluation windows. Every quarter uses exactly 100 historically selected factors and training-only directions, with inner-validation tuning. Models learn a cross-sectional rank target; IC and RankIC evaluate scores against raw forward one-session VWAP returns. Scores are not return percentages.

The four tables below show **all 50 configurations**, ranked separately by mean RankIC within each market and evaluation window. Negative metrics are retained. Both periods have been viewed during research, so these are extended comparisons rather than fresh blind holdouts. Related loss-function and architecture variants are included; these are not fifty independent algorithm families.

**Reading the tables:** IC is mean daily Pearson correlation; RankIC is mean daily Spearman correlation. Native rank R² uses the cross-sectional rank target and a zero-score benchmark, not raw-return prediction or IC squared. R² values are decimals (0.002 = 0.2%). Valid days counts days where both IC and RankIC are defined for the added suites.

### CSI300 — 2015–2024 historical rolling evaluation

| Rank | Model | IC | RankIC | Native rank R² | Valid days |
|---:|---|---:|---:|---:|---:|
| 1 | extra_trees | 0.03679 | 0.05039 | 0.002911 | 2429 |
| 2 | bayesian_ridge | 0.03800 | 0.05010 | 0.002510 | 2429 |
| 3 | catboost | 0.03785 | 0.04916 | 0.002707 | 2429 |
| 4 | spline_additive_ridge | 0.03771 | 0.04904 | 0.000046 | 2429 |
| 5 | rbf_random_features | 0.03706 | 0.04902 | 0.002216 | 2429 |
| 6 | polynomial_sketch_ridge | 0.03886 | 0.04892 | 0.001443 | 2429 |
| 7 | elastic_net | 0.03715 | 0.04881 | 0.002263 | 2429 |
| 8 | lasso | 0.03652 | 0.04871 | 0.002233 | 2429 |
| 9 | laplacian_nystroem | 0.03437 | 0.04852 | 0.002325 | 2429 |
| 10 | pca_ridge | 0.03583 | 0.04797 | 0.001912 | 2429 |
| 11 | xgboost | 0.03767 | 0.04766 | 0.001514 | 2429 |
| 12 | random_forest | 0.03630 | 0.04754 | 0.002299 | 2429 |
| 13 | ft_transformer | 0.03541 | 0.04749 | -0.004464 | 2429 |
| 14 | ridge | 0.03656 | 0.04711 | 0.000372 | 2429 |
| 15 | pls | 0.03545 | 0.04708 | 0.000671 | 2429 |
| 16 | catboost_mae | 0.03537 | 0.04689 | -0.002722 | 2429 |
| 17 | hist_gradient_boosting | 0.03653 | 0.04638 | 0.001447 | 2429 |
| 18 | huber_gradient_boosting | 0.03507 | 0.04612 | 0.002205 | 2429 |
| 19 | huber_regression | 0.03652 | 0.04595 | -0.008048 | 2429 |
| 20 | ordinary_least_squares | 0.03601 | 0.04579 | -0.001024 | 2429 |
| 21 | linear_svr | 0.03599 | 0.04576 | -0.000447 | 2429 |
| 22 | lightgbm | 0.03579 | 0.04568 | 0.001422 | 2429 |
| 23 | bagging_trees | 0.03513 | 0.04557 | 0.002070 | 2429 |
| 24 | boosted_stumps | 0.03294 | 0.04524 | 0.002537 | 2429 |
| 25 | median_hist_boosting | 0.03536 | 0.04524 | -0.005358 | 2429 |
| 26 | random_projection_ridge | 0.03374 | 0.04501 | 0.001087 | 2429 |
| 27 | passive_aggressive | 0.03460 | 0.04496 | -0.007347 | 2429 |
| 28 | lightgbm_dart | 0.03429 | 0.04484 | 0.002407 | 2429 |
| 29 | orthogonal_matching_pursuit | 0.03352 | 0.04438 | 0.000960 | 2429 |
| 30 | extreme_learning_machine | 0.03207 | 0.04375 | -0.001030 | 2429 |
| 31 | adaboost | 0.03003 | 0.04348 | 0.002216 | 2429 |
| 32 | least_angle_regression | 0.03172 | 0.04287 | 0.001386 | 2429 |
| 33 | ard_regression | 0.03269 | 0.04174 | 0.001072 | 2373 |
| 34 | soft_threshold_network | 0.03207 | 0.04166 | -0.005536 | 2429 |
| 35 | gated_mlp | 0.03489 | 0.03996 | -0.014358 | 2429 |
| 36 | neural_additive | 0.03257 | 0.03978 | -0.007284 | 2429 |
| 37 | highway_network | 0.03306 | 0.03897 | -0.021855 | 2429 |
| 38 | moe | 0.03181 | 0.03843 | -0.022892 | 2429 |
| 39 | deep_cross_network | 0.03159 | 0.03822 | -0.019736 | 2429 |
| 40 | supervised_autoencoder | 0.03084 | 0.03766 | -0.015862 | 2429 |
| 41 | mlp | 0.03043 | 0.03601 | -0.020502 | 2429 |
| 42 | low_rank_bilinear | 0.03095 | 0.03555 | -0.025917 | 2429 |
| 43 | factorization_machine | 0.02887 | 0.03534 | -0.026693 | 2429 |
| 44 | equal_weight | 0.02627 | 0.03506 | -0.598332 | 2429 |
| 45 | resnet | 0.02925 | 0.03504 | -0.019663 | 2429 |
| 46 | pca_knn | 0.02617 | 0.03495 | -0.002048 | 2429 |
| 47 | cross_network | 0.02883 | 0.03470 | -0.004406 | 2429 |
| 48 | deep_fm_numeric | 0.02671 | 0.03050 | -0.036977 | 2429 |
| 49 | decision_tree | 0.02126 | 0.02876 | -0.002851 | 2429 |
| 50 | tabnet | 0.01982 | 0.02827 | 0.000198 | 2429 |

### CSI500 — 2015–2024 historical rolling evaluation

| Rank | Model | IC | RankIC | Native rank R² | Valid days |
|---:|---|---:|---:|---:|---:|
| 1 | catboost | 0.04875 | 0.06728 | 0.005527 | 2429 |
| 2 | extra_trees | 0.04311 | 0.06571 | 0.005189 | 2429 |
| 3 | catboost_mae | 0.04710 | 0.06536 | -0.000661 | 2429 |
| 4 | random_forest | 0.04434 | 0.06505 | 0.004998 | 2429 |
| 5 | laplacian_nystroem | 0.04017 | 0.06501 | 0.004537 | 2429 |
| 6 | bagging_trees | 0.04361 | 0.06477 | 0.004852 | 2429 |
| 7 | xgboost | 0.04669 | 0.06457 | 0.004590 | 2429 |
| 8 | lightgbm | 0.04597 | 0.06381 | 0.004625 | 2429 |
| 9 | hist_gradient_boosting | 0.04526 | 0.06355 | 0.004575 | 2429 |
| 10 | rbf_random_features | 0.04452 | 0.06346 | 0.004428 | 2429 |
| 11 | huber_gradient_boosting | 0.04304 | 0.06324 | 0.004467 | 2429 |
| 12 | lightgbm_dart | 0.04339 | 0.06309 | 0.004909 | 2429 |
| 13 | spline_additive_ridge | 0.04600 | 0.06305 | 0.003502 | 2429 |
| 14 | bayesian_ridge | 0.04399 | 0.06246 | 0.004121 | 2429 |
| 15 | elastic_net | 0.04336 | 0.06228 | 0.004134 | 2429 |
| 16 | median_hist_boosting | 0.04453 | 0.06223 | -0.003122 | 2429 |
| 17 | ft_transformer | 0.04376 | 0.06208 | 0.000469 | 2429 |
| 18 | lasso | 0.04189 | 0.06175 | 0.004220 | 2429 |
| 19 | boosted_stumps | 0.03768 | 0.06076 | 0.004288 | 2429 |
| 20 | polynomial_sketch_ridge | 0.04430 | 0.06063 | 0.003611 | 2429 |
| 21 | pca_ridge | 0.04065 | 0.05987 | 0.003623 | 2429 |
| 22 | ridge | 0.04291 | 0.05978 | 0.002829 | 2429 |
| 23 | pls | 0.04273 | 0.05970 | 0.002863 | 2429 |
| 24 | huber_regression | 0.04323 | 0.05937 | -0.004616 | 2429 |
| 25 | orthogonal_matching_pursuit | 0.04146 | 0.05932 | 0.003283 | 2429 |
| 26 | linear_svr | 0.04291 | 0.05913 | 0.002386 | 2429 |
| 27 | adaboost | 0.03470 | 0.05872 | 0.003676 | 2429 |
| 28 | ordinary_least_squares | 0.04290 | 0.05857 | 0.002123 | 2429 |
| 29 | random_projection_ridge | 0.04125 | 0.05832 | 0.003277 | 2429 |
| 30 | passive_aggressive | 0.04170 | 0.05811 | -0.003807 | 2429 |
| 31 | extreme_learning_machine | 0.03980 | 0.05807 | 0.001789 | 2429 |
| 32 | gated_mlp | 0.04650 | 0.05679 | -0.007905 | 2429 |
| 33 | highway_network | 0.04639 | 0.05670 | -0.011694 | 2429 |
| 34 | supervised_autoencoder | 0.04657 | 0.05616 | -0.006303 | 2429 |
| 35 | ard_regression | 0.03845 | 0.05511 | 0.002591 | 2429 |
| 36 | deep_cross_network | 0.04370 | 0.05463 | -0.011763 | 2429 |
| 37 | mlp | 0.04429 | 0.05339 | -0.010209 | 2429 |
| 38 | least_angle_regression | 0.03349 | 0.05245 | 0.001826 | 2429 |
| 39 | resnet | 0.04194 | 0.05189 | -0.008401 | 2429 |
| 40 | moe | 0.04399 | 0.05159 | -0.013498 | 2429 |
| 41 | soft_threshold_network | 0.03891 | 0.05142 | -0.002787 | 2429 |
| 42 | pca_knn | 0.03431 | 0.05105 | 0.000431 | 2429 |
| 43 | low_rank_bilinear | 0.04112 | 0.05066 | -0.015182 | 2429 |
| 44 | neural_additive | 0.03799 | 0.05027 | -0.004682 | 2429 |
| 45 | factorization_machine | 0.03939 | 0.04846 | -0.015847 | 2429 |
| 46 | equal_weight | 0.02896 | 0.04715 | -0.624821 | 2429 |
| 47 | decision_tree | 0.02779 | 0.04510 | -0.000620 | 2429 |
| 48 | tabnet | 0.02512 | 0.04463 | 0.001514 | 2429 |
| 49 | deep_fm_numeric | 0.03741 | 0.04418 | -0.023064 | 2429 |
| 50 | cross_network | 0.03448 | 0.04411 | -0.001406 | 2429 |

### CSI300 — 2025–2026 new-period evaluation

| Rank | Model | IC | RankIC | Native rank R² | Valid days |
|---:|---|---:|---:|---:|---:|
| 1 | extra_trees | -0.00418 | 0.02983 | 0.000415 | 430 |
| 2 | bagging_trees | -0.00014 | 0.02963 | -0.000091 | 430 |
| 3 | least_angle_regression | -0.00206 | 0.02938 | 0.000812 | 430 |
| 4 | catboost | -0.00387 | 0.02892 | -0.000648 | 430 |
| 5 | laplacian_nystroem | -0.00567 | 0.02829 | 0.000037 | 430 |
| 6 | random_forest | -0.00423 | 0.02827 | -0.000690 | 430 |
| 7 | median_hist_boosting | -0.00517 | 0.02809 | -0.006994 | 430 |
| 8 | catboost_mae | -0.00430 | 0.02808 | -0.006478 | 430 |
| 9 | adaboost | -0.00353 | 0.02801 | 0.000381 | 430 |
| 10 | huber_gradient_boosting | -0.00460 | 0.02792 | -0.000407 | 430 |
| 11 | ft_transformer | 0.00039 | 0.02729 | -0.008088 | 430 |
| 12 | boosted_stumps | -0.00692 | 0.02677 | 0.000540 | 430 |
| 13 | lightgbm_dart | -0.00504 | 0.02569 | 0.000113 | 430 |
| 14 | lightgbm | -0.00482 | 0.02562 | -0.001306 | 430 |
| 15 | bayesian_ridge | -0.00518 | 0.02476 | -0.000408 | 430 |
| 16 | spline_additive_ridge | -0.00179 | 0.02466 | -0.003357 | 430 |
| 17 | xgboost | -0.00574 | 0.02365 | -0.002937 | 430 |
| 18 | hist_gradient_boosting | -0.00548 | 0.02343 | -0.002303 | 430 |
| 19 | extreme_learning_machine | -0.00058 | 0.02295 | -0.003616 | 430 |
| 20 | random_projection_ridge | -0.00288 | 0.02281 | -0.001116 | 430 |
| 21 | equal_weight | -0.01083 | 0.02190 | -0.659483 | 430 |
| 22 | pca_ridge | -0.00552 | 0.02184 | -0.000971 | 430 |
| 23 | elastic_net | -0.00886 | 0.02171 | -0.000271 | 430 |
| 24 | rbf_random_features | -0.00968 | 0.02126 | -0.002806 | 430 |
| 25 | decision_tree | -0.00177 | 0.02053 | -0.005922 | 430 |
| 26 | polynomial_sketch_ridge | -0.00112 | 0.02043 | -0.002804 | 430 |
| 27 | pls | -0.00578 | 0.02041 | -0.002973 | 430 |
| 28 | huber_regression | -0.00363 | 0.02036 | -0.008694 | 430 |
| 29 | passive_aggressive | -0.00235 | 0.02027 | -0.009161 | 430 |
| 30 | linear_svr | -0.00423 | 0.02015 | -0.004001 | 430 |
| 31 | ridge | -0.00553 | 0.02000 | -0.003334 | 430 |
| 32 | low_rank_bilinear | 0.00004 | 0.01975 | -0.022487 | 430 |
| 33 | ordinary_least_squares | -0.00415 | 0.01961 | -0.004774 | 430 |
| 34 | lasso | -0.01048 | 0.01950 | -0.000063 | 430 |
| 35 | tabnet | 0.00009 | 0.01848 | -0.000920 | 430 |
| 36 | orthogonal_matching_pursuit | -0.00877 | 0.01843 | -0.002185 | 430 |
| 37 | ard_regression | -0.00613 | 0.01795 | -0.001457 | 370 |
| 38 | deep_fm_numeric | -0.00023 | 0.01779 | -0.035256 | 430 |
| 39 | highway_network | -0.00422 | 0.01719 | -0.026875 | 430 |
| 40 | pca_knn | -0.00264 | 0.01581 | -0.007246 | 430 |
| 41 | cross_network | -0.00244 | 0.01553 | -0.007925 | 430 |
| 42 | supervised_autoencoder | -0.00542 | 0.01511 | -0.018731 | 430 |
| 43 | factorization_machine | 0.00292 | 0.01505 | -0.021930 | 430 |
| 44 | soft_threshold_network | -0.00444 | 0.01482 | -0.009403 | 430 |
| 45 | deep_cross_network | 0.00292 | 0.01449 | -0.018315 | 430 |
| 46 | gated_mlp | -0.00404 | 0.01448 | -0.016251 | 430 |
| 47 | resnet | -0.00579 | 0.01442 | -0.021169 | 430 |
| 48 | mlp | -0.00036 | 0.01078 | -0.024272 | 430 |
| 49 | moe | -0.00804 | 0.00662 | -0.032351 | 430 |
| 50 | neural_additive | -0.01452 | 0.00618 | -0.009928 | 430 |

### CSI500 — 2025–2026 new-period evaluation

| Rank | Model | IC | RankIC | Native rank R² | Valid days |
|---:|---|---:|---:|---:|---:|
| 1 | extra_trees | 0.00560 | 0.04527 | 0.002072 | 430 |
| 2 | random_forest | 0.00611 | 0.04352 | 0.001821 | 430 |
| 3 | adaboost | 0.00371 | 0.04326 | 0.001865 | 430 |
| 4 | huber_gradient_boosting | 0.00571 | 0.04297 | 0.001400 | 430 |
| 5 | laplacian_nystroem | 0.00280 | 0.04278 | 0.001681 | 430 |
| 6 | catboost_mae | 0.00376 | 0.04252 | -0.004403 | 430 |
| 7 | lightgbm_dart | 0.00455 | 0.04228 | 0.001830 | 430 |
| 8 | bagging_trees | 0.00537 | 0.04223 | 0.001601 | 430 |
| 9 | catboost | 0.00355 | 0.04207 | 0.001506 | 430 |
| 10 | boosted_stumps | 0.00223 | 0.04136 | 0.001830 | 430 |
| 11 | median_hist_boosting | 0.00540 | 0.04029 | -0.005820 | 430 |
| 12 | hist_gradient_boosting | 0.00428 | 0.04019 | 0.001092 | 430 |
| 13 | xgboost | 0.00630 | 0.03996 | 0.000828 | 430 |
| 14 | ft_transformer | 0.00557 | 0.03996 | -0.001695 | 430 |
| 15 | equal_weight | 0.00166 | 0.03992 | -0.690152 | 430 |
| 16 | lightgbm | 0.00325 | 0.03818 | 0.000644 | 430 |
| 17 | pca_ridge | 0.00372 | 0.03747 | 0.001076 | 430 |
| 18 | rbf_random_features | 0.00411 | 0.03527 | 0.000323 | 430 |
| 19 | elastic_net | -0.00160 | 0.03496 | 0.000883 | 430 |
| 20 | bayesian_ridge | 0.00370 | 0.03422 | 0.000327 | 430 |
| 21 | soft_threshold_network | 0.00747 | 0.03418 | -0.002894 | 430 |
| 22 | least_angle_regression | -0.00143 | 0.03347 | 0.001090 | 430 |
| 23 | random_projection_ridge | 0.00334 | 0.03345 | -0.000061 | 430 |
| 24 | spline_additive_ridge | 0.00358 | 0.03329 | -0.001049 | 430 |
| 25 | orthogonal_matching_pursuit | 0.00316 | 0.03239 | -0.000197 | 430 |
| 26 | polynomial_sketch_ridge | 0.00408 | 0.03220 | -0.001108 | 430 |
| 27 | lasso | -0.00469 | 0.03207 | 0.000978 | 430 |
| 28 | pca_knn | 0.00255 | 0.03183 | -0.002872 | 430 |
| 29 | pls | 0.00387 | 0.03141 | -0.000603 | 430 |
| 30 | decision_tree | 0.00452 | 0.03098 | -0.000906 | 430 |
| 31 | ard_regression | -0.00124 | 0.03085 | -0.000170 | 430 |
| 32 | ridge | 0.00318 | 0.03049 | -0.001421 | 430 |
| 33 | extreme_learning_machine | 0.00202 | 0.03046 | -0.001641 | 430 |
| 34 | tabnet | 0.00324 | 0.02972 | -0.000298 | 430 |
| 35 | huber_regression | 0.00356 | 0.02972 | -0.008982 | 430 |
| 36 | ordinary_least_squares | 0.00285 | 0.02939 | -0.002332 | 430 |
| 37 | linear_svr | 0.00271 | 0.02917 | -0.001992 | 430 |
| 38 | passive_aggressive | 0.00160 | 0.02821 | -0.009321 | 430 |
| 39 | neural_additive | 0.00897 | 0.02146 | -0.008041 | 430 |
| 40 | resnet | -0.00172 | 0.02126 | -0.020828 | 430 |
| 41 | highway_network | 0.00013 | 0.02063 | -0.015855 | 430 |
| 42 | supervised_autoencoder | -0.00215 | 0.01822 | -0.016063 | 430 |
| 43 | factorization_machine | 0.00739 | 0.01813 | -0.016773 | 430 |
| 44 | low_rank_bilinear | 0.00446 | 0.01614 | -0.021324 | 430 |
| 45 | mlp | -0.00159 | 0.01583 | -0.020776 | 430 |
| 46 | gated_mlp | -0.00263 | 0.01498 | -0.015726 | 430 |
| 47 | deep_cross_network | -0.00383 | 0.01413 | -0.020100 | 430 |
| 48 | deep_fm_numeric | -0.00955 | 0.01351 | -0.040565 | 430 |
| 49 | moe | -0.00293 | 0.01320 | -0.030079 | 430 |
| 50 | cross_network | -0.00787 | 0.00489 | -0.006304 | 430 |

Extra Trees leads RankIC in three panels; CatBoost leads historical CSI500. Historical CSI300 Pearson IC is highest for Polynomial Sketch + Ridge, while new-period Pearson IC leaders differ from RankIC leaders. New-period CSI300 top-RankIC models still have negative Pearson IC. Rankings do not establish statistically significant improvement, executable profits, or general model superiority; no fifty-model multiple-testing correction has yet been applied.

Historical coverage is 2429 sessions (2015-01-05–2024-12-27), and new-period coverage is 430 (2025-01-02–2026-10-15), except CSI300 ARD with 2373 and 370 valid IC days. Undefined correlations remain missing. The representative single-factor table above retains original signs and is not a strongest-single-factor benchmark.

[Full 50-model tables and native rank R²](research/README.md) · [All 200 model/market/period results](research/results/all_50_models_comparison.csv) · [Model catalog](research/model_catalog.json)

## Research pipeline

```mermaid
flowchart LR
 A[Licensed prices and dynamic constituents] --> B[Date, adjustment, suspension and industry checks]
 B --> C[Price-volume formulas and trailing windows]
 C --> D[MAD clipping, industry-size neutralization, standardization]
 D --> E[Single factors: 1 / 5 / 20-session rebalancing]
 D --> F[Quarterly historical selection of 100 factors]
 F --> G[50-model prediction]
 G --> H[IC, RankIC and native rank R-squared]
 H --> I[2025-2026 new-period evaluation]
```

## Navigation

| Content | Location |
|---|---|
| Data sources, private input formats and limitations | [Data](docs/DATA.md) |
| Factor, backtest and model methodology | [Methodology](docs/METHODOLOGY.md) |
| Formula and field catalogs | [Factor catalog](factors/catalog.json), [report formulas](factors/report_formulas.json) |
| Single-factor periods and rebalancing frequencies | [Backtests](backtests/README.md) |
| Historical and new-period 50-model results | [Research](research/README.md) |
| Interactive results notebook | [Results explorer](notebooks/results_explorer.ipynb) |

## Installation and usage

Aggregate CSVs can be viewed without private data. Running the research requires Python 3.13 and independently licensed data. Pinned dependencies record the local environment; matching binary packages may not exist on every platform.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`PRIVATE_ROOT` is a local directory following the documented schema, not data bundled with this repository. Generated datasets, model weights and stock-level predictions stay in local output directories.

```bash
# Export corrected standardized inputs from the original local project, if available
python scripts/prepare_local_inputs.py --source ORIGINAL_PROJECT --out PRIVATE_ROOT

# Alternatively, build factors from your own standardized prices and industry records
python scripts/build_factors.py --root PRIVATE_ROOT --universe csi300 --industry INDUSTRY.parquet --start-year 2010 --end-year 2026

# Single-factor results with 1/5/20-session rebalancing
python scripts/run_single_factor.py --root PRIVATE_ROOT --universe csi300 --factor PV_MOM_20 --out outputs/single_factor

# Pilot one historical quarter; remove --pilot for the full interval
python scripts/train_models.py --root PRIVATE_ROOT --universe csi300 --start 2015Q1 --end 2024Q4 --out outputs/csi300_history --pilot

# New period with quarterly retraining; use --universe csi500 for the second universe
python scripts/train_models.py --root PRIVATE_ROOT --universe csi300 --start 2025Q1 --end 2026Q4 --out outputs/csi300_new_period
```

The original 15 models are Equal Weight, Ridge, Lasso, Elastic Net, PCA + Ridge, PLS, Random Forest, XGBoost, LightGBM, CatBoost, MLP, ResNet, TabNet, FT-Transformer and Mixture of Experts (MoE).

## Versions and reproducibility

Historical 2015–2024 results use corrected report factors. The 2025–2026 extension uses the same corrected engine with historical continuity checks.

For single-factor results, 154 changed factor–universe inputs were rerun, two unchanged controls were rerun, and 1550 unchanged outputs were reused. See [result lineage](backtests/result_lineage.json). Complete input hashes were unavailable for the old batch; we do not claim that all 1706 results were newly computed.

The public refactor preserves core formulas, preprocessing, model architectures, hyperparameters and evaluation definitions, with synthetic-data tests. Private source data is not distributed, so cloning this repository alone cannot reproduce every historical number. Vendor revisions, hardware and floating-point paths can also affect results. Validate a small sample first.

## Release scope and license

Included: formulas, implementations, input documentation and aggregate results. Excluded: vendor spreadsheets, raw prices, stock-level factors/predictions, model weights, brokerage PDFs and unrelated experiments. The `research` folder evaluates prediction only; it contains neither model portfolio-return backtests nor historical score-to-return calibration results.

Original code is licensed under [MIT](LICENSE). Qlib attribution and notices are retained in [THIRD_PARTY.md](THIRD_PARTY.md). This license does not grant redistribution rights to vendor datasets, source reports or external dependencies.

## Running the added model suites

Use separate output directories per suite. Each added suite also computes an equal-weight control; that control is not an additional unique model. The local published extension reused audited original selections; the portable runner recalculates the same selection protocol from normalized private inputs.

```bash
python scripts/train_models.py --root PRIVATE_ROOT --universe csi300 --suite additional15 --start 2015Q1 --end 2024Q4 --out outputs/csi300_additional15 --pilot
python scripts/train_models.py --root PRIVATE_ROOT --universe csi500 --suite additional20 --start 2025Q1 --end 2026Q4 --out outputs/csi500_additional20 --pilot
```

Remove `--pilot` for all requested quarters. The default suite remains `original15`. Neural adaptations and fixed parameter candidates are documented in the source; limited compute budgets are part of the experiment.
