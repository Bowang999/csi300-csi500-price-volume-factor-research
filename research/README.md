# Fifty-model prediction research

Fifty estimator or architecture configurations on dynamic CSI300/CSI500 constituents. Some are related loss or architecture variants, not fifty independent algorithm families. Both periods have been viewed; comparisons are exploratory, not fresh blind holdouts.

Each quarter selects exactly 100 factors using prior realized labels. Historical evaluation: 2015-01-05–2024-12-27; new-period evaluation: 2025-01-02–2026-10-15. IC and RankIC are signed daily cross-sectional means. Native rank R² is a decimal (0.002 = 0.2%), not return R² or IC squared. No model portfolio returns or score-to-return calibration are published.

Additional-suite means use days where both IC and RankIC are defined; daily files retain individual missing values.

**Coverage:** CSI300 ARD has 2373 historical / 370 new-period valid days rather than 2429 / 430. Constant predictions retain undefined IC, not zero. Other coverage is in the tables. Missing days mean ARD is not a perfectly paired comparison.

Extra Trees leads RankIC in three of four panels; CatBoost leads historical CSI500. New-period CSI300 leaders still have negative Pearson IC. Point estimates do not establish significant superiority; fifty-model selection requires multiple-comparison control.

[All results](results/all_50_models_comparison.csv) · [Model catalog](model_catalog.json) · [Extension protocol](extension_protocol.json)

## CSI300 — historical

| model                       |       IC |   RankIC |   rank_R2 |   valid_days |
|:----------------------------|---------:|---------:|----------:|-------------:|
| extra_trees                 | 0.036792 | 0.050387 |  0.002911 |         2429 |
| bayesian_ridge              | 0.037999 | 0.050096 |  0.002510 |         2429 |
| catboost                    | 0.037847 | 0.049156 |  0.002707 |         2429 |
| spline_additive_ridge       | 0.037712 | 0.049039 |  0.000046 |         2429 |
| rbf_random_features         | 0.037064 | 0.049017 |  0.002216 |         2429 |
| polynomial_sketch_ridge     | 0.038857 | 0.048924 |  0.001443 |         2429 |
| elastic_net                 | 0.037145 | 0.048812 |  0.002263 |         2429 |
| lasso                       | 0.036518 | 0.048705 |  0.002233 |         2429 |
| laplacian_nystroem          | 0.034368 | 0.048515 |  0.002325 |         2429 |
| pca_ridge                   | 0.035832 | 0.047965 |  0.001912 |         2429 |
| xgboost                     | 0.037670 | 0.047661 |  0.001514 |         2429 |
| random_forest               | 0.036304 | 0.047541 |  0.002299 |         2429 |
| ft_transformer              | 0.035414 | 0.047488 | -0.004464 |         2429 |
| ridge                       | 0.036560 | 0.047106 |  0.000372 |         2429 |
| pls                         | 0.035451 | 0.047084 |  0.000671 |         2429 |
| catboost_mae                | 0.035369 | 0.046885 | -0.002722 |         2429 |
| hist_gradient_boosting      | 0.036534 | 0.046384 |  0.001447 |         2429 |
| huber_gradient_boosting     | 0.035068 | 0.046120 |  0.002205 |         2429 |
| huber_regression            | 0.036516 | 0.045950 | -0.008048 |         2429 |
| ordinary_least_squares      | 0.036014 | 0.045787 | -0.001024 |         2429 |
| linear_svr                  | 0.035991 | 0.045759 | -0.000447 |         2429 |
| lightgbm                    | 0.035792 | 0.045676 |  0.001422 |         2429 |
| bagging_trees               | 0.035135 | 0.045569 |  0.002070 |         2429 |
| boosted_stumps              | 0.032945 | 0.045242 |  0.002537 |         2429 |
| median_hist_boosting        | 0.035357 | 0.045237 | -0.005358 |         2429 |
| random_projection_ridge     | 0.033735 | 0.045012 |  0.001087 |         2429 |
| passive_aggressive          | 0.034596 | 0.044955 | -0.007347 |         2429 |
| lightgbm_dart               | 0.034287 | 0.044840 |  0.002407 |         2429 |
| orthogonal_matching_pursuit | 0.033522 | 0.044378 |  0.000960 |         2429 |
| extreme_learning_machine    | 0.032074 | 0.043752 | -0.001030 |         2429 |
| adaboost                    | 0.030033 | 0.043481 |  0.002216 |         2429 |
| least_angle_regression      | 0.031716 | 0.042875 |  0.001386 |         2429 |
| ard_regression              | 0.032692 | 0.041738 |  0.001072 |         2373 |
| soft_threshold_network      | 0.032071 | 0.041657 | -0.005536 |         2429 |
| gated_mlp                   | 0.034895 | 0.039958 | -0.014358 |         2429 |
| neural_additive             | 0.032565 | 0.039781 | -0.007284 |         2429 |
| highway_network             | 0.033055 | 0.038971 | -0.021855 |         2429 |
| moe                         | 0.031806 | 0.038433 | -0.022892 |         2429 |
| deep_cross_network          | 0.031594 | 0.038223 | -0.019736 |         2429 |
| supervised_autoencoder      | 0.030836 | 0.037657 | -0.015862 |         2429 |
| mlp                         | 0.030432 | 0.036009 | -0.020502 |         2429 |
| low_rank_bilinear           | 0.030950 | 0.035551 | -0.025917 |         2429 |
| factorization_machine       | 0.028873 | 0.035343 | -0.026693 |         2429 |
| equal_weight                | 0.026270 | 0.035058 | -0.598332 |         2429 |
| resnet                      | 0.029250 | 0.035035 | -0.019663 |         2429 |
| pca_knn                     | 0.026171 | 0.034947 | -0.002048 |         2429 |
| cross_network               | 0.028829 | 0.034699 | -0.004406 |         2429 |
| deep_fm_numeric             | 0.026712 | 0.030499 | -0.036977 |         2429 |
| decision_tree               | 0.021259 | 0.028762 | -0.002851 |         2429 |
| tabnet                      | 0.019818 | 0.028268 |  0.000198 |         2429 |

## CSI500 — historical

| model                       |       IC |   RankIC |   rank_R2 |   valid_days |
|:----------------------------|---------:|---------:|----------:|-------------:|
| catboost                    | 0.048745 | 0.067280 |  0.005527 |         2429 |
| extra_trees                 | 0.043107 | 0.065710 |  0.005189 |         2429 |
| catboost_mae                | 0.047099 | 0.065365 | -0.000661 |         2429 |
| random_forest               | 0.044341 | 0.065047 |  0.004998 |         2429 |
| laplacian_nystroem          | 0.040175 | 0.065009 |  0.004537 |         2429 |
| bagging_trees               | 0.043605 | 0.064767 |  0.004852 |         2429 |
| xgboost                     | 0.046695 | 0.064566 |  0.004590 |         2429 |
| lightgbm                    | 0.045968 | 0.063811 |  0.004625 |         2429 |
| hist_gradient_boosting      | 0.045258 | 0.063550 |  0.004575 |         2429 |
| rbf_random_features         | 0.044521 | 0.063456 |  0.004428 |         2429 |
| huber_gradient_boosting     | 0.043037 | 0.063236 |  0.004467 |         2429 |
| lightgbm_dart               | 0.043387 | 0.063086 |  0.004909 |         2429 |
| spline_additive_ridge       | 0.045997 | 0.063049 |  0.003502 |         2429 |
| bayesian_ridge              | 0.043988 | 0.062458 |  0.004121 |         2429 |
| elastic_net                 | 0.043362 | 0.062280 |  0.004134 |         2429 |
| median_hist_boosting        | 0.044535 | 0.062232 | -0.003122 |         2429 |
| ft_transformer              | 0.043764 | 0.062081 |  0.000469 |         2429 |
| lasso                       | 0.041889 | 0.061753 |  0.004220 |         2429 |
| boosted_stumps              | 0.037678 | 0.060757 |  0.004288 |         2429 |
| polynomial_sketch_ridge     | 0.044303 | 0.060635 |  0.003611 |         2429 |
| pca_ridge                   | 0.040648 | 0.059867 |  0.003623 |         2429 |
| ridge                       | 0.042914 | 0.059783 |  0.002829 |         2429 |
| pls                         | 0.042731 | 0.059697 |  0.002863 |         2429 |
| huber_regression            | 0.043232 | 0.059371 | -0.004616 |         2429 |
| orthogonal_matching_pursuit | 0.041463 | 0.059321 |  0.003283 |         2429 |
| linear_svr                  | 0.042911 | 0.059128 |  0.002386 |         2429 |
| adaboost                    | 0.034702 | 0.058720 |  0.003676 |         2429 |
| ordinary_least_squares      | 0.042903 | 0.058574 |  0.002123 |         2429 |
| random_projection_ridge     | 0.041251 | 0.058319 |  0.003277 |         2429 |
| passive_aggressive          | 0.041695 | 0.058113 | -0.003807 |         2429 |
| extreme_learning_machine    | 0.039797 | 0.058073 |  0.001789 |         2429 |
| gated_mlp                   | 0.046500 | 0.056786 | -0.007905 |         2429 |
| highway_network             | 0.046387 | 0.056703 | -0.011694 |         2429 |
| supervised_autoencoder      | 0.046571 | 0.056160 | -0.006303 |         2429 |
| ard_regression              | 0.038451 | 0.055112 |  0.002591 |         2429 |
| deep_cross_network          | 0.043698 | 0.054630 | -0.011763 |         2429 |
| mlp                         | 0.044287 | 0.053387 | -0.010209 |         2429 |
| least_angle_regression      | 0.033486 | 0.052455 |  0.001826 |         2429 |
| resnet                      | 0.041941 | 0.051894 | -0.008401 |         2429 |
| moe                         | 0.043994 | 0.051591 | -0.013498 |         2429 |
| soft_threshold_network      | 0.038911 | 0.051417 | -0.002787 |         2429 |
| pca_knn                     | 0.034314 | 0.051046 |  0.000431 |         2429 |
| low_rank_bilinear           | 0.041120 | 0.050655 | -0.015182 |         2429 |
| neural_additive             | 0.037988 | 0.050270 | -0.004682 |         2429 |
| factorization_machine       | 0.039388 | 0.048463 | -0.015847 |         2429 |
| equal_weight                | 0.028959 | 0.047153 | -0.624821 |         2429 |
| decision_tree               | 0.027787 | 0.045100 | -0.000620 |         2429 |
| tabnet                      | 0.025123 | 0.044633 |  0.001514 |         2429 |
| deep_fm_numeric             | 0.037413 | 0.044183 | -0.023064 |         2429 |
| cross_network               | 0.034480 | 0.044111 | -0.001406 |         2429 |

## CSI300 — new_period

| model                       |        IC |   RankIC |   rank_R2 |   valid_days |
|:----------------------------|----------:|---------:|----------:|-------------:|
| extra_trees                 | -0.004180 | 0.029829 |  0.000415 |          430 |
| bagging_trees               | -0.000140 | 0.029627 | -0.000091 |          430 |
| least_angle_regression      | -0.002064 | 0.029381 |  0.000812 |          430 |
| catboost                    | -0.003874 | 0.028923 | -0.000648 |          430 |
| laplacian_nystroem          | -0.005674 | 0.028290 |  0.000037 |          430 |
| random_forest               | -0.004233 | 0.028267 | -0.000690 |          430 |
| median_hist_boosting        | -0.005173 | 0.028090 | -0.006994 |          430 |
| catboost_mae                | -0.004297 | 0.028084 | -0.006478 |          430 |
| adaboost                    | -0.003530 | 0.028008 |  0.000381 |          430 |
| huber_gradient_boosting     | -0.004602 | 0.027924 | -0.000407 |          430 |
| ft_transformer              |  0.000390 | 0.027295 | -0.008088 |          430 |
| boosted_stumps              | -0.006919 | 0.026767 |  0.000540 |          430 |
| lightgbm_dart               | -0.005038 | 0.025692 |  0.000113 |          430 |
| lightgbm                    | -0.004820 | 0.025617 | -0.001306 |          430 |
| bayesian_ridge              | -0.005184 | 0.024759 | -0.000408 |          430 |
| spline_additive_ridge       | -0.001785 | 0.024656 | -0.003357 |          430 |
| xgboost                     | -0.005737 | 0.023646 | -0.002937 |          430 |
| hist_gradient_boosting      | -0.005480 | 0.023428 | -0.002303 |          430 |
| extreme_learning_machine    | -0.000582 | 0.022949 | -0.003616 |          430 |
| random_projection_ridge     | -0.002879 | 0.022807 | -0.001116 |          430 |
| equal_weight                | -0.010832 | 0.021898 | -0.659483 |          430 |
| pca_ridge                   | -0.005515 | 0.021838 | -0.000971 |          430 |
| elastic_net                 | -0.008856 | 0.021712 | -0.000271 |          430 |
| rbf_random_features         | -0.009679 | 0.021261 | -0.002806 |          430 |
| decision_tree               | -0.001768 | 0.020534 | -0.005922 |          430 |
| polynomial_sketch_ridge     | -0.001119 | 0.020431 | -0.002804 |          430 |
| pls                         | -0.005785 | 0.020413 | -0.002973 |          430 |
| huber_regression            | -0.003626 | 0.020363 | -0.008694 |          430 |
| passive_aggressive          | -0.002346 | 0.020272 | -0.009161 |          430 |
| linear_svr                  | -0.004234 | 0.020145 | -0.004001 |          430 |
| ridge                       | -0.005527 | 0.020004 | -0.003334 |          430 |
| low_rank_bilinear           |  0.000040 | 0.019750 | -0.022487 |          430 |
| ordinary_least_squares      | -0.004145 | 0.019606 | -0.004774 |          430 |
| lasso                       | -0.010479 | 0.019496 | -0.000063 |          430 |
| tabnet                      |  0.000089 | 0.018483 | -0.000920 |          430 |
| orthogonal_matching_pursuit | -0.008774 | 0.018425 | -0.002185 |          430 |
| ard_regression              | -0.006125 | 0.017951 | -0.001457 |          370 |
| deep_fm_numeric             | -0.000226 | 0.017794 | -0.035256 |          430 |
| highway_network             | -0.004217 | 0.017191 | -0.026875 |          430 |
| pca_knn                     | -0.002635 | 0.015811 | -0.007246 |          430 |
| cross_network               | -0.002436 | 0.015530 | -0.007925 |          430 |
| supervised_autoencoder      | -0.005423 | 0.015114 | -0.018731 |          430 |
| factorization_machine       |  0.002922 | 0.015049 | -0.021930 |          430 |
| soft_threshold_network      | -0.004438 | 0.014816 | -0.009403 |          430 |
| deep_cross_network          |  0.002920 | 0.014486 | -0.018315 |          430 |
| gated_mlp                   | -0.004039 | 0.014484 | -0.016251 |          430 |
| resnet                      | -0.005790 | 0.014420 | -0.021169 |          430 |
| mlp                         | -0.000362 | 0.010782 | -0.024272 |          430 |
| moe                         | -0.008042 | 0.006623 | -0.032351 |          430 |
| neural_additive             | -0.014518 | 0.006175 | -0.009928 |          430 |

## CSI500 — new_period

| model                       |        IC |   RankIC |   rank_R2 |   valid_days |
|:----------------------------|----------:|---------:|----------:|-------------:|
| extra_trees                 |  0.005604 | 0.045269 |  0.002072 |          430 |
| random_forest               |  0.006113 | 0.043524 |  0.001821 |          430 |
| adaboost                    |  0.003706 | 0.043256 |  0.001865 |          430 |
| huber_gradient_boosting     |  0.005709 | 0.042971 |  0.001400 |          430 |
| laplacian_nystroem          |  0.002804 | 0.042779 |  0.001681 |          430 |
| catboost_mae                |  0.003763 | 0.042515 | -0.004403 |          430 |
| lightgbm_dart               |  0.004554 | 0.042276 |  0.001830 |          430 |
| bagging_trees               |  0.005374 | 0.042227 |  0.001601 |          430 |
| catboost                    |  0.003553 | 0.042071 |  0.001506 |          430 |
| boosted_stumps              |  0.002230 | 0.041361 |  0.001830 |          430 |
| median_hist_boosting        |  0.005404 | 0.040289 | -0.005820 |          430 |
| hist_gradient_boosting      |  0.004283 | 0.040185 |  0.001092 |          430 |
| xgboost                     |  0.006301 | 0.039961 |  0.000828 |          430 |
| ft_transformer              |  0.005573 | 0.039955 | -0.001695 |          430 |
| equal_weight                |  0.001660 | 0.039922 | -0.690152 |          430 |
| lightgbm                    |  0.003248 | 0.038181 |  0.000644 |          430 |
| pca_ridge                   |  0.003717 | 0.037470 |  0.001076 |          430 |
| rbf_random_features         |  0.004114 | 0.035265 |  0.000323 |          430 |
| elastic_net                 | -0.001602 | 0.034958 |  0.000883 |          430 |
| bayesian_ridge              |  0.003696 | 0.034218 |  0.000327 |          430 |
| soft_threshold_network      |  0.007466 | 0.034177 | -0.002894 |          430 |
| least_angle_regression      | -0.001426 | 0.033473 |  0.001090 |          430 |
| random_projection_ridge     |  0.003341 | 0.033446 | -0.000061 |          430 |
| spline_additive_ridge       |  0.003577 | 0.033292 | -0.001049 |          430 |
| orthogonal_matching_pursuit |  0.003157 | 0.032385 | -0.000197 |          430 |
| polynomial_sketch_ridge     |  0.004081 | 0.032204 | -0.001108 |          430 |
| lasso                       | -0.004694 | 0.032066 |  0.000978 |          430 |
| pca_knn                     |  0.002551 | 0.031835 | -0.002872 |          430 |
| pls                         |  0.003874 | 0.031409 | -0.000603 |          430 |
| decision_tree               |  0.004515 | 0.030975 | -0.000906 |          430 |
| ard_regression              | -0.001241 | 0.030849 | -0.000170 |          430 |
| ridge                       |  0.003178 | 0.030488 | -0.001421 |          430 |
| extreme_learning_machine    |  0.002017 | 0.030459 | -0.001641 |          430 |
| tabnet                      |  0.003239 | 0.029724 | -0.000298 |          430 |
| huber_regression            |  0.003559 | 0.029724 | -0.008982 |          430 |
| ordinary_least_squares      |  0.002853 | 0.029388 | -0.002332 |          430 |
| linear_svr                  |  0.002714 | 0.029167 | -0.001992 |          430 |
| passive_aggressive          |  0.001597 | 0.028210 | -0.009321 |          430 |
| neural_additive             |  0.008974 | 0.021465 | -0.008041 |          430 |
| resnet                      | -0.001723 | 0.021261 | -0.020828 |          430 |
| highway_network             |  0.000130 | 0.020631 | -0.015855 |          430 |
| supervised_autoencoder      | -0.002150 | 0.018220 | -0.016063 |          430 |
| factorization_machine       |  0.007393 | 0.018129 | -0.016773 |          430 |
| low_rank_bilinear           |  0.004459 | 0.016142 | -0.021324 |          430 |
| mlp                         | -0.001593 | 0.015830 | -0.020776 |          430 |
| gated_mlp                   | -0.002633 | 0.014978 | -0.015726 |          430 |
| deep_cross_network          | -0.003829 | 0.014130 | -0.020100 |          430 |
| deep_fm_numeric             | -0.009547 | 0.013515 | -0.040565 |          430 |
| moe                         | -0.002927 | 0.013205 | -0.030079 |          430 |
| cross_network               | -0.007869 | 0.004885 | -0.006304 |          430 |

## Implementations and references

Original models: `src/model_helpers.py` and `src/model_pipeline.py`. Added configurations: `src/additional_models_15.py` and `src/additional_models_20.py`. Compact DeepFM, cross-network and neural-additive implementations are numeric adaptations, not exact reproductions of published benchmarks.

- [scikit-learn estimator reference](https://scikit-learn.org/stable/api/index.html)
- [DeepFM](https://arxiv.org/abs/1703.04247)
- [Neural Additive Models](https://arxiv.org/abs/2004.13912)

The original fifteen-model result files remain available for provenance. New daily files contain date-level aggregate correlations only, never stock-level predictions.