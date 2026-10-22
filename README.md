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

Tables below show the top five by mean RankIC in each panel. Full results retain all 50, including negative metrics. Both periods have been viewed during research, so these are extended comparisons rather than fresh blind holdouts. Related loss-function and architecture variants are explicitly included; the catalog is not fifty independent algorithm families.

### CSI300 — 2015–2024 historical rolling evaluation

| model                 |      IC |   RankIC |
|:----------------------|--------:|---------:|
| extra_trees           | 0.03679 |  0.05039 |
| bayesian_ridge        | 0.03800 |  0.05010 |
| catboost              | 0.03785 |  0.04916 |
| spline_additive_ridge | 0.03771 |  0.04904 |
| rbf_random_features   | 0.03706 |  0.04902 |

### CSI500 — 2015–2024 historical rolling evaluation

| model              |      IC |   RankIC |
|:-------------------|--------:|---------:|
| catboost           | 0.04875 |  0.06728 |
| extra_trees        | 0.04311 |  0.06571 |
| catboost_mae       | 0.04710 |  0.06536 |
| random_forest      | 0.04434 |  0.06505 |
| laplacian_nystroem | 0.04017 |  0.06501 |

### CSI300 — 2025–2026 new-period evaluation

| model                  |       IC |   RankIC |
|:-----------------------|---------:|---------:|
| extra_trees            | -0.00418 |  0.02983 |
| bagging_trees          | -0.00014 |  0.02963 |
| least_angle_regression | -0.00206 |  0.02938 |
| catboost               | -0.00387 |  0.02892 |
| laplacian_nystroem     | -0.00567 |  0.02829 |

### CSI500 — 2025–2026 new-period evaluation

| model                   |      IC |   RankIC |
|:------------------------|--------:|---------:|
| extra_trees             | 0.00560 |  0.04527 |
| random_forest           | 0.00611 |  0.04352 |
| adaboost                | 0.00371 |  0.04326 |
| huber_gradient_boosting | 0.00571 |  0.04297 |
| laplacian_nystroem      | 0.00280 |  0.04278 |

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
