# Methodology

## Factor construction

The candidate library contains 857 columns per universe: 244 Qlib Alpha158-style adaptations, 335 price/volume extensions and 278 available report-formula implementations. Window variants are not independent discoveries. Some columns are unavailable for one universe; the single-factor catalog contains 852 CSI300 and 854 CSI500 tests.

Use closing-time information, trailing windows, then daily `5 × 1.4826 × MAD` clipping, industry and log-cap OLS residuals, and population-standard-deviation normalization. Unknown industry/cap, singleton groups and constant residuals remain unavailable. Formula engine fixes use window-local arithmetic and exact two-point correlations to avoid machine-precision differences becoming artificial ranking differences.

Qlib adaptation requires complete windows and uses observed volume. This is not a byte-identical replication of Qlib's default Alpha158 data handler. Report formulas retain documented source conventions, exclusions and availability in the formula catalog. Formula-internal industries and neutralization industries are distinct.

## Single-factor tests

2010–2024 historical research; closing signal, next-session supplied VWAP or eligible close fallback. Top/bottom halves have target +50%/−50%, equal within sides; this differs from a Q5−Q1 strategy. Compare 1/5/20-session rebalancing. Frozen positions and corporate events follow the supplied ledger. Baseline costs are 10bp per buy/sell notional; other scenarios separate dated stamp taxes and assumed commission/slippage. Defaults do not include complete historical borrow fees/availability, transfer fees, impact, limits or minimum commissions. Do not claim fully executable shorting.

The sign chosen from the 2010–2018 development sample makes that part descriptive rather than a genuinely prospective trading record. Previously viewed validation/test periods must not be advertised as untouched. Report fixed history, factor-valid-start and common-period metrics separately. Arithmetic cost contributions and CAGR are different; no fee is simply subtracted from annual CAGR. `official_valid` is the software valuation flag, not an external certification.

154 changed factor/universe inputs were rerun, plus two unchanged controls; 1550 unchanged outputs were reused. Old complete input hashes were unavailable. Lineage is disclosed rather than presenting all 1706 results as newly computed. Unknown or unresolved valuation scenarios retain unavailable metrics.

## Fifteen-model prediction

Each quarter: trailing three years, already realized next-session-to-following-session VWAP return labels only. Select exactly 100 factors by historical absolute mean daily Pearson IC, orient by training IC, require coverage and valid-IC fractions ≥80%. Prioritize correlation-diverse representatives (absolute mean daily rank correlation <0.90); backfill to 100 if necessary and log it.

Inner validation uses the last 20% of historical dates, purges overlapping labels, and separately reselects features. Two fixed candidate configurations per learned model; choose by validation RankIC. Final training samples every fifth historical date but retains all eligible stocks on those dates. Predict daily. Input = 100 signed factors, 100 missing indicators and two close-known index-state features. Training-only median/scalers; no future label filter on prediction rows. Neural models use eight epochs; this is a limited-budget comparison, not proof of architectural superiority. FT-Transformer/ResNet/MoE are compact adaptations, not exact official benchmark reproductions.

2015–2024 is retrospective rolling out-of-sample, already viewed during research. 2025–2026 uses the fixed protocol with quarterly updating, not a model permanently frozen in 2024. 2026Q4 is partial. Prediction ends 2026-10-19; labels support evaluation through 2026-10-15 (430 sessions).

Daily IC = Pearson correlation of score and raw forward return. RankIC = Spearman correlation on the same valid cross-section, minimum 30 stocks. Native rank R² = `1 − sum((score − rank_target)^2) / sum(rank_target^2)`; `rank_target = 2 × daily_percentile_rank(return) − 1`, zero-score benchmark. R² is pooled across stock-days, not IC squared. Equal-weight scores are not trained on this target and have a different scale: their large negative native R² is not a fair standalone measure of model inferiority.

The public research folder contains no model portfolio-return experiment and no score-to-return calibration experiment. Negative results are retained. Differences in mean IC are not evidence of statistical superiority without paired tests and appropriate multiple-comparison handling.

## Fifty-model extension

The additional 15 and 20 configurations retain the original feature selection and evaluation protocol. See [extension protocol](../research/extension_protocol.json) for exact control reuse, candidate failures and coverage. The additional neural architectures train for eight epochs. These are related model configurations, not independent discoveries. Both evaluation windows have already been viewed. No statistical superiority or multiple-testing-adjusted winner is claimed. Constant validation predictions have undefined RankIC; select from valid candidates when possible. Preserve missing evaluation correlations and report coverage.
