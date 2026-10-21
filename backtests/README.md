# Single-factor backtest results

852 CSI300 and 854 CSI500 factor-universe pairs (1706 total), each with 1/5/20-session rebalancing. Download the CSV files and explore them with pandas or the notebook; private data is not required to browse results.

- `results/research_summary.csv`: development-period direction, data/valuation limitations, key performance periods and research screening diagnostics.
- `results/all_period_metrics.csv`: cost/execution scenarios and fixed, valid-start, common, validation and previously viewed test periods.
- `results/rebalance_frequency_summary.csv`: major metrics by 1/5/20-session rebalancing.
- `results/representative_single_factor_ic.csv`: signed one-day IC/RankIC means for the homepage examples, with valid-day counts and dates.
- `result_lineage.json`: rerun and reuse counts.
- `validation/`: accounting tests with hand-checkable outcomes and pilot checks.

Return, volatility and turnover fields are generally decimals, not percentages; Sharpe is dimensionless. `CAGR` uses 252 trading sessions; `CAGR_calendar` uses actual dates. Filter `scenario`, `period` and `rebalance_days` together; do not directly compare different periods or cost assumptions.

The baseline targets 50% long and 50% short, equally weighted within each half, trading the next session after the closing signal. `baseline_10bp` charges 10 bp on each buy/sell. `conservative_one_price` adds single-price restrictions and historical tax assumptions. Complete stock-loan, price-limit and impact data is unavailable; these are research scenarios. Between scheduled rebalances targets are not actively changed, but corporate actions and valuations can change.

Unresolved valuation scenarios retain unavailable metrics. `official_valid` is not a real-world execution certification. Selecting direction in development means development returns are not independent tests. Do not choose direction after viewing test performance and then claim out-of-sample success.

```python
import pandas as pd
d = pd.read_csv('backtests/results/rebalance_frequency_summary.csv')
print(d.query("universe == 'csi300' and factor == 'PV_MOM_20' and scenario == 'baseline_10bp' and period == 'FIXED_2010_2024'")[['rebalance_days','CAGR_calendar','Sharpe','max_drawdown','daily_two_way_turnover']])
```
