# 单因子回测结果

852份沪深300＋854份中证500，共1706份；每份包含1、5、20交易日调仓。可直接下载CSV并用pandas或笔记本筛选，无需私有数据。

- `results/research_summary.csv`：每个因子的研究期方向、数据与估值限制、关键时期绩效及研究筛选诊断。
- `results/all_period_metrics.csv`：不同费用/交易情景、全期/有效起始/共同期/验证期/已查看测试期指标。
- `results/rebalance_frequency_summary.csv`：以上主要指标按1、5、20日调仓展开。
- `result_lineage.json`：重跑与复用数量。
- `validation/`：手工可核算的记账测试及试跑检查结果。

收益、波动、换手字段通常是小数，非百分数；Sharpe无量纲。`CAGR`使用252交易日年化，`CAGR_calendar`使用实际日期。必须同时筛选`scenario`、`period`与`rebalance_days`，不能把不同时期或费用假设的指标直接比较。

核心基准为50%多头、50%空头、两侧组内等权，前收盘信号后下一交易日成交。`baseline_10bp`买卖各10bp。`conservative_one_price`加入一字价限制和历史税费假设。完整券源、价格限制和冲击成本不具备，结果是研究情景。非调仓日不主动变更目标持仓，但企业行动及估值仍可变化。

未知估值情景保留不可用指标；勿将`official_valid`误解为真实可交易性认证。研究期确定方向意味着研究期收益不是严格独立检验。不要根据已经查看的测试期挑选方向后再宣称样本外成功。

```python
import pandas as pd
d = pd.read_csv('backtests/results/rebalance_frequency_summary.csv')
print(d.query("universe == 'csi300' and factor == 'PV_MOM_20' and scenario == 'baseline_10bp' and period == 'FIXED_2010_2024'")[['rebalance_days','CAGR_calendar','Sharpe','max_drawdown','daily_two_way_turnover']])
```
