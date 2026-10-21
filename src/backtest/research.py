"""Versioned research audit. Old notebooks and original engine remain unchanged."""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from . import factor_backtest as legacy
from .daily_execution import simulate,half_weights,full_metrics,pnl_table

@dataclass
class Config(legacy.Config):
    common_start: str='2019-01-02'
    commission_bps: float=3.
    slippage_bps: float=2.
    rolling_train_years: int=3
    rolling_min_ic_days: int=252
    rebalance_intervals: tuple=(1,5,20)
    rebalance_offset: int=0

def rebalance_schedule(dates,first_eligible,every,offset=0):
    if not isinstance(every,(int,np.integer)) or every<1:raise ValueError('Invalid interval')
    if not isinstance(offset,(int,np.integer)) or offset<0:raise ValueError('Invalid offset')
    mask=pd.Series(False,index=dates)
    if pd.notna(first_eligible):
        start=dates.searchsorted(first_eligible)+offset%every
        mask.iloc[start::every]=True
    return mask

def terminal_value_at_cost(target,market,bps):
    """An insolvent stress-cost ledger brackets a loss, not a valid return series."""
    try:
        p=simulate(target,market,bps,bps)
    except ValueError as exc:
        if 'Portfolio insolvent' not in str(exc):raise
        return -np.inf
    return float(p.nav.iloc[-1]) if p.official_valid.all() else np.nan

def metrics(series,annual=252,rf=0):
    """252-day convention plus elapsed calendar years, including first return day."""
    m=full_metrics(series,annual,rf)
    years=((series.index[-1]-series.index[0]).days+1)/365.2425 if len(series) else np.nan
    m['CAGR_calendar']=np.expm1(np.log1p(m['total_return'])/years) if years>0 and np.isfinite(m['total_return']) else np.nan
    m['calendar_years']=years
    return m

def historical_fees(dates,commission=3.,slippage=2.):
    """Only verified historical stamp tax; other inputs are explicit assumptions.

    Transfer fee is NOT claimed complete; zero placeholder flagged in every report.
    Slippage is a cash debit against VWAP not an estimated market-impact model.
    """
    if commission<0 or slippage<0:raise ValueError('Negative cost')
    f=pd.DataFrame(index=dates)
    for side in ('buy','sell'):
        f['commission_'+side]=commission
        f['slippage_'+side]=slippage
        f['transfer_'+side]=0.
    f['stamp_buy']=0.
    f['stamp_sell']=np.where(dates<pd.Timestamp('2023-08-28'),10.,5.)
    return f

@lru_cache(maxsize=2)
def restrictions(root,universe,start,end):
    cols=['date','symbol','high_hfq','low_hfq','provider_trade_status','provider_is_st']
    parts=[]
    # Successor symbols can come from the other universe. Prefer own observations.
    for u in (universe,'csi500' if universe=='csi300' else 'csi300'):
        d=pd.read_parquet(Path(root)/f'data/{u}/parquet/{u}_daily_market_2010_2024.parquet',columns=cols)
        d.date=pd.to_datetime(d.date);parts.append(d.loc[d.date.between(start,end)])
    d=pd.concat(parts).drop_duplicates(['date','symbol'])
    # Do not infer 5/10/20% price limits from HFQ series or infer ST from missing data.
    # One-price bars are conservatively blocked in BOTH directions as a scenario.
    d['one_price']=d.high_hfq.gt(0)&d.low_hfq.gt(0)&np.isclose(d.high_hfq,d.low_hfq,rtol=0,atol=1e-8)
    # Provider status semantics are not documented; do not guess 0/1 meaning.
    return {k:d.pivot(index='date',columns='symbol',values=k) for k in ['one_price','provider_trade_status','provider_is_st']}

def restricted_market(market,extra):
    out=dict(market);fill=market['fill'].copy()
    lock=extra['one_price'].reindex_like(fill).fillna(False).astype(bool)
    out['fill']=fill.mask(lock)
    return out

def summary_row(name,p,annual,rf,start=None,end=None):
    x=p.loc[start:end];s=x.net.where(x.official_valid)
    return dict(scenario=name,from_date=str(x.index.min().date()) if len(x) else '',to_date=str(x.index.max().date()) if len(x) else '',
        **metrics(s,annual,rf),daily_two_way_turnover=x.two_way_turnover.mean(),
        arithmetic_annual_cost=x.cost.mean()*annual,official_valid=bool(len(x) and x.official_valid.all()))

def yearly_direction(ic,dates,years=3,minimum=252):
    """Relearn once per calendar year; discard labels exiting on/after cutoff.

    1d label at signal t exits t+2. Purging uses positions in trading calendar.
    """
    exits=pd.Series(dates,index=dates).shift(-2)
    directions=pd.Series(0.,index=dates);rows=[]
    for year in sorted(set(dates.year)):
        cutoff=pd.Timestamp(year=year,month=1,day=1)
        train=ic.loc[(ic.index>=pd.Timestamp(year=year-years,month=1,day=1))&(exits<cutoff)]
        valid=train.dropna();sign=float(np.sign(valid.mean())) if len(valid)>=minimum else 0.
        if not np.isfinite(sign):sign=0.
        directions.loc[dates.year==year]=sign
        rows.append(dict(year=year,direction=int(sign),train_ic_days=len(valid),train_RankIC=valid.mean(),
                         latest_label_exit=str(exits.loc[valid.index].max()) if len(valid) else ''))
    return directions,pd.DataFrame(rows)

def run_backtest(config,root=None):
    c=config;root=Path(root) if root else legacy.find_root()
    if c.annual_days!=252 or c.risk_free_annual!=0:
        raise ValueError('V3 comparison currently requires annual_days=252 and risk_free_annual=0 for consistent scenario conventions')
    if tuple(c.rebalance_intervals)!=(1,5,20):raise ValueError('This report compares rebalance_intervals=(1,5,20)')
    # Always load the entire fixed research interval; never hide warmup cash days.
    c=replace(c,start='2010-01-04',end='2024-12-31',direction=1)
    old=legacy.Config(**{k:v for k,v in asdict(c).items() if k in legacy.Config.__dataclass_fields__})
    r=legacy.run_backtest(old,root);r['config']=asdict(c)
    f=r.pop('_factors');market=r.pop('_market');benchmark=r.pop('_index').reindex(f.index)
    available=f.notna().sum(axis=1).ge(c.min_stocks)
    eligible=available.shift(1,fill_value=False)
    first=eligible[eligible].index.min()
    exits=pd.Series(f.index,index=f.index).shift(-2)
    train=r['ic'].RankIC_1d.loc[exits.le(pd.Timestamp('2018-12-31'))].dropna()
    direction=int(np.sign(train.mean())) if len(train)>=c.rolling_min_ic_days and np.isfinite(train.mean()) else 0
    target=half_weights(f,c.min_stocks)*direction
    base=r['ls'] if direction==1 else r['reverse_ls'] if direction==-1 else simulate(target,market)
    extra=restrictions(str(root),c.universe,c.start,c.end);guard=restricted_market(market,extra)
    fee=historical_fees(f.index,c.commission_bps,c.slippage_bps)
    historical=simulate(target,market,fee_schedule=fee,borrow_bps=c.borrow_cost_bps_annual)
    guarded=simulate(target,guard,fee_schedule=fee,borrow_bps=c.borrow_cost_bps_annual)
    no_fallback=dict(guard);no_fallback['fill']=guard['fill'].mask(guard['fallback'].fillna(False).astype(bool))
    vwap_only=simulate(target,no_fallback,fee_schedule=fee,borrow_bps=c.borrow_cost_bps_annual)
    long=simulate(target.clip(lower=0)*2,guard,fee_schedule=fee,cash_only=True)
    zero=simulate(target,guard,0,0)
    rollsign,rolltable=yearly_direction(r['ic'].RankIC_1d,f.index,c.rolling_train_years,c.rolling_min_ic_days)
    rolling=simulate(half_weights(f,c.min_stocks).mul(rollsign,axis=0),guard,fee_schedule=fee,borrow_bps=c.borrow_cost_bps_annual)
    scenarios={'baseline_10bp':base,'historical_stamp_assumed_cost':historical,'conservative_one_price':guarded,
               'vwap_only':vwap_only,'long_only':long,'rolling_direction':rolling,'zero_cost':zero}
    valuation=[]
    if '000540' in target.columns and guarded.estimated_mark_exposure.gt(0).any():
        anchor=market['mark'].loc[pd.Timestamp('2023-05-18'),'000540']/.4
        selection=market['estimated_mark']['000540'].fillna(False).astype(bool)
        for raw in (0.,.01,.11,.14):
            sm=dict(guard);sm['mark']=guard['mark'].copy();sm['mark'].loc[selection,'000540']=raw*anchor
            p=simulate(target,sm,fee_schedule=fee,borrow_bps=c.borrow_cost_bps_annual)
            valuation.append(dict(raw_valuation=raw,**metrics(p.net.where(p.official_valid))))
    rows=[]
    ranges={'FIXED_2010_2024':(None,None),'FACTOR_VALID_START':(first,None),'COMMON_2019_2024':(c.common_start,None),
            'development':(None,'2018-12-31'),'validation':('2019-01-01','2021-12-31'),'test_seen':('2022-01-01',None)}
    for name,p in scenarios.items():
        for period,(start,end) in ranges.items():
            if pd.isna(first) and period=='FACTOR_VALID_START':continue
            row=summary_row(name,p,c.annual_days,c.risk_free_annual,start,end);row['period']=period;rows.append(row)
    frequency_ledgers={};frequency_rows=[];frequency_yearly=[];frequency_audit=[]
    for every in c.rebalance_intervals:
        schedule=rebalance_schedule(f.index,first,every,c.rebalance_offset)
        # Common training direction, common data/costs. Only order dates differ.
        for name,ledger in [('baseline_10bp',base),('conservative_one_price',guarded),('long_only',long)]:
            if every==1:p=ledger
            elif name=='baseline_10bp':p=simulate(target,market,c.buy_cost_bps,c.sell_cost_bps,c.borrow_cost_bps_annual,rebalance_mask=schedule)
            else:p=simulate(target.clip(lower=0)*2 if name=='long_only' else target,guard,fee_schedule=fee,
                           borrow_bps=0 if name=='long_only' else c.borrow_cost_bps_annual,cash_only=name=='long_only',rebalance_mask=schedule)
            if every>1:
                assert p.loc[~schedule,['buys','sells','cost']].abs().to_numpy().max()==0,'Non-rebalance trading'
            assert np.allclose(p.net,p.gross_same_holdings-p.cost-p.borrow_cost,atol=1e-10)
            if name=='long_only':assert p.cash_balance.min()>-1e-10
            frequency_ledgers[(name,every)]=p
            for period,(start,end) in ranges.items():
                if pd.isna(first) and period=='FACTOR_VALID_START':continue
                x=p.loc[start:end]
                frequency_rows.append(dict(rebalance_days=every,period=period,**summary_row(name,p,252,0,start,end),
                    trading_days=int(x.two_way_turnover.gt(1e-12).sum()),
                    planned_rebalance_days=int(schedule.loc[start:end].sum())))
            table=pnl_table(p).reset_index();table['rebalance_days']=every;table['scenario']=name;frequency_yearly.append(table)
            frequency_audit.append(dict(scenario=name,rebalance_days=every,first_scheduled_execution=str(schedule[schedule].index.min()),
                official_valid=bool(p.official_valid.all()),first_invalid_date=str(p.index[~p.official_valid].min()),
                valuation_events=json.dumps(p.attrs.get('valuation_events',[]),ensure_ascii=False)))
    # Relative wealth is compounded portfolio/index NAV ratio, not compounded r_p-r_b.
    relative=(1+long.net.where(long.official_valid))/(1+benchmark)-1
    active=long.net.where(long.official_valid)-benchmark
    excess=[]
    for period,(start,end) in ranges.items():
        s=relative.loc[start:end];a=active.loc[start:end]
        excess.append(dict(period=period,**metrics(s),tracking_error=a.std(ddof=1)*np.sqrt(252),
            information_ratio=a.mean()/a.std(ddof=1)*np.sqrt(252) if a.std(ddof=1)>0 else np.nan))
    # Every sensitivity is an independent cash ledger, never subtract annual cost from CAGR.
    sensitivity=[]
    for bps in (0,5,10,20,30):
        p=zero if bps==0 else simulate(target,guard,bps,bps)
        sensitivity.append(dict(cost_bps_each_side=bps,**metrics(p.net.where(p.official_valid))))
    # Break-even arithmetic approximation is clearly separate from exact log-NAV root.
    approx=zero.net.mean()/zero.two_way_turnover.mean()*10000 if zero.two_way_turnover.mean()>0 else np.nan
    root_bps=np.nan;root_status='no_positive_zero_cost_return'
    if zero.official_valid.all() and zero.nav.iloc[-1]>1:
        lo,hi=0.,100.;upper=terminal_value_at_cost(target,guard,hi)
        if upper<1:
            for _ in range(9):
                mid=(lo+hi)/2;terminal=terminal_value_at_cost(target,guard,mid)
                if np.isnan(terminal):raise ValueError('Unresolved valuation during cost search')
                if terminal>1:lo=mid
                else:hi=mid
            root_bps=(lo+hi)/2;root_status='independent_ledger_bisection_tolerance_0.1bp'
        else:root_status='not_bracketed_below_100bp'
    firstsignal=available[available].index.min()
    member=f.notna();known=extra['provider_trade_status'].reindex_like(f).notna()
    audit=dict(direction_from_development=direction,training_ic_days=len(train),development_purged_RankIC=train.mean(),
        first_valid_signal=str(firstsignal),first_eligible_execution=str(first),warmup_cash_days=int((f.index<first).sum()) if pd.notna(first) else len(f),
        later_insufficient_signal_days=int((~eligible&(f.index>=first)).sum()) if pd.notna(first) else 0,
        common_period_eligible=bool(pd.notna(first) and first<=pd.Timestamp(c.common_start)),
        provider_status_known_factor_fraction=float((member&known).to_numpy().sum()/max(1,member.to_numpy().sum())),
        price_limit_history_complete=False,borrow_availability_known=False,transfer_fee_history_complete=False,
        common_period=c.common_start+' to 2024-12-31',test_is_previously_seen=True,
        fallback_turnover_fraction=float(guarded.fallback_notional.sum()/guarded.two_way_turnover.sum()) if guarded.two_way_turnover.sum()>0 else 0,
        vwap_turnover_fraction=float(1-guarded.fallback_notional.sum()/guarded.two_way_turnover.sum()) if guarded.two_way_turnover.sum()>0 else 0,
        fallback_scenario_CAGR_difference=metrics(guarded.net.where(guarded.official_valid))['CAGR']-metrics(vwap_only.net.where(vwap_only.official_valid))['CAGR'],
        scenario_valuation_events={k:p.attrs.get('valuation_events',[]) for k,p in scenarios.items()},
        scenario_first_invalid_date={k:str(p.index[~p.official_valid].min()) if (~p.official_valid).any() else '' for k,p in scenarios.items()},
        breakeven_each_side_bps=root_bps,breakeven_status=root_status,breakeven_arithmetic_approx_bps=approx)
    assert long.cash_balance.min()>-1e-10,'Long-only cash borrowing'
    for p in scenarios.values():
        assert np.allclose(p.net,p.gross_same_holdings-p.cost-p.borrow_cost,atol=1e-10)
    r.update(v3_scenarios=scenarios,v3_comparison=pd.DataFrame(rows).set_index(['scenario','period']),
             v3_audit=audit,v3_rolling=rolltable,v3_excess=pd.DataFrame(excess).set_index('period'),
             v3_cost_sensitivity=pd.DataFrame(sensitivity).set_index('cost_bps_each_side'),v3_benchmark=benchmark,
             v3_valuation_sensitivity=pd.DataFrame(valuation))
    r.update(frequency_ledgers=frequency_ledgers,
             frequency_comparison=pd.DataFrame(frequency_rows).set_index(['scenario','period','rebalance_days']),
             frequency_yearly=pd.concat(frequency_yearly,ignore_index=True),frequency_audit=pd.DataFrame(frequency_audit))
    return r

def show_report(r):
    from IPython.display import display,Markdown
    import matplotlib.pyplot as plt
    def md(x):display(Markdown(x))
    a=r['v3_audit'];c=r['config'];sc=r['v3_scenarios']
    md(f"# V3 | {c['universe']} | {c['factor']}\n\n**理论多空因子研究，不代表可融券执行策略。** 研究期 RankIC 固定方向：{a['direction_from_development']:+d}；全期含研究期内拟合，验证期及已看过的测试期单列。")
    md('## 0. 1 / 5 / 20交易日调仓对比\n三个策略使用同一研究期方向、因子和成本。首次因子有效后的下一交易日起计，每隔N个交易日调仓；非调仓日股数不变（公司行为除外），每天估值并计提借券费。无法成交的订单不在中途补单，下次计划调仓时按最新前日信号重新决定。成分股中途调出亦等到计划调仓处理。这里是交易日间隔，不是自然周/月。')
    fc=r['frequency_comparison']
    showcols=['CAGR','CAGR_calendar','Sharpe','max_drawdown','daily_two_way_turnover','arithmetic_annual_cost','win_rate','trading_days','official_valid']
    labels={'CAGR':'年化252(%)','CAGR_calendar':'自然年CAGR(%)','Sharpe':'夏普','max_drawdown':'最大回撤(%)',
            'daily_two_way_turnover':'平均日双边换手(%)','arithmetic_annual_cost':'年度算术费用拖累(%)','win_rate':'日胜率(%)','trading_days':'实际交易天数','official_valid':'估值有效'}
    for name in ['baseline_10bp','conservative_one_price','long_only']:
        md('### '+name+'：全期（2010–2024）')
        t=fc.xs((name,'FIXED_2010_2024'))[showcols].copy()
        for key in ['CAGR','CAGR_calendar','max_drawdown','daily_two_way_turnover','arithmetic_annual_cost','win_rate']:t[key]*=100
        display(t.rename(columns=labels).round(3))
    md('多空保守情景：分期比较（以下比例为小数，0.05=5%）')
    display(fc.xs('conservative_one_price')[showcols].round(5))
    navf=pd.DataFrame({f'{n} trading days':r['frequency_ledgers'][('conservative_one_price',n)].nav.where(r['frequency_ledgers'][('conservative_one_price',n)].official_valid) for n in (1,5,20)})
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    navf.plot(ax=axes[0],title='Rebalance interval comparison: net NAV')
    (navf/navf.cummax().clip(lower=1)-1).plot(ax=axes[1],title='Drawdown');plt.tight_layout();plt.show()
    display(r['frequency_yearly'].query("scenario=='conservative_one_price'").set_index(['period','rebalance_days']).drop(columns='scenario').round(3))
    display(r['frequency_audit'])
    md('IC的1/5/10/20日是预测标签期限，与调仓频率不同；夏普等指标仍从每日净值计算。以下原有V3情景、成本敏感性和滚动方向分析仍为1日调仓。5/20日结果只代表当前起始位置，尚未遍历所有调仓偏移。')
    md('## 1. 收益总览与净值\nCAGR：252交易日复利年化；CAGR_calendar：首末日期含首日、365.2425天自然年复利。所有比例为小数（0.05=5%）。固定全期、因子有效起点和统一2019–2024比较期同时保留。')
    columns=['CAGR','CAGR_calendar','Sharpe','max_drawdown','daily_two_way_turnover','valid_days','missing_days']
    display(r['v3_comparison'][columns].round(5))
    md('基准情景逐年结果（买卖各10bp，方向只由研究期确定）')
    display(pnl_table(sc['baseline_10bp']).round(3))
    nav=pd.DataFrame({k:p.nav.where(p.official_valid) for k,p in sc.items() if k in ['baseline_10bp','historical_stamp_assumed_cost','conservative_one_price','long_only','rolling_direction']})
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    nav.plot(ax=axes[0],title='Research NAV: independent cash ledgers')
    (nav/nav.cummax().clip(lower=1)-1).plot(ax=axes[1],title='Drawdown');plt.tight_layout();plt.show()
    md('## 2. IC / RankIC\nIC保持原始因子方向；收益总览采用研究期确定方向。标签为t+1至t+h+1真实VWAP，不用close回填。')
    display(r['ic_summary'].round(5));display(r['yearly_ic'].round(4))
    fig,axes=plt.subplots(1,2,figsize=(14,4))
    r['ic'].RankIC_1d.rolling(63,min_periods=21).mean().plot(ax=axes[0],title='63-day RankIC')
    r['ic'][['IC_1d','RankIC_1d']].cumsum().plot(ax=axes[1],title='Cumulative IC, not NAV');plt.show()
    md('## 3. 成本与成交审计\n历史情景：卖出印花税2023-08-28前10bp、之后5bp；佣金双边各3bp、滑点双边各2bp为可配置研究假设。滑点按成交额扣现金。**过户费历史尚未完整核实，暂未计入；无最低佣金、冲击成本及历史借券费推定。** 基准各10bp保留。')
    md('印花税来源：[2008年起卖方千分之一](https://guangdong.chinatax.gov.cn/siteapps/webpage/gdtax/tjjcx/tjjcx_result.jsp?cid=9221d98a85144686a7f321a4b8294701)；[2023年8月28日起减半](https://shanghai.chinatax.gov.cn/tax/zcfw/zcfgk/yhs/202308/t468451.html)。')
    costcols=[x for x in sc['conservative_one_price'].columns if x.endswith('_cost')]
    display((sc['conservative_one_price'][costcols].mean()*252).rename('arithmetic_annual_drag').to_frame())
    display(r['v3_cost_sensitivity'][['CAGR','CAGR_calendar','Sharpe','max_drawdown']].round(5))
    display(pd.Series({k:v for k,v in a.items() if not k.startswith('scenario_')},name='audit').to_frame())
    display(pd.Series(a['scenario_first_invalid_date'],name='first_unresolved_valuation_date').to_frame())
    events=[dict(scenario=k,date=e[0],symbol=e[1],issue=e[2],adjusted_units=e[3]) for k,values in a['scenario_valuation_events'].items() for e in values]
    if events:
        md('**以下对照情景经过未解决的持仓估值：受影响的完整期间收益及情景收益差均不可用，不据此评价因子。**')
        display(pd.DataFrame(events))
    md('一字价保守情景：high=low时买卖都不成交，不声称识别了全部涨跌停。成交量/额为零、价格不可靠等继续禁成交；持仓冻结，不以close假装成交。VWAP-only仅禁用close替代，其他规则相同；收益差为整个独立情景差，包含后续持仓路径变化。')
    md('## 4. 多头与指数超额\n多头为高分半数股票组内等权、目标100%资金；费用和冻结持仓计入预算，不借现金。相对净值=(多头净值/指数净值)，IR用每日收益差；后复权资产与指数股息口径未确认一致，超额为现有指数口径比较。')
    display(r['v3_excess'][['CAGR','CAGR_calendar','Sharpe','max_drawdown','information_ratio','tracking_error']].round(5))
    md('## 5. 滚动样本外历史重放\n每年仅用此前3年的RankIC确定方向，至少252个有效IC日，剔除在训练截止后才实现的标签；未满足条件时空仓。整个历史已查看，不能称为新的独立测试。跨因子选择仍只许用研究/验证期。')
    display(r['v3_rolling'])
    md('## 6. 数据与研究边界\n行业、市值中性化和MAD沿用因子库。历史ST/交易状态字段极稀疏，不猜测0/1含义，不用未来填充。完整涨跌停价、券源、容量、真实股数整手和最低佣金仍缺失。中天金融0.11元及已核实换股政策沿用，保留敏感性。')
    if len(r['v3_valuation_sensitivity']):
        md('以下估值敏感性采用研究期确定的方向及一字价保守成交/历史印花税情景，不虚构退出成交。')
        display(r['v3_valuation_sensitivity'][['raw_valuation','CAGR','max_drawdown']])
    display(r['coverage'].describe().T.round(4))

def export_report(r,folder):
    p=Path(folder);p.mkdir(parents=True,exist_ok=True)
    for k in ['v3_comparison','v3_excess','v3_rolling','v3_cost_sensitivity','ic','ic_summary','coverage']:
        r[k].to_parquet(p/(k+'.parquet'))
    pd.concat(r['v3_scenarios'],names=['scenario','date']).to_parquet(p/'daily_ledgers.parquet')
    r['v3_valuation_sensitivity'].to_parquet(p/'valuation_sensitivity.parquet')
    for key in ['frequency_comparison','frequency_yearly','frequency_audit']:r[key].to_parquet(p/(key+'.parquet'))
    pd.concat(r['frequency_ledgers'],names=['scenario','rebalance_days','date']).to_parquet(p/'frequency_ledgers.parquet')
    (p/'audit.json').write_text(json.dumps(r['v3_audit'],ensure_ascii=False,indent=2,default=str),encoding='utf-8')
