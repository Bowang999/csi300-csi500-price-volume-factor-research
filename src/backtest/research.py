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
    md(f"# V3 | {c['universe']} | {c['factor']}\n\n**Theoretical long-short factor research, not an executable stock-loan strategy.** Development RankIC direction: {a['direction_from_development']:+d}; full history includes development fitting. Validation and the previously viewed test period are reported separately.")
    md('## 0. Rebalancing every 1 / 5 / 20 trading sessions\nAll strategies share the development-period direction, factor and costs. Rebalance every N sessions starting the session after the first valid signal. Shares remain unchanged between rebalances except corporate actions; valuation and borrowing fees accrue daily. Unfilled orders are not retried between scheduled rebalances; the next rebalance uses the latest previous-close signal. Index deletions also wait until that rebalance. These are trading-session intervals, not calendar weeks or months.')
    fc=r['frequency_comparison']
    showcols=['CAGR','CAGR_calendar','Sharpe','max_drawdown','daily_two_way_turnover','arithmetic_annual_cost','win_rate','trading_days','official_valid']
    labels={'CAGR':'Annualized 252 (%)','CAGR_calendar':'Calendar CAGR (%)','Sharpe':'Sharpe','max_drawdown':'Max drawdown (%)',
            'daily_two_way_turnover':'Mean daily two-way turnover (%)','arithmetic_annual_cost':'Annual arithmetic cost drag (%)','win_rate':'Daily win rate (%)','trading_days':'Days with trades','official_valid':'Valuation valid'}
    for name in ['baseline_10bp','conservative_one_price','long_only']:
        md('### '+name+': full period (2010–2024)')
        t=fc.xs((name,'FIXED_2010_2024'))[showcols].copy()
        for key in ['CAGR','CAGR_calendar','max_drawdown','daily_two_way_turnover','arithmetic_annual_cost','win_rate']:t[key]*=100
        display(t.rename(columns=labels).round(3))
    md('Conservative long-short scenario by period (ratios are decimals: 0.05 = 5%)')
    display(fc.xs('conservative_one_price')[showcols].round(5))
    navf=pd.DataFrame({f'{n} trading days':r['frequency_ledgers'][('conservative_one_price',n)].nav.where(r['frequency_ledgers'][('conservative_one_price',n)].official_valid) for n in (1,5,20)})
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    navf.plot(ax=axes[0],title='Rebalance interval comparison: net NAV')
    (navf/navf.cummax().clip(lower=1)-1).plot(ax=axes[1],title='Drawdown');plt.tight_layout();plt.show()
    display(r['frequency_yearly'].query("scenario=='conservative_one_price'").set_index(['period','rebalance_days']).drop(columns='scenario').round(3))
    display(r['frequency_audit'])
    md('IC horizons of 1/5/10/20 sessions describe prediction labels, not rebalancing frequency. Sharpe and other performance metrics use daily NAV. The following original V3 scenarios, cost sensitivities and rolling direction analysis use daily rebalancing. Results at 5/20 sessions use the current starting offset, not all possible offsets.')
    md('## 1. Performance and NAV\nCAGR uses 252-session compound annualization; CAGR_calendar uses inclusive first-to-last calendar dates and 365.2425 days per year. Ratios are decimals (0.05 = 5%). Fixed history, factor-valid-start and the common 2019–2024 window are retained.')
    columns=['CAGR','CAGR_calendar','Sharpe','max_drawdown','daily_two_way_turnover','valid_days','missing_days']
    display(r['v3_comparison'][columns].round(5))
    md('Annual baseline results (10 bp on each buy/sell; direction determined only in development)')
    display(pnl_table(sc['baseline_10bp']).round(3))
    nav=pd.DataFrame({k:p.nav.where(p.official_valid) for k,p in sc.items() if k in ['baseline_10bp','historical_stamp_assumed_cost','conservative_one_price','long_only','rolling_direction']})
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    nav.plot(ax=axes[0],title='Research NAV: independent cash ledgers')
    (nav/nav.cummax().clip(lower=1)-1).plot(ax=axes[1],title='Drawdown');plt.tight_layout();plt.show()
    md('## 2. IC / RankIC\nIC retains the original factor direction; performance uses the development-period direction. Labels use actual VWAP from t+1 to t+h+1, with no close fallback.')
    display(r['ic_summary'].round(5));display(r['yearly_ic'].round(4))
    fig,axes=plt.subplots(1,2,figsize=(14,4))
    r['ic'].RankIC_1d.rolling(63,min_periods=21).mean().plot(ax=axes[0],title='63-day RankIC')
    r['ic'][['IC_1d','RankIC_1d']].cumsum().plot(ax=axes[1],title='Cumulative IC, not NAV');plt.show()
    md('## 3. Cost and execution audit\nHistorical scenario: sell-side stamp tax of 10 bp before 2023-08-28 and 5 bp thereafter; commission of 3 bp and slippage of 2 bp on each side are configurable research assumptions. Slippage reduces cash according to traded notional. **Historical transfer fees are not fully verified and are omitted, as are minimum commission, market impact and inferred historical borrow fees.** The 10 bp per-side baseline is retained.')
    md('Stamp-tax sources: [sell-side 10 bp from 2008](https://guangdong.chinatax.gov.cn/siteapps/webpage/gdtax/tjjcx/tjjcx_result.jsp?cid=9221d98a85144686a7f321a4b8294701); [halved from 2023-08-28](https://shanghai.chinatax.gov.cn/tax/zcfw/zcfgk/yhs/202308/t468451.html).')
    costcols=[x for x in sc['conservative_one_price'].columns if x.endswith('_cost')]
    display((sc['conservative_one_price'][costcols].mean()*252).rename('arithmetic_annual_drag').to_frame())
    display(r['v3_cost_sensitivity'][['CAGR','CAGR_calendar','Sharpe','max_drawdown']].round(5))
    display(pd.Series({k:v for k,v in a.items() if not k.startswith('scenario_')},name='audit').to_frame())
    display(pd.Series(a['scenario_first_invalid_date'],name='first_unresolved_valuation_date').to_frame())
    events=[dict(scenario=k,date=e[0],symbol=e[1],issue=e[2],adjusted_units=e[3]) for k,values in a['scenario_valuation_events'].items() for e in values]
    if events:
        md('**The following comparison scenarios contain unresolved holding valuations. Affected full-period returns and scenario return differences are unavailable and must not be used to evaluate factors.**')
        display(pd.DataFrame(events))
    md('Conservative single-price scenario: neither buys nor sells execute when high equals low; this does not identify all price limits. Zero volume/notional and unreliable prices continue to block execution. Holdings freeze; close prices do not create fictitious trades. VWAP-only disables close fallback with other rules unchanged. Return differences compare independently simulated paths, including subsequent holding changes.')
    md('## 4. Long-only and index-relative performance\nLong-only holds the upper-scoring half equally, targeting 100% capital. Costs and frozen holdings consume the budget; no cash borrowing. Relative NAV = long-only NAV / index NAV; information ratio uses daily return differences. Dividend conventions between adjusted assets and the index are not confirmed identical: this is a comparison against the supplied index series.')
    display(r['v3_excess'][['CAGR','CAGR_calendar','Sharpe','max_drawdown','information_ratio','tracking_error']].round(5))
    md('## 5. Rolling out-of-sample historical replay\nEach year uses only the preceding three years of RankIC to choose direction, requiring 252 valid IC days and excluding labels realized after the training cutoff. Otherwise stay in cash. The full history has been viewed and is not a new independent test. Cross-factor selection must still use development/validation only.')
    display(r['v3_rolling'])
    md('## 6. Data and research limitations\nIndustry/size neutralization and MAD follow the factor library. Historical ST/trading-state fields are sparse: do not guess binary semantics or fill from future records. Complete price limits, stock loans, capacity, actual-share board lots and minimum commissions remain unavailable. Retain the CNY 0.11 Zhongtian valuation and verified share-exchange policies with sensitivity analysis.')
    if len(r['v3_valuation_sensitivity']):
        md('The following valuation sensitivity uses development-period direction, conservative single-price execution and historical stamp taxes; no fictitious exit trades.')
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
