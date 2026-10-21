"""Daily factor research. Signals at t close, fills at t+1 VWAP, returns thereafter.

Research portfolios assume fills; missing ST/limit/order-book history prevents a
claim of fully executable A-share returns. No future returns enter target weights.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from .vwap_execution import vwap_prices, signed_portfolio
from .daily_execution import load_market, half_weights, simulate, full_metrics, pnl_table


@dataclass
class Config:
    universe: str = 'csi300'
    factor: str = 'PV_MOM_20'
    stage: str = 'processed'
    start: str = '2010-01-04'
    end: str = '2024-12-31'
    quantiles: int = 5
    horizons: tuple = (1, 5, 10, 20)
    direction: int = 1
    buy_cost_bps: float = 10.0
    sell_cost_bps: float = 10.0
    borrow_cost_bps_annual: float = 0.0
    execution: str = "auto"
    min_stocks: int = 30
    annual_days: int = 252
    risk_free_annual: float = 0.0
    valuation_policy: str = 'published_fund'
    terminal_raw_mark: float = .11


def find_root(start=None):
    for p in [Path(start or Path.cwd()).resolve(), *Path(start or Path.cwd()).resolve().parents]:
        if (p/'factor_store/_factor_catalog.json').is_file() and (p/'data').is_dir():
            return p
    raise FileNotFoundError('Set ROOT to the folder containing data and factor_store.')


def forward_returns(daily_returns, horizon):
    # At signal t, buy at VWAP t+1; first held return is the one ending t+2.
    valid=daily_returns.where(np.isfinite(daily_returns)&daily_returns.gt(-1))
    return np.expm1(np.log1p(valid).rolling(horizon,min_periods=horizon).sum().shift(-(horizon+1)))


def quantile_groups(factors, groups=5, min_stocks=30):
    if groups!=5:
        out=pd.DataFrame(np.nan,index=factors.index,columns=factors.columns)
        for day,row in factors.iterrows():
            x=row.dropna()
            if len(x)<max(min_stocks,groups*2) or x.nunique()<groups:continue
            bins=pd.qcut(x,groups,labels=False,duplicates='drop')
            if bins.nunique()==groups:out.loc[day,x.index]=bins.to_numpy()+1
        return out
    values=factors.to_numpy();out=np.full(values.shape,np.nan)
    quantiles=np.linspace(0,1,groups+1)
    np.putmask(quantiles,groups*quantiles!=np.arange(groups+1),np.nextafter(quantiles,1))
    for i,row in enumerate(values):
        valid=np.isfinite(row);x=row[valid]
        if len(x)<max(min_stocks,groups*2) or len(np.unique(x))<groups:continue
        edges=np.unique(np.quantile(x,quantiles,method='linear'))
        if len(edges)!=groups+1:continue
        bins=np.maximum(1,np.searchsorted(edges,x,side='left'))
        if len(np.unique(bins))==groups:out[i,valid]=bins
    return pd.DataFrame(out,index=factors.index,columns=factors.columns)


def row_corr(x,y,minimum=30,rank=False):
    ok=x.notna()&y.notna();a=x.where(ok);b=y.where(ok)
    if rank:a=a.rank(axis=1,method='average');b=b.rank(axis=1,method='average')
    return a.corrwith(b,axis=1).where(ok.sum(axis=1)>=minimum)


def hac_mean_stats(series,lags=5):
    # Preserve calendar gaps: covariance pairs use original trading-day positions.
    a=series.to_numpy(float);n=np.isfinite(a).sum()
    if n<3:return np.nan,np.nan
    mean=np.nanmean(a);u=a-mean
    variance=np.nansum(u*u)/n
    for lag in range(1,min(lags,len(a)-1)+1):
        variance+=2*(1-lag/(lags+1))*np.nansum(u[lag:]*u[:-lag])/n
    se=np.sqrt(max(variance,0)/n)
    return se,mean/se if se>0 else np.nan


def ic_summary(ic,horizons,annual_days=252):
    rows=[]
    for kind in ('IC','RankIC'):
        for h in horizons:
            s=ic[f'{kind}_{h}d'];x=s.dropna();sd=x.std(ddof=1)
            se,t=hac_mean_stats(s,max(5,h-1))
            rows.append({'metric':kind,'horizon':h,'valid_days':len(x),'mean':x.mean(),'std':sd,
                         'IR_unannualized':x.mean()/sd if sd>0 else np.nan,
                         'IR_sqrt252':x.mean()/sd*np.sqrt(annual_days) if sd>0 else np.nan,
                         'positive_fraction':x.gt(0).mean(),'mean_abs':x.abs().mean(),
                         'HAC_standard_error':se,'HAC_t':t,
                         'HAC_p_approx_normal':2*norm.sf(abs(t)) if np.isfinite(t) else np.nan,
                         'HAC_CI95_low':x.mean()-1.96*se,'HAC_CI95_high':x.mean()+1.96*se})
    return pd.DataFrame(rows).set_index(['metric','horizon'])


def group_portfolio(weights,returns,buy_bps=10,sell_bps=10):
    """Fixed signal weights; incomplete held labels invalidate a day, never reweight."""
    active=weights.sum(axis=1).gt(0)
    complete=((weights.eq(0))|returns.notna()).all(axis=1)
    gross=(weights*returns.fillna(0)).sum(axis=1).where(complete)
    # Pre-trade weights include price drift from yesterday's holding interval.
    drift=weights.mul(1+returns).div(1+gross,axis=0).where(weights.ne(0),0)
    prev=drift.shift(1);prev_active=active.shift(1,fill_value=False)
    prev.loc[~prev_active]=0
    dw=weights-prev
    valid_trade=(gross.shift(1).notna()|~prev_active)
    buys=dw.clip(lower=0).sum(axis=1).where(valid_trade)
    sells=(-dw.clip(upper=0)).sum(axis=1).where(valid_trade)
    # Close all positions at the final requested holding interval's exit.
    if len(active) and active.iloc[-1]:
        sells.iloc[-1]=sells.iloc[-1]+1
    cost=buys*buy_bps/10000+sells*sell_bps/10000
    net=gross-cost
    return pd.DataFrame({'gross':gross,'net':net,'buys':buys,'sells':sells,
                         'one_way_turnover':(buys+sells)/2,'cost':cost,
                         'selected_stocks':weights.gt(0).sum(axis=1),
                         'missing_label_stocks':(weights.gt(0)&returns.isna()).sum(axis=1),
                         'complete_labels':complete})


def longest_valid_segment(s):
    valid=s.notna();blocks=(valid!=valid.shift(fill_value=False)).cumsum()
    candidates=[g for _,g in s[valid].groupby(blocks[valid])]
    return max(candidates,key=len) if candidates else s.iloc[:0]


def segment_nav(s):
    """No returns filled with zero; each contiguous observed segment restarts at 1."""
    out=pd.Series(np.nan,index=s.index);valid=s.notna();blocks=(valid!=valid.shift(fill_value=False)).cumsum()
    for _,g in s[valid].groupby(blocks[valid]):out.loc[g.index]=(1+g).cumprod()
    return out


def performance(s,benchmark=None,annual=252,rf=0):
    result=full_metrics(s,annual,rf)
    # Compatibility field names now explicitly refer to the entire requested interval.
    result.update(segment_CAGR=result['CAGR'],segment_max_drawdown=result['max_drawdown'],segment_Calmar=result['Calmar'])
    return result


def run_backtest(config,root=None):
    c=config;root=Path(root) if root else find_root()
    if c.universe=='csi500' and c.execution=='next_day_close':
        import factor_backtest_close as legacy
        values={k:v for k,v in asdict(c).items() if k in legacy.Config.__dataclass_fields__}
        result=legacy.run_backtest(legacy.Config(**values),root)
        result['config']['execution']='next_day_close'
        return result
    if c.execution=='auto':
        from dataclasses import replace
        c=replace(c,execution='next_day_vwap_close_fallback')
    assert c.universe in ('csi300','csi500') and c.stage in ('raw','processed')
    assert c.execution=='next_day_vwap_close_fallback' and c.borrow_cost_bps_annual>=0
    assert c.direction in (-1,1) and 1 in c.horizons and min(c.horizons)>0
    assert c.quantiles>=2 and c.buy_cost_bps>=0 and c.sell_cost_bps>=0 and c.start<=c.end
    catalog=json.loads((root/'factor_store/_factor_catalog.json').read_text(encoding='utf-8'))
    definition=next(x for x in catalog if x['name']==c.factor)
    data=root/'data'/c.universe/'parquet'
    index=pd.read_parquet(data/f'{c.universe}_index_returns_2010_2024.parquet')
    index.date=pd.to_datetime(index.date);index=index.set_index('date').sort_index()['index_ret']
    calendar=index.index
    cols=['date','symbol',c.factor,'industry_code','industry_scheme','industry_known','market_cap','neutralization_eligible']
    f=pd.read_parquet(root/'factor_store'/c.stage,columns=cols,
                      filters=[('universe','=',c.universe),('year','>=',int(c.start[:4])),('year','<=',int(c.end[:4]))])
    f.date=pd.to_datetime(f.date);f=f.loc[f.date.between(c.start,c.end)].sort_values(['date','symbol'])
    assert not f.duplicated(['date','symbol']).any()
    if f.empty:raise ValueError('No observations in requested date range')
    assert f.groupby('date').size().eq(300 if c.universe=='csi300' else 500).all()
    signal_dates=calendar[(calendar>=c.start)&(calendar<=c.end)]
    # Do not bridge missing trading sessions through an observed-row shift.
    last_pos=min(len(calendar)-1,calendar.searchsorted(signal_dates[-1])+max(c.horizons)+1)
    market=pd.read_parquet(data/f'{c.universe}_daily_market_2010_2024.parquet',
        columns=['date','symbol','vwap_hfq','vwap_source'],
        filters=[('date','>=',signal_dates[0].date()),('date','<=',calendar[last_pos].date())])
    market.date=pd.to_datetime(market.date)
    execution_audit=market.vwap_source.value_counts().to_dict()
    market_model=load_market(str(root),c.universe,c.start,c.end,c.valuation_policy,c.terminal_raw_mark)
    symbols=sorted(set(f.symbol.unique())|{a['new'] for a in market_model['actions']})
    prices=market.pivot(index='date',columns='symbol',values='vwap_hfq').reindex(index=calendar,columns=symbols)
    daily=prices.div(prices.shift(1))-1
    del market,prices
    factors=f.pivot(index='date',columns='symbol',values=c.factor).reindex(index=signal_dates,columns=symbols)*c.direction
    factors=factors.where(np.isfinite(factors))
    labels={h:forward_returns(daily,h).reindex(signal_dates) for h in c.horizons}
    del daily
    groups=quantile_groups(factors,c.quantiles,c.min_stocks)
    ic=pd.DataFrame(index=signal_dates)
    for h,y in labels.items():
        ic[f'IC_{h}d']=row_corr(factors,y,c.min_stocks)
        ic[f'RankIC_{h}d']=row_corr(factors,y,c.min_stocks,rank=True)
    coverage=pd.DataFrame({'members':f.groupby('date').size(),'factor_available':factors.notna().sum(axis=1),
                          'industry_missing':f.groupby('date').industry_known.apply(lambda s:(~s).sum()),
                          'neutralization_eligible':f.groupby('date').neutralization_eligible.sum(),
                          'valid_group_assignment':groups.notna().sum(axis=1),
                          'valid_factor_and_1d_label':(factors.notna()&labels[1].notna()).sum(axis=1)})
    coverage['factor_coverage']=coverage.factor_available/coverage.members
    coverage['rank_autocorrelation']=row_corr(factors,factors.shift(),c.min_stocks,rank=True)
    # Portfolio index is EXIT date, not signal date. Exclude unobservable final exits.
    exit_map=pd.Series(calendar,index=calendar).shift(-2).reindex(signal_dates)
    realized=exit_map.notna();ports={};weights={};group_forward=[]
    sleeve_weights={}
    for q in range(1,c.quantiles+1):
        mask=groups.eq(q);w=mask.div(mask.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
        held=w.loc[realized].copy();ret=labels[1].loc[realized].copy()
        held.index=pd.DatetimeIndex(exit_map[realized]);ret.index=held.index
        if q in (1,c.quantiles):sleeve_weights[q]=held
        ports[q]=group_portfolio(held,ret,c.buy_cost_bps,c.sell_cost_bps)
        for h,y in labels.items():
            day=y.where(mask).mean(axis=1)
            group_forward.append({'quantile':q,'horizon':h,'mean_forward_return':day.mean(),
                                  'valid_days':int(day.notna().sum()),'label_observations':int((mask&y.notna()).sum().sum())})
    gross=pd.DataFrame({f'Q{q}':p.gross for q,p in ports.items()})
    net=pd.DataFrame({f'Q{q}':p.net for q,p in ports.items()})
    benchmark=index.reindex(gross.index)
    perf={}
    for q in ports:
        perf[f'Q{q}_gross']=performance(gross[f'Q{q}'],benchmark,c.annual_days,c.risk_free_annual)
        perf[f'Q{q}_net']=performance(net[f'Q{q}'],benchmark,c.annual_days,c.risk_free_annual)
    perf['index']=performance(benchmark,annual=c.annual_days,rf=c.risk_free_annual)
    high=ports[c.quantiles];low=ports[1]
    target=half_weights(factors,c.min_stocks)
    ls=simulate(target,market_model,c.buy_cost_bps,c.sell_cost_bps,c.borrow_cost_bps_annual,c.annual_days)
    reverse_ls=simulate(-target,market_model,c.buy_cost_bps,c.sell_cost_bps,c.borrow_cost_bps_annual,c.annual_days)
    costfree=simulate(target,market_model,0,0,0,c.annual_days)
    ls['gross']=costfree.net
    decomposition=[]
    for name,series in [('same_holdings_before_cost',ls.gross_same_holdings),('transaction_cost',-ls.cost),('borrow_cost',-ls.borrow_cost),('net',ls.net),('independent_zero_cost',costfree.net)]:
        m=full_metrics(series,c.annual_days,c.risk_free_annual)
        decomposition.append(dict(series=name,mean_daily_bp=series.mean()*10000,std_daily_bp=series.std(ddof=1)*10000,arithmetic_annual_pct=series.mean()*c.annual_days*100,annual_vol_pct=m['annual_vol']*100,Sharpe=m['Sharpe'] if name not in ['transaction_cost','borrow_cost'] else np.nan))
    assert np.allclose(ls.net,ls.gross_same_holdings-ls.cost-ls.borrow_cost,atol=1e-10)
    valuation_sensitivity=[]
    if c.valuation_policy=='published_fund' and '000540' in target.columns and ls.estimated_mark_exposure.gt(0).any():
        for raw_mark in [0.,.01,.11,.14]:
            scenario_market=dict(market_model)
            if raw_mark!=c.terminal_raw_mark:
                marks=market_model['mark'].copy();selection=market_model['estimated_mark']['000540'].fillna(False).astype(bool)
                if pd.Timestamp('2023-05-18') in marks.index:
                    anchor=marks.loc[pd.Timestamp('2023-05-18'),'000540']/.40
                else:
                    anchor_data=pd.read_parquet(root/'data'/c.universe/'parquet'/f'{c.universe}_daily_market_2010_2024.parquet',columns=['date','symbol','close_hfq'],filters=[('symbol','==','000540')])
                    anchor=float(anchor_data.loc[pd.to_datetime(anchor_data.date).eq('2023-05-18'),'close_hfq'].iloc[0])/.40
                marks.loc[selection,'000540']=raw_mark*anchor;scenario_market['mark']=marks
            scenario=ls if raw_mark==c.terminal_raw_mark else simulate(target,scenario_market,c.buy_cost_bps,c.sell_cost_bps,c.borrow_cost_bps_annual,c.annual_days)
            stats=full_metrics(scenario.net.where(scenario.official_valid),c.annual_days,c.risk_free_annual)
            valuation_sensitivity.append(dict(raw_valuation=raw_mark,**stats))
    execution_audit.update(close_fallback_executions=int(ls.fallback_count.sum()),
        close_fallback_turnover_fraction=float(ls.fallback_notional.sum()/ls.two_way_turnover.sum()) if ls.two_way_turnover.sum()>0 else 0.,
        unresolved_held_price_events=int(ls.valuation_issue_count.sum()),official_full_period_valid=bool(ls.official_valid.all()))
    spreads=pd.DataFrame({'high_minus_low':high.gross-low.gross,
        'half_long_half_short_gross':ls.gross,'half_long_half_short_net':ls.net})
    spread_performance=pd.DataFrame({name:performance(x,annual=c.annual_days,rf=c.risk_free_annual) for name,x in spreads.items()}).T
    perf['LS_gross']=performance(ls.gross.where(ls.official_valid),annual=c.annual_days,rf=c.risk_free_annual)
    perf['LS_net']=performance(ls.net.where(ls.official_valid),annual=c.annual_days,rf=c.risk_free_annual)
    perf['reverse_LS_net']=performance(reverse_ls.net.where(reverse_ls.official_valid),annual=c.annual_days,rf=c.risk_free_annual)
    monthly=ic.resample('ME').mean();yearly=ic.resample('YE').mean()
    yearly.index=yearly.index.year
    annual_rows=[]
    for year,s in ls.net.where(ls.official_valid).groupby(ls.net.index.year):
        v=performance(s,annual=c.annual_days,rf=c.risk_free_annual);v['year']=year;annual_rows.append(v)
    sensitivity=[]
    for bps in [0,5,10,20,30]:
        scenario=simulate(target,market_model,bps,bps,c.borrow_cost_bps_annual,c.annual_days)
        s=scenario.net.where(scenario.official_valid)
        row=performance(s,annual=c.annual_days,rf=c.risk_free_annual);row['cost_bps_each_side']=bps
        row.update({'indicative_'+k:v for k,v in full_metrics(scenario.net,c.annual_days,c.risk_free_annual).items()})
        sensitivity.append(row)
    periods=[]
    for label,start,end in [('development','2010-01-01','2018-12-31'),('validation','2019-01-01','2021-12-31'),('test','2022-01-01','2024-12-31')]:
        v=performance(ls.net.where(ls.official_valid).loc[start:end],annual=c.annual_days,rf=c.risk_free_annual)
        v.update(period=label,RankIC_1d=ic.loc[start:end,'RankIC_1d'].mean());periods.append(v)
    # Align only member rows; avoid materializing all non-member stock-date pairs.
    detail=f[['date','symbol','industry_scheme','industry_code',c.factor]].copy()
    ti=labels[1].index.get_indexer(detail.date);si=labels[1].columns.get_indexer(detail.symbol)
    detail['forward_1d']=labels[1].to_numpy()[ti,si]
    detail[c.factor]*=c.direction
    group_keys=['industry_scheme','industry_code','date']
    valid=detail.dropna(subset=[c.factor,'forward_1d','industry_code']).copy()
    ranks=valid.groupby(group_keys)[[c.factor,'forward_1d']].rank(method='average')
    means=ranks.groupby([valid[k] for k in group_keys]).transform('mean')
    centered=ranks-means
    valid['_xy']=centered[c.factor]*centered.forward_1d
    valid['_xx']=centered[c.factor]**2;valid['_yy']=centered.forward_1d**2
    stats=valid.groupby(group_keys).agg(n=('_xy','count'),xy=('_xy','sum'),xx=('_xx','sum'),yy=('_yy','sum'))
    stats['ic']=(stats.xy/np.sqrt(stats.xx*stats.yy)).where((stats.n>=8)&(stats.xx>0)&(stats.yy>0))
    summary=stats.groupby(level=[0,1]).ic.agg(['count','mean'])
    universe_industries=detail.dropna(subset=['industry_code'])[['industry_scheme','industry_code']].drop_duplicates()
    industry=[]
    for scheme,code in universe_industries.itertuples(index=False,name=None):
        a=summary.loc[(scheme,code)] if (scheme,code) in summary.index else None
        industry.append({'scheme':scheme,'industry':code,'valid_days':int(a['count']) if a is not None else 0,'mean_RankIC':float(a['mean']) if a is not None else np.nan})
    industry.sort(key=lambda x:(x['scheme'],x['industry']))
    del detail,valid,ranks,means,centered,stats
    distribution=f.groupby('date')[c.factor].agg(['count','mean','std','min','median','max'])
    cap=f.pivot(index='date',columns='symbol',values='market_cap').reindex_like(factors)
    coverage['factor_logcap_correlation']=row_corr(factors,np.log(cap.where(cap>0)),c.min_stocks)
    group_means=gross.mean();monotonic=group_means.corr(pd.Series(range(1,c.quantiles+1),index=group_means.index),method='spearman')
    return dict(_factors=factors,_market=market_model,_index=index,config=asdict(c),definition=definition,sharpe_decomposition=pd.DataFrame(decomposition).set_index('series'),valuation_sensitivity=pd.DataFrame(valuation_sensitivity),pnl_table=pnl_table(ls,c.annual_days,c.risk_free_annual),ls=ls,reverse_ls=reverse_ls,execution_audit=execution_audit,coverage=coverage,distribution=distribution,
                ic=ic,ic_summary=ic_summary(ic,c.horizons,c.annual_days),monthly_ic=monthly,yearly_ic=yearly,
                group_forward=pd.DataFrame(group_forward),gross=gross,net=net,benchmark=benchmark,
                portfolios=ports,performance=pd.DataFrame(perf).T,spreads=spreads,
                annual=pd.DataFrame(annual_rows).set_index('year'),cost_sensitivity=pd.DataFrame(sensitivity).set_index('cost_bps_each_side'),
                periods=pd.DataFrame(periods).set_index('period'),industry_ic=pd.DataFrame(industry),
                group_monotonicity=monotonic,spread_performance=spread_performance)


def show_report(r):
    from IPython.display import display,Markdown
    c=r['config'];p=r['ls'];valid=bool(p.official_valid.all())
    def heading(text):display(Markdown(text))
    def heatmap(series,title,percent=False):
        mat=series.groupby([series.index.year,series.index.month]).mean().unstack()
        if mat.empty:return
        limit=max(.001,float(np.nanmax(np.abs(mat.to_numpy())))) if mat.notna().any().any() else .01
        fig,ax=plt.subplots(figsize=(12,4.5));im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_xticks(range(len(mat.columns)),mat.columns);ax.set_yticks(range(len(mat.index)),mat.index);ax.set_title(title)
        for i in range(len(mat)):
            for j in range(len(mat.columns)):
                v=mat.iloc[i,j]
                if pd.notna(v):ax.text(j,i,f'{v*100:.1f}' if percent else f'{v:.2f}',ha='center',va='center',fontsize=7)
        fig.colorbar(im,ax=ax);fig.tight_layout();plt.show()
    heading(f"# {c['universe'].upper()} | {c['factor']}\n\n**Daily half-universe long/short**: top 50% of valid names long, bottom 50% short; each side 50% equity, equal weights. Signal at t close, execution t+1 VWAP with eligible close fallback; close-to-close portfolio accounting. Direction {c['direction']:+d}; buy/sell {c['buy_cost_bps']:g}/{c['sell_cost_bps']:g} bp.")
    heading('## 1. PnL overview')
    heading('Research valuation: consistently backward-adjusted prices; verified share exchanges convert holdings proportionally without turnover or fees. Zhongtian Financial is valued from 2023-06-12 at the adjusted equivalent of CNY 0.11 unadjusted. Holdings remain frozen; no sale is assumed. Sensitivity results follow. This is a research backtest under disclosed assumptions.')
    if not valid:
        heading('**Data quality: actual positions cross unresolved price, adjustment or delisting boundaries. Formal performance after the affected point is unavailable. Continuous curves below are provisional close-marked, carried-value scenarios, not verified full-period returns.**')
    display(r['pnl_table'].style.format(precision=2,na_rep='N/A').set_properties(subset=None,**{'font-size':'11px'}))
    if not valid:
        heading('### Provisional PnL (not formal performance)\nDiagnostic only: existing backward-adjusted closes and carried valuations when prices are missing. Delisting consideration and adjustment conflicts remain unresolved. Do not use these results to select winning factors or claim executable full-period returns.')
        display(pnl_table(p,c['annual_days'],c['risk_free_annual'],indicative=True).style.format(precision=2,na_rep='N/A'))
        heading('Provisional direction comparison (not used to select direction automatically)')
        display(pd.DataFrame({name:full_metrics(s,c['annual_days'],c['risk_free_annual']) for name,s in {'gross':p.gross,'net':p.net,'reverse_net':r['reverse_ls'].net}.items()}).T[['CAGR','Sharpe','max_drawdown','annual_vol','Calmar']])
    display(r['performance'].loc[['LS_gross','LS_net','reverse_LS_net'],['CAGR','Sharpe','max_drawdown','annual_vol','Sortino','Calmar','total_return','valid_days','missing_days']].style.format(precision=4,na_rep='N/A'))
    heading('Turnover = (buy notional + sell notional) / prior-close equity, no division by two. CAGR uses 252 trading days/year. Initial entry costs included; final positions marked at close, not forcibly liquidated. Gross is a separately simulated zero-cost ledger. Cash and short proceeds earn zero; borrowing availability assumed, annual borrow fee configurable (default zero).')
    nav=pd.DataFrame({'gross':(1+p.gross).cumprod(),'net':p.nav})
    fig,ax=plt.subplots(3,1,figsize=(14,10),sharex=True,constrained_layout=True)
    nav.plot(ax=ax[0],title='Full-period NAV (initial equity = 1)'+(' — INDICATIVE / UNRESOLVED' if not valid else ''))
    (nav/nav.cummax().clip(lower=1)-1).plot(ax=ax[1],title='Drawdown')
    p.two_way_turnover.plot(ax=ax[2],alpha=.2,lw=.5,label='daily')
    p.two_way_turnover.rolling(21,min_periods=1).mean().plot(ax=ax[2],label='21-day mean',title='Two-way turnover');ax[2].legend();plt.show()
    heatmap(p.net.resample('ME').apply(lambda s:(1+s).prod()-1),'Monthly net return (%)'+(' — INDICATIVE' if not valid else ''),True)
    heading('## 2. IC / RankIC | Predictive ability\nLabels remain validated VWAP t+1 to t+h+1, with no close fallback or stale-price filling. They differ from portfolio close-marked returns. HAC accounts for serial dependence; overlapping horizons are diagnostics. Cumulative IC is not NAV.')
    display(r['ic_summary'].round(5));display(r['yearly_ic'].round(4))
    fig,ax=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    r['ic'].RankIC_1d.plot(ax=ax[0,0],alpha=.2,lw=.5)
    r['ic'].RankIC_1d.rolling(63,min_periods=21).mean().plot(ax=ax[0,0],title='Rank IC / 63-day mean')
    r['ic'][['IC_1d','RankIC_1d']].cumsum().plot(ax=ax[0,1],title='Cumulative IC (not NAV)')
    r['ic'].RankIC_1d.hist(ax=ax[1,0],bins=50);ax[1,0].set_title('Rank IC distribution')
    r['ic_summary']['mean'].unstack(0).plot.bar(ax=ax[1,1],title='IC by prediction horizon');plt.show()
    heatmap(r['monthly_ic'].RankIC_1d,'Monthly mean Rank IC')
    display(r['industry_ic'].round(4))
    heading('## 3. Attribution / trading | Returns and costs\nQuintiles are auxiliary forward-return diagnostics, not the primary portfolio. Missing labels are excluded from those means. Long/short attribution below is gross arithmetic daily contribution; transaction costs are separate.')
    heading('Sharpe check: mean daily excess return / sample standard deviation of daily excess returns times sqrt(252), not annual return divided by daily volatility. Cost contributions are return drags: gross returns of the same positions minus trading and borrowing costs equal net returns each day. The independent zero-cost strategy is simulated separately.')
    display(r['sharpe_decomposition'].round(4))
    if len(r['valuation_sensitivity']):
        heading('Zhongtian Financial valuation sensitivity: only frozen-asset valuation changes; no fictitious trade. CNY 0.11 is the main assumption.')
        display(r['valuation_sensitivity'].set_index('raw_valuation')[['CAGR','Sharpe','max_drawdown','total_return']].round(5))
    display(r['group_forward'].round(5));print('Quintile monotonicity:',r['group_monotonicity'])
    display(r['cost_sensitivity'][['CAGR','Sharpe','max_drawdown','annual_vol','missing_days']].round(4))
    if not valid:
        heading('Provisional cost sensitivity (not formal performance; holdings and NAV recomputed independently for each fee rate)')
        display(r['cost_sensitivity'][['indicative_CAGR','indicative_Sharpe','indicative_max_drawdown','indicative_annual_vol']].round(4))
    fig,ax=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    r['group_forward'].query('horizon==1').set_index('quantile').mean_forward_return.plot.bar(ax=ax[0,0],title='Quintile mean next-day forward return')
    p[['long_pnl','short_pnl']].cumsum().plot(ax=ax[0,1],title='Cumulative arithmetic PnL contributions')
    p[['long_exposure','short_exposure']].plot(ax=ax[1,0],title='Actual long / short exposure')
    r['coverage'][['factor_logcap_correlation','rank_autocorrelation']].plot(ax=ax[1,1],title='Factor size correlation / rank persistence');plt.show()
    display(p[['long_count','short_count','two_way_turnover','cost','cash_weight','frozen_exposure','blocked_count']].describe().T.round(4))
    heading('## 4. Data / methods\nNo new orders on nontrading or quarantined rows. Frozen positions persist, reduce the available budget on their side, and may cause target-weight deviations. No future availability filter. Close fallback is a historical-data approximation, not a guaranteed executable order. No complete limit-up/down or stock-loan availability model. Adjusted prices represent analytical units, not actual share orders. A valuation conflict permanently invalidates certified NAV until resolved; it is not silently filled with zero.')
    display(pd.Series(r['execution_audit'],name='execution_audit').to_frame())
    display(p[['fallback_count','fallback_notional','frozen_count','max_frozen_days','valuation_issue_count']].describe().T.round(4))
    events=p.attrs.get('valuation_events',[])
    if events:display(pd.DataFrame(events,columns=['date','symbol','issue','adjusted_units']).head(30))
    display(r['coverage'].describe().T.round(4));display(r['distribution'].describe().T.round(4))
    display(pd.Series(r['definition'],name='factor_definition').to_frame())
    display(pd.Series(c,name='configuration').to_frame())
    heading('Processed factors retain the existing MAD / industry-and-size neutralization / standardization. Missing factors do not enter the ranking. Odd median excluded; ties use stock-code order. Development 2010–2018, validation 2019–2021, test 2022–2024. Reverse direction is diagnostic, not selected on full-period profit. Previously viewed test data are not an untouched holdout. Related factors require multiple-testing and independent out-of-sample checks.')


def export_report(r,folder):
    """Optional, explicitly invoked export; normal notebook execution stays inline."""
    if 'ls' not in r:
        from factor_backtest_close import export_report as export_close
        return export_close(r,folder)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for key in ['ic','ic_summary','coverage','distribution','monthly_ic','yearly_ic','group_forward','gross','net','performance','annual','periods','industry_ic','cost_sensitivity','spreads','ls','reverse_ls','pnl_table']:
        r[key].to_csv(folder/f'{key}.csv',encoding='utf-8-sig')
    (folder/'config.json').write_text(json.dumps(r['config'],ensure_ascii=False,indent=2),encoding='utf-8')
    return folder
