"""Point-in-time half-universe portfolio accounting; adjusted units, not real shares."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def load_market(root, universe, start, end, valuation_policy='strict', terminal_raw_mark=.11):
    path=Path(root)/'data'/universe/'parquet'/f'{universe}_daily_market_2010_2024.parquet'
    cols=['date','symbol','close_hfq','high_hfq','low_hfq','vwap_hfq','volume','amount','market_data_status','return_quality','source_transition']
    d=pd.read_parquet(path,columns=cols)
    actions=json.loads((Path(__file__).parent/'corporate_actions.json').read_text(encoding='utf-8'))
    present=set(d.symbol);needed={a['new'] for a in actions if a['old'] in present}-present
    if needed:
        other='csi500' if universe=='csi300' else 'csi300'
        extra=pd.read_parquet(Path(root)/'data'/other/'parquet'/f'{other}_daily_market_2010_2024.parquet',columns=cols,filters=[('symbol','in',list(needed))])
        d=pd.concat([d,extra],ignore_index=True)
    d.date=pd.to_datetime(d.date)
    terminal_anchor=d.loc[d.symbol.eq('000540')&d.date.eq('2023-05-18'),'close_hfq']
    d=d.loc[d.date.between(start,end)]
    core=d.close_hfq.gt(0)&d.close_hfq.between(d.low_hfq-.002,d.high_hfq+.002)
    # Source ChangeRatio and supplied adjusted-price returns are different fields.
    # A discrepancy is an audit warning, not proof that the supplied HFQ close is wrong.
    trade=core&d.volume.gt(0)&d.amount.gt(0)&~d.source_transition&d.market_data_status.eq('observed_trading')
    vwap=d.vwap_hfq.gt(0)&d.vwap_hfq.between(d.low_hfq-.002,d.high_hfq+.002)
    d['fill']=d.vwap_hfq.where(vwap,d.close_hfq).where(trade)
    d['fallback']=trade&~vwap
    d['mark']=d.close_hfq.where(core)
    d['uncertain']=(~core|d.source_transition)
    d['return_disagreement']=d.return_quality.eq('source_hfq_conflict')
    d['no_trade']=d.market_data_status.eq('zero_volume')
    d['delisted']=d.market_data_status.eq('known_delisted')
    d['pending_action']=False;d['estimated_mark']=False
    for a in actions:
        old=d.symbol.eq(a['old'])
        pending=old&d.date.ge(a['pending_from'])&d.date.lt(a['date'])
        d.loc[pending,'pending_action']=True;d.loc[pending,'uncertain']=False;d.loc[pending,'no_trade']=True
        if a['old']==a['new']:
            d.loc[old&d.date.eq(a['date']),'uncertain']=False
    if valuation_policy=='published_fund':
        # Announced 2023-06-10, effective 2023-06-12; no retrospective mark or simulated sale.
        stock=d.symbol.eq('000540');anchor=terminal_anchor
        if len(anchor):
            factor=float(anchor.iloc[0])/.40
            sel=stock&d.date.ge('2023-06-12')
            d.loc[sel,'mark']=terminal_raw_mark*factor
            d.loc[sel,'uncertain']=False;d.loc[sel,'no_trade']=True
            d.loc[sel,'estimated_mark']=True;d.loc[sel,'fill']=np.nan
    elif valuation_policy!='strict':raise ValueError('Unknown valuation policy')
    result={k:d.pivot(index='date',columns='symbol',values=k) for k in ['fill','fallback','mark','uncertain','no_trade','delisted','pending_action','estimated_mark','return_disagreement']}
    result['actions']=[a for a in actions if a['old'] in present and start<=a['date']<=end]
    return result


def half_weights(factors,min_stocks=30):
    """Stable code tie break. Odd median omitted. No future execution filter."""
    a=factors.to_numpy();out=np.zeros(a.shape)
    for i,row in enumerate(a):
        valid=np.flatnonzero(np.isfinite(row))
        if len(valid)<min_stocks:continue
        ordered=valid[np.argsort(row[valid],kind='stable')];n=len(ordered)//2
        out[i,ordered[:n]]=-.5/n;out[i,ordered[-n:]]=.5/n
    return pd.DataFrame(out,index=factors.index,columns=factors.columns)


def simulate(weights,market,buy_bps=10,sell_bps=10,borrow_bps=0,annual=252,fee_schedule=None,cash_only=False,rebalance_mask=None):
    """Self-financing ledger. t-1 targets executed on t; terminal holdings marked, not liquidated.

    Target sizing uses previous-close equity (known before execution). Frozen side
    exposures reduce that side's tradable budget. Unfilled target budgets stay cash.
    Prices are adjusted analytical units. Suspensions can retain stale marks; an
    uncertain held-price boundary permanently taints the official NAV from that day.
    Indicative raw-price ledger is retained separately for diagnosis, never certified.
    """
    ix=weights.index;cols=weights.columns
    if rebalance_mask is None:
        rebalance=np.ones(len(ix),dtype=bool)
    else:
        mask=rebalance_mask.reindex(ix)
        if mask.isna().any():raise ValueError('Incomplete rebalance schedule')
        rebalance=mask.to_numpy(dtype=bool)
    arr={k:v.reindex(index=ix,columns=cols).to_numpy(dtype=float) for k,v in market.items() if k!='actions'}
    scheduled={}
    for a in market.get('actions',[]):
        if a['old'] in cols and a['new'] in cols:scheduled.setdefault(pd.Timestamp(a['date']),[]).append(a)
    targets=weights.shift(1,fill_value=0).to_numpy()
    for key in ['can_buy','can_sell']:arr.setdefault(key,np.ones_like(targets,dtype=bool))
    for key in ['estimated_mark','return_disagreement']:arr.setdefault(key,np.zeros_like(targets,dtype=bool))
    if fee_schedule is not None:
        fee_schedule=fee_schedule.reindex(ix)
        if fee_schedule.isna().any().any():raise ValueError('Incomplete fee schedule')
    qty=np.zeros(len(cols));last=np.full(len(cols),np.nan);cash=1.;previous=1.;tainted=False
    stale=np.zeros(len(cols),dtype=int);rows=[];events=[];event_symbols=set()
    for i,date in enumerate(ix):
        raw=arr['mark'][i];fill=arr['fill'][i];can=np.isfinite(fill)&(fill>0)
        corporate_pnl=0.;corp_long=0.;corp_count=0
        # Deliver successor units before trading; net with any existing holding.
        for a in scheduled.get(date,[]):
            old=cols.get_loc(a['old']);new=cols.get_loc(a['new']);units=qty[old]
            if abs(units)<=1e-14:continue
            value=units*last[old];converted=units*a['adjusted_unit_multiplier']
            if old==new:
                qty[old]=converted;last[old]/=a['adjusted_unit_multiplier']
            else:
                basis=last[new] if np.isfinite(last[new]) else raw[new]
                if not np.isfinite(basis):raise ValueError('Missing successor conversion price '+a['new'])
                change=converted*basis-value
                corporate_pnl+=change;corp_long+=change if units>0 else 0.
                qty[old]=0.;qty[new]+=converted;last[new]=basis
            corp_count+=1
        held=np.abs(qty)>1e-14
        # No bridging of uncertain returns as if they had been independently validated.
        bad=held&((arr['uncertain'][i]>0)|(~np.isfinite(raw)&~(arr['no_trade'][i]>0)))
        if bad.any():
            tainted=True
            for j in np.flatnonzero(bad):
                if j not in event_symbols:
                    events.append((str(date.date()),str(cols[j]),'first_unresolved_held_price',float(qty[j])))
                    event_symbols.add(j)
        mark=np.where(np.isfinite(raw),raw,last)
        stale=np.where(held&(~can),stale+1,0)
        if previous<=0:raise ValueError('Portfolio insolvent; cannot continue percentage-return simulation')
        target=targets[i];desired=np.zeros(len(cols));frozen=held&~can
        old_values=qty*np.nan_to_num(last,nan=0.)
        for sign in (1,-1):
            selected=target*sign>0
            frozen_value=np.sum(np.maximum(sign*old_values[frozen],0.))
            budget=max(0.,np.sum(np.abs(target[selected]))*previous-frozen_value)
            # Frozen selected names already consume the side budget; blocked new orders stay cash.
            pool=selected&~frozen;n=int(pool.sum())
            if n:desired[pool&can]=sign*budget/n
        newqty=qty.copy()
        if rebalance[i]:newqty[can]=desired[can]/fill[can]
        delta=newqty-qty
        # Optional directional restrictions act on orders, not signal ranking.
        can_buy=arr['can_buy'][i]>0
        can_sell=arr['can_sell'][i]>0
        blocked_order=(delta>0)&~can_buy | (delta<0)&~can_sell
        newqty[blocked_order]=qty[blocked_order]
        delta=newqty-qty;notional=np.where(can,delta*np.nan_to_num(fill),0.)
        buy=notional.clip(min=0).sum();sell=(-notional).clip(min=0).sum()
        components={}
        if fee_schedule is None:
            components['baseline_fee']=(buy*buy_bps+sell*sell_bps)/10000
            rb,rs=buy_bps/10000,sell_bps/10000
        else:
            fr=fee_schedule.iloc[i]
            rb=sum(fr[k] for k in fr.index if k.endswith('_buy'))/10000
            rs=sum(fr[k] for k in fr.index if k.endswith('_sell'))/10000
            for name in ('commission','stamp','slippage','transfer'):
                components[name]=(buy*fr.get(name+'_buy',0)+sell*fr.get(name+'_sell',0))/10000
        if cash_only:
            if (newqty < -1e-12).any():raise ValueError('Cash-only portfolio cannot short')
            available=max(0.,cash+sell*(1-rs))
            scale=min(1.,available/(buy*(1+rb))) if buy>0 else 1.
            if scale<1:
                delta=np.where(delta>0,delta*scale,delta);newqty=qty+delta
                notional=np.where(can,delta*np.nan_to_num(fill),0.)
                buy=notional.clip(min=0).sum()
                if fee_schedule is None:components['baseline_fee']=(buy*buy_bps+sell*sell_bps)/10000
                else:
                    for name in components:components[name]=(buy*fr.get(name+'_buy',0)+sell*fr.get(name+'_sell',0))/10000
        fee=sum(components.values())
        borrow=np.maximum(-old_values,0).sum()*borrow_bps/10000/annual
        cash-=notional.sum()+fee+borrow
        old_mark_pnl=qty*np.nan_to_num(mark-last,nan=0.)
        new_mark_pnl=delta*np.where(can,np.nan_to_num(mark-fill,nan=0.),0.)
        pnl=old_mark_pnl+new_mark_pnl
        long_delta=newqty.clip(min=0)-qty.clip(min=0)
        longs=np.sum(qty.clip(min=0)*np.nan_to_num(mark-last,nan=0.))+np.sum(long_delta*np.where(can,np.nan_to_num(mark-fill,nan=0.),0.))
        qty=newqty;values=qty*np.nan_to_num(mark,nan=0.);equity=cash+values.sum()
        if equity<=0:raise ValueError('Portfolio insolvent; cannot continue percentage-return simulation')
        gross=(pnl.sum()+corporate_pnl)/previous;net=equity/previous-1
        if not np.isclose(net,gross-(fee+borrow)/previous,atol=1e-10):raise AssertionError('Cash/holdings reconciliation failed')
        turnover=(buy+sell)/previous;fallback=arr['fallback'][i]>0
        blocked=(~can)&((np.abs(target)>0)|held)&rebalance[i]
        rows.append(dict(date=date,gross=gross,net=net,nav=equity,buys=buy/previous,sells=sell/previous,
            **{name+'_cost':value/previous for name,value in components.items()},
            scheduled_rebalance=bool(rebalance[i]),restricted_order_count=int(blocked_order.sum()),cash_balance=cash,
            two_way_turnover=turnover,one_way_turnover=turnover/2,cost=fee/previous,borrow_cost=borrow/previous,
            selected_stocks=int(np.count_nonzero(target)),long_count=int((qty>1e-14).sum()),short_count=int((qty< -1e-14).sum()),
            long_exposure=values.clip(min=0).sum()/equity,short_exposure=(-values).clip(min=0).sum()/equity,cash_weight=cash/equity,
            frozen_count=int(frozen.sum()),frozen_exposure=np.abs(values[frozen]).sum()/equity,
            blocked_count=int(blocked.sum()),fallback_count=int((fallback&(np.abs(notional)>1e-14)).sum()),
            fallback_notional=np.abs(notional[fallback]).sum()/previous,missing_label_stocks=int(bad.sum()),
            valuation_issue_count=int(bad.sum()),official_valid=not tainted,max_frozen_days=int(stale.max()),
            gross_same_holdings=gross,corporate_action_count=corp_count,corporate_action_pnl=corporate_pnl/previous,
            estimated_mark_exposure=np.abs(values[arr['estimated_mark'][i]>0]).sum()/equity,
            source_return_warning_count=int((held&(arr['return_disagreement'][i]>0)).sum()),
            long_pnl=(longs+corp_long)/previous,short_pnl=(pnl.sum()+corporate_pnl-longs-corp_long)/previous))
        last=mark;previous=equity
    result=pd.DataFrame(rows).set_index('date')
    result.attrs['valuation_events']=events
    return result


def full_metrics(s,annual=252,rf=0):
    """All dates required; never silently substitute longest observed segment."""
    n=len(s);base={'valid_days':int(s.notna().sum()),'missing_days':int(s.isna().sum())}
    keys=['mean_daily','annual_vol','Sharpe','Sortino','win_rate','CAGR','max_drawdown','Calmar','total_return']
    if not n or not np.isfinite(s).all() or (s<=-1).any():return {**base,**{k:np.nan for k in keys},'drawdown_start':'','drawdown_trough':'','drawdown_recovery':''}
    nav=(1+s).cumprod();dd=nav/nav.cummax().clip(lower=1)-1;trough=dd.idxmin()
    before=nav.loc[:trough];peak=before.max()
    peak_date=before.idxmax() if peak>1 else s.index[0]
    recovery=nav.loc[trough:];recovery=recovery[recovery>=max(1,peak)]
    rf_daily=np.expm1(np.log1p(rf)/annual);excess=s-rf_daily
    sd=excess.std(ddof=1);down=np.sqrt(np.mean(np.minimum(excess,0)**2));cagr=nav.iloc[-1]**(annual/n)-1
    return {**base,'mean_daily':s.mean(),'annual_vol':s.std(ddof=1)*np.sqrt(annual),'Sharpe':excess.mean()/sd*np.sqrt(annual) if sd>0 else np.nan,
        'Sortino':excess.mean()/down*np.sqrt(annual) if down>0 else np.nan,'win_rate':s.gt(0).mean(),
        'CAGR':cagr,'max_drawdown':dd.min(),'Calmar':cagr/abs(dd.min()) if dd.min()<0 else np.nan,'total_return':nav.iloc[-1]-1,
        'drawdown_start':str(peak_date.date()),'drawdown_trough':str(trough.date()),'drawdown_recovery':str(recovery.index[0].date()) if len(recovery) else 'not recovered'}


def pnl_table(p,annual=252,rf=0,indicative=False):
    ranges=[('FULL',p.index[0],p.index[-1])]+[(str(y),f'{y}-01-01',f'{y}-12-31') for y in sorted(set(p.index.year))]
    ranges += [('development','2010-01-01','2018-12-31'),('validation','2019-01-01','2021-12-31'),('test','2022-01-01','2024-12-31')]
    rows=[]
    for label,start,end in ranges:
        x=p.loc[start:end]
        if x.empty:continue
        valid=x.official_valid.all();m=full_metrics(x.net if valid or indicative else x.net*np.nan,annual,rf)
        estimated=x.get('estimated_mark_exposure',pd.Series(0,index=x.index)).gt(0).any()
        rows.append(dict(period=label,from_date=str(x.index[0].date()),to_date=str(x.index[-1].date()),status=('RESEARCH / ESTIMATED' if estimated else 'RESEARCH') if valid else ('INDICATIVE ONLY' if indicative else 'UNRESOLVED VALUATION'),
            annual_return_pct=m['CAGR']*100,SR=m['Sharpe'],max_dd_pct=m['max_drawdown']*100,dd_start=m['drawdown_start'],dd_trough=m['drawdown_trough'],
            daily_two_way_turnover_pct=x.two_way_turnover.mean()*100,win_pct=m['win_rate']*100))
    return pd.DataFrame(rows).set_index('period')
