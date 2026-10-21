"""Independent hand-calculable acceptance checks, run before any real-data batch."""
import numpy as np
import pandas as pd
from .daily_execution import simulate,full_metrics
from .research import metrics,historical_fees,yearly_direction,rebalance_schedule,terminal_value_at_cost

def fixture(prices,weights,fill=None,actions=None):
    ix=pd.bdate_range('2020-01-01',periods=len(prices));p=pd.DataFrame(prices,index=ix,dtype=float)
    w=pd.DataFrame(weights,index=ix,columns=p.columns,dtype=float)
    m={k:pd.DataFrame(False,index=ix,columns=p.columns) for k in ['fallback','uncertain','no_trade','delisted','pending_action','estimated_mark','return_disagreement']}
    m.update(mark=p,fill=p.copy() if fill is None else pd.DataFrame(fill,index=ix,columns=p.columns,dtype=float),actions=actions or [])
    return w,m

def run():
    checks=[]
    w,m=fixture([[10],[11],[12]],[[0],[0],[0]])
    p=simulate(w,m);assert np.allclose(p.nav,1);assert p.two_way_turnover.sum()==0;checks.append('no_trade_cash_unchanged')
    w,m=fixture([[10],[10],[11],[11]],[[1],[0],[0],[0]])
    p=simulate(w,m,10,10)
    assert np.allclose(p.nav,[1,.999,1.0979,1.0979]);assert np.isclose(p.cost.iloc[1],.001)
    checks.append('single_buy_sell_exact_cash_fees')
    w,m=fixture([[10,10],[10,10],[10,10]],[[.5,-.5],[-.5,.5],[0,0]])
    p=simulate(w,m,0,0);assert np.allclose(p.two_way_turnover,[0,1,2]);assert np.allclose(p.nav,1)
    checks.append('long_short_flip_two_way_turnover')
    w,m=fixture([[10],[10],[10],[12]],[[1],[0],[0],[0]],[[10],[10],[np.nan],[12]])
    m['no_trade'].iloc[2]=True;p=simulate(w,m,0,0)
    assert p.frozen_count.iloc[2]==1 and p.two_way_turnover.iloc[2]==0;assert np.isclose(p.nav.iloc[-1],1.2)
    checks.append('suspension_freeze_then_real_exit')
    w,m=fixture([[10,5],[10,5],[np.nan,5]],[[1,0],[0,1],[0,0]])
    w.columns=['old','new']
    for k in m:
        if k!='actions':m[k].columns=w.columns
    m['actions']=[dict(old='old',new='new',date=str(w.index[2].date()),adjusted_unit_multiplier=2)]
    p=simulate(w,m,0,0);assert np.allclose(p.nav,1);assert p.two_way_turnover.iloc[2]==0
    assert p.corporate_action_count.iloc[2]==1;checks.append('share_conversion_no_fictitious_trade')
    w,m=fixture([[10],[10],[11]],[[1],[1],[1]])
    p=simulate(w,m,10,10,cash_only=True)
    assert p.cash_balance.min()>-1e-12;assert np.isclose(p.nav.iloc[1],1/1.001)
    checks.append('long_only_fee_reserved_no_cash_borrowing')
    # Known directional block must not fabricate a fill.
    m['can_buy']=pd.DataFrame(False,index=w.index,columns=w.columns)
    p=simulate(w,m,0,0);assert np.allclose(p.nav,1);assert p.restricted_order_count.sum()>0
    checks.append('blocked_buy_stays_cash')
    dates=pd.to_datetime(['2023-08-25','2023-08-28']);fees=historical_fees(dates)
    assert list(fees.stamp_sell)==[10,5] and list(fees.stamp_buy)==[0,0]
    w,m=fixture([[10],[10],[10]],[[1],[0],[0]])
    p=simulate(w,m,fee_schedule=historical_fees(w.index))
    assert np.allclose(p.nav,[1,.9995,.998]);assert np.allclose(p.cost,p.commission_cost+p.stamp_cost+p.slippage_cost+p.transfer_cost)
    checks.append('dated_stamp_and_cost_component_reconciliation')
    s=pd.Series([.1,-.1],index=pd.to_datetime(['2020-01-01','2020-12-31']))
    a=metrics(s);assert np.isclose(a['total_return'],-.01)
    assert np.isclose(a['CAGR'],.99**126-1);assert np.isclose(a['CAGR_calendar'],.99**(365.2425/366)-1)
    assert np.isclose(a['Sharpe'],0);assert np.isclose(a['max_drawdown'],-.1)
    checks.append('compounding_calendar_sharpe_drawdown_handcheck')
    s=pd.Series([.02,-.01,.03],index=pd.bdate_range('2020-01-01',periods=3))
    m=full_metrics(s);expected_sd=np.sqrt(((.02- .04/3)**2+(-.01- .04/3)**2+(.03- .04/3)**2)/2)
    assert np.isclose(m['Sharpe'],(.04/3)/expected_sd*np.sqrt(252))
    firstloss=full_metrics(pd.Series([-.1,0],index=s.index[:2]))
    assert np.isclose(firstloss['max_drawdown'],-.1)
    checks.append('nonzero_sharpe_ddof1_and_initial_nav_drawdown')
    ix=pd.bdate_range('2010-01-01','2015-12-31');ic=pd.Series(.1,index=ix)
    signs,table=yearly_direction(ic,ix);changed=ic.copy();changed.loc['2015':]=-10
    other,_=yearly_direction(changed,ix)
    assert np.array_equal(signs,other)
    for row in table.itertuples():
        if row.direction:assert pd.Timestamp(row.latest_label_exit)<pd.Timestamp(f'{row.year}-01-01')
    checks.append('rolling_direction_future_perturbation_and_purge')
    # Frozen delisted value is not cash and does not create turnover.
    w,m=fixture([[.4],[.4],[.11]],[[1],[0],[0]],[[.4],[.4],[np.nan]])
    m['no_trade'].iloc[2]=True;m['estimated_mark'].iloc[2]=True
    p=simulate(w,m,0,0);assert np.isclose(p.nav.iloc[-1],.275);assert p.two_way_turnover.iloc[-1]==0
    checks.append('delisted_estimated_mark_no_sale')
    w,m=fixture([[10+i] for i in range(24)],[[1] for _ in range(24)])
    for every in (5,20):
        schedule=rebalance_schedule(w.index,w.index[1],every)
        p=simulate(w,m,0,0,rebalance_mask=schedule)
        assert list(np.flatnonzero(schedule))==list(range(1,24,every))
        assert np.allclose(p.nav.iloc[1:every+1],np.arange(11,11+every)/11)
        assert p.loc[~schedule,['buys','sells','cost']].abs().to_numpy().max()==0
    checks.append('5_20_day_fixed_units_and_exact_schedule')
    schedule=rebalance_schedule(w.index,w.index[1],5)
    changed=w.copy();changed.iloc[1:5]=-1
    a=simulate(w,m,0,0,rebalance_mask=schedule);b=simulate(changed,m,0,0,rebalance_mask=schedule)
    assert np.allclose(a.nav,b.nav)
    checks.append('off_schedule_signals_do_not_change_holdings')
    a=simulate(w,m,10,10);b=simulate(w,m,10,10,rebalance_mask=pd.Series(True,index=w.index))
    assert np.array_equal(a.net,b.net)
    checks.append('daily_schedule_exact_backward_compatibility')
    w,m=fixture([[10],[10],[10],[10],[10]],[[1],[0],[0],[0],[0]],[[10],[10],[np.nan],[10],[10]])
    schedule=pd.Series([False,True,True,False,True],index=w.index)
    m['no_trade'].iloc[2]=True;p=simulate(w,m,0,0,rebalance_mask=schedule)
    assert p.long_count.iloc[3]==1 and p.two_way_turnover.iloc[3]==0 and p.long_count.iloc[4]==0
    checks.append('blocked_order_waits_until_next_scheduled_day')
    w,m=fixture([[10],[10],[10],[10]],[[ -1],[-1],[-1],[-1]])
    schedule=pd.Series([False,True,False,False],index=w.index)
    p=simulate(w,m,0,0,borrow_bps=252,rebalance_mask=schedule)
    assert p.borrow_cost.iloc[2]>0 and p.borrow_cost.iloc[3]>0 and p.cost.iloc[2]==0
    checks.append('daily_borrow_charge_between_rebalances')
    w,m=fixture([[10,5],[10,5],[np.nan,5]],[[1,0],[0,1],[0,0]])
    w.columns=['old','new']
    for key in m:
        if key!='actions':m[key].columns=w.columns
    m['actions']=[dict(old='old',new='new',date=str(w.index[2].date()),adjusted_unit_multiplier=2)]
    p=simulate(w,m,0,0,rebalance_mask=pd.Series([False,True,False],index=w.index))
    assert p.corporate_action_count.iloc[2]==1 and p.two_way_turnover.iloc[2]==0 and np.isclose(p.nav.iloc[2],1)
    checks.append('corporate_conversion_between_rebalances')
    w,m=fixture([[10],[10],[10]],[[100],[100],[100]])
    assert terminal_value_at_cost(w,m,100)==-np.inf
    checks.append('insolvent_cost_stress_brackets_loss_without_aborting_report')
    return checks

if __name__=='__main__':
    import json
    print(json.dumps({'success':True,'checks':run()},indent=2))
