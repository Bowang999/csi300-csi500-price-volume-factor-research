"""Validated VWAP prices and signed, equal-capital long/short portfolios."""
import numpy as np
import pandas as pd

def vwap_prices(m):
    # Actual source amount is CNY, volume is shares, not adjusted share volume.
    raw_close=m.market_cap/m.total_shares
    float_close=m.float_market_cap/m.float_shares
    adjustment=m.close_hfq/raw_close
    raw_vwap=m.amount/m.volume
    raw_low=m.low_hfq/adjustment
    raw_high=m.high_hfq/adjustment
    valid=(m.volume.gt(0)&m.amount.gt(0)&raw_close.gt(0)&adjustment.gt(0)
        &m.total_shares.gt(0)&m.float_shares.gt(0)
        &(raw_close-raw_close.round(2)).abs().le(0.000001)
        &(raw_close-float_close).abs().le(.011)
        &raw_vwap.ge(raw_low-.011)&raw_vwap.le(raw_high+.011)
        &m.return_quality.eq('ok_hfq')&~m.source_transition.fillna(True))
    adjusted=(raw_vwap*adjustment).where(valid)
    adjusted=adjusted.where(np.isfinite(adjusted)&adjusted.gt(0))
    return adjusted, valid, raw_vwap

def signed_portfolio(weights, returns, buy_bps=10, sell_bps=10, borrow_bps=0, annual=252):
    """Targets are signed equity weights, +0.5 long / -0.5 short.

    Cash including short proceeds earns zero. Drift uses common portfolio equity,
    not independent long-only sleeves. Costs are first-order return deductions.
    Missing held returns invalidate the whole day and next trading-cost estimate.
    """
    active=weights.abs().sum(axis=1).gt(0)
    complete=(weights.eq(0)|returns.notna()).all(axis=1)
    gross=(weights*returns.fillna(0)).sum(axis=1).where(complete)
    solvent=(1+gross).gt(0)
    gross=gross.where(solvent)
    drift=weights.mul(1+returns).div(1+gross,axis=0).where(weights.ne(0),0)
    prev=drift.shift();prev_active=active.shift(fill_value=False)
    prev.loc[~prev_active]=0
    valid_trade=gross.shift().notna()|~prev_active
    dw=weights-prev
    buys=dw.clip(lower=0).sum(axis=1).where(valid_trade)
    sells=(-dw.clip(upper=0)).sum(axis=1).where(valid_trade)
    if len(weights) and active.iloc[-1]:
        if complete.iloc[-1] and solvent.iloc[-1]:
            # Final cover/sale notionals divided by beginning-of-interval equity.
            end_notional=weights.iloc[-1]*(1+returns.iloc[-1])
            buys.iloc[-1]+=(-end_notional.clip(upper=0)).sum()
            sells.iloc[-1]+=end_notional.clip(lower=0).sum()
        else:
            buys.iloc[-1]=np.nan;sells.iloc[-1]=np.nan
    short_exposure=(-weights.clip(upper=0)).sum(axis=1)
    borrow=short_exposure*borrow_bps/10000/annual
    trade_cost=buys*buy_bps/10000+sells*sell_bps/10000
    cost=trade_cost+borrow
    return pd.DataFrame(dict(gross=gross,net=gross-cost,buys=buys,sells=sells,
        one_way_turnover=(buys+sells)/2,trade_cost=trade_cost,borrow_cost=borrow,cost=cost,
        long_exposure=weights.clip(lower=0).sum(axis=1),short_exposure=short_exposure,
        selected_stocks=weights.ne(0).sum(axis=1),
        missing_label_stocks=(weights.ne(0)&returns.isna()).sum(axis=1),complete_labels=complete))
