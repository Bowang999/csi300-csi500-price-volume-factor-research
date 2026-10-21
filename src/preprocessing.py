INDUSTRY_PERIODS=[('P0201', '1900-01-01', '2012-05-16'), ('P0206', '2012-05-16', '2016-12-12'), ('P0215', '2016-12-12', '2021-12-13'), ('P0219', '2021-12-13', '2100-01-01')]
from pathlib import Path
import json,warnings
import numpy as np
import pandas as pd
def industry_records(path):
    d=pd.read_parquet(path) if path.suffix=='.parquet' else pd.DataFrame(json.loads(path.read_text(encoding='utf-8')))
    d['Symbol']=d.Symbol.str.zfill(6).replace({'302132':'300114'})
    d['ImplementDate']=pd.to_datetime(d.ImplementDate)
    key=['Symbol','IndustryClassificationID','ImplementDate']
    if d.groupby(key).IndustryCode.nunique().gt(1).any():raise ValueError('Conflicting industry observations')
    return d.drop_duplicates(key)

def attach_industry(meta, records):
    """Date-wide CSI level 1, with early CSRC2001 fallback before local CSI history.

    Backward as-of lookup carries past effective records forward, never backfills.
    Versions switch for the entire cross section; missing names do not fall back
    to another scheme. ImplementDate is an assumed, not verified publication date.
    """
    d=meta.copy();d['row_id']=np.arange(len(d))
    pieces=[]
    for version,start,end in INDUSTRY_PERIODS:
        left=d.loc[d.date.ge(start)&d.date.lt(end)].sort_values('date')
        if left.empty:continue
        right=records.loc[records.IndustryClassificationID.eq(version),['Symbol','ImplementDate','IndustryCode']].rename(columns={'Symbol':'symbol','ImplementDate':'industry_effective_date','IndustryCode':'industry_detail_code'}).sort_values('industry_effective_date')
        merged=pd.merge_asof(left,right,left_on='date',right_on='industry_effective_date',by='symbol',direction='backward')
        merged['industry_scheme']=version
        # CSI hierarchical codes use two digits per level; preserve leading zeros.
        if version!='P0201':
            codes=merged.industry_detail_code.dropna()
            assert codes.str.fullmatch(r'\d{8}').all(), 'Invalid CSI detail code'
        merged['industry_code']=merged.industry_detail_code.str[:1 if version=='P0201' else 2]
        merged['industry_available_date']=merged.industry_effective_date
        merged['industry_available_date']=merged.industry_available_date.clip(lower=pd.Timestamp(start))
        pieces.append(merged)
    d=pd.concat(pieces).sort_values('row_id').drop(columns='row_id').reset_index(drop=True)
    d['industry_known']=d.industry_code.notna()
    d['market_cap_valid']=np.isfinite(d.market_cap)&d.market_cap.gt(0)
    d['log_market_cap']=np.log(d.market_cap.where(d.market_cap_valid))
    d['neutralization_eligible']=d.industry_known&d.market_cap_valid&d.factor_input_valid
    assert not (d.industry_available_date>d.date).fillna(False).any()
    return d

def preprocess(values, meta):
    """Per-date 5 scaled-MAD winsor -> industry+logcap OLS residual -> zscore.

    Factor-specific finite samples; industries with fewer than two finite names
    excluded. FWL demeaning is algebraically equivalent to industry-dummy OLS.
    No imputation and no post-neutralization clipping.
    """
    out=np.full(values.shape,np.nan,np.float32)
    diag={'max_abs_mean':0.,'max_std_error':0.,'max_abs_industry_mean':0.,'max_abs_logcap_corr':0.,'constant_or_insufficient_columns':0,'mad_zero_columns':0}
    dates=meta.date.to_numpy(); cuts=np.r_[0,np.flatnonzero(dates[1:]!=dates[:-1])+1,len(meta)]
    for begin,end in zip(cuts[:-1],cuts[1:]):
        m=meta.iloc[begin:end]; x=values[begin:end].astype(np.float64)
        risk=m.neutralization_eligible.to_numpy(bool)
        # MAD uses all valid factor observations in today's member universe.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            med=np.nanmedian(x,axis=0); mad=np.nanmedian(np.abs(x-med),axis=0)
        positive=mad>1e-12
        diag['mad_zero_columns']+=int((~positive & np.isfinite(med)).sum())
        width=5*1.4826*mad
        x[:,positive]=np.clip(x[:,positive],med[positive]-width[positive],med[positive]+width[positive])
        x[~risk]=np.nan
        ln=m.log_market_cap.to_numpy(float)
        industry=m.industry_code.fillna('MISSING').to_numpy(str)
        yd=np.full_like(x,np.nan); xd=np.full_like(x,np.nan)
        group_counts=np.zeros(x.shape[1],int)
        for group in np.unique(industry[risk]):
            ix=np.flatnonzero(risk&(industry==group)); y=x[ix]; finite=np.isfinite(y); counts=finite.sum(axis=0)
            good=counts>=2
            if not good.any():continue
            group_counts+=good
            sy=np.nansum(y,axis=0); sx=np.sum(np.where(finite,ln[ix,None],0),axis=0)
            my=np.divide(sy,counts,out=np.zeros_like(sy),where=counts>0)
            mx=np.divide(sx,counts,out=np.zeros_like(sx),where=counts>0)
            y0=y-my; x0=np.broadcast_to(ln[ix,None],y.shape)-mx
            y0[:,~good]=np.nan;x0[:,~good]=np.nan;x0[~finite]=np.nan
            yd[ix]=y0;xd[ix]=x0
        denom=np.nansum(xd*xd,axis=0); n=np.isfinite(yd).sum(axis=0)
        fit=(n>=np.maximum(30,group_counts+10))&(denom>1e-12)
        slope=np.divide(np.nansum(xd*yd,axis=0),denom,out=np.zeros_like(denom),where=denom>1e-12)
        res=yd-xd*slope
        var=np.divide(np.nansum(res*res,axis=0),n,out=np.zeros_like(denom),where=n>0)
        valid=fit&(var>1e-16)
        res[:,~valid]=np.nan
        z=np.divide(res,np.sqrt(var),out=np.full_like(res,np.nan),where=(var>1e-16)[None,:]).astype(np.float32)
        out[begin:end]=z
        diag['constant_or_insufficient_columns']+=int((~valid).sum())
        if valid.any():
            zz=z[:,valid].astype(float); nn=np.isfinite(zz).sum(axis=0)
            mu=np.nansum(zz,axis=0)/nn; sd=np.sqrt(np.nansum(zz*zz,axis=0)/nn)
            diag['max_abs_mean']=max(diag['max_abs_mean'],float(np.max(abs(mu))))
            diag['max_std_error']=max(diag['max_std_error'],float(np.max(abs(sd-1))))
            lc=np.broadcast_to(ln[:,None],zz.shape)
            mask=np.isfinite(zz); lm=np.sum(np.where(mask,lc,0),axis=0)/nn
            lc=lc-lm; numer=np.nansum(zz*lc,axis=0); den=np.sqrt(np.sum(np.where(mask,lc*lc,0),axis=0)*nn)
            cor=np.divide(numer,den,out=np.zeros_like(den),where=den>0)
            diag['max_abs_logcap_corr']=max(diag['max_abs_logcap_corr'],float(np.max(abs(cor))))
            for group in np.unique(industry[risk]):
                zgroup=zz[industry==group]; ng=np.isfinite(zgroup).sum(axis=0)
                gm=np.divide(np.nansum(zgroup,axis=0),ng,out=np.zeros(zz.shape[1]),where=ng>0)
                diag['max_abs_industry_mean']=max(diag['max_abs_industry_mean'],float(np.max(abs(gm))))
    assert diag['max_abs_mean']<1e-5 and diag['max_std_error']<1e-5
    assert diag['max_abs_logcap_corr']<1e-5 and diag['max_abs_industry_mean']<1e-5
    return out,diag
