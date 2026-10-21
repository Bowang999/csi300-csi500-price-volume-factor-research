"""Daily OHLCV research factors. Qlib-inspired definitions, MIT attribution in THIRD_PARTY.

This implementation intentionally requires complete trailing windows. Volume is
observed shares (not split-adjusted Qlib volume); VWAP/adjusted-price mixtures
are excluded. All operators are trailing and include the observation date.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit

WINDOWS = (3, 5, 10, 20, 30, 60, 120, 252)
CATALOG = {}
QLIB_URL = 'https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py'

def div(a, b):
    return a / b.where(np.isfinite(b) & b.abs().gt(1e-12)) if isinstance(b, pd.DataFrame) else a / b

@njit(cache=True)
def rolling_extreme_position(a, window, maximum):
    """One-based position from oldest; first occurrence wins ties, like Qlib."""
    nt, ns = a.shape
    out = np.full(a.shape, np.nan)
    for j in range(ns):
        queue = np.empty(nt, np.int64)
        head = tail = valid = 0
        for i in range(nt):
            if i >= window and np.isfinite(a[i-window,j]): valid -= 1
            if np.isfinite(a[i,j]):
                valid += 1
                while tail > head:
                    old = a[queue[tail-1],j]
                    if (maximum and old < a[i,j]) or ((not maximum) and old > a[i,j]): tail -= 1
                    else: break
                queue[tail] = i
                tail += 1
            while head < tail and queue[head] <= i-window: head += 1
            if i >= window-1 and valid == window:
                out[i,j] = (queue[head] - (i-window+1) + 1) / window
    return out

def generate_factors(p, benchmark, windows=WINDOWS):
    """Yield name, wide float64 values, trailing required-good-history count.

    p contains time-by-symbol DataFrames. No membership filtering until after
    calculation; history_valid_days prevents windows crossing bad observations.
    """
    o,h,l,c,v,a,r,turn = [p[k] for k in ('open_hfq','high_hfq','low_hfq','close_hfq','volume','amount','return_1d','turnover_total')]
    prev=c.shift(1); delta=c-prev; vdelta=v-v.shift(1)
    intr=div(c,o)-1; overnight=div(o,prev)-1
    hl=np.log(div(h,l)); co=np.log(div(c,o))
    logv=np.log1p(v); va=np.log1p(div(v,v.shift(1)))
    loga=np.log1p(a)
    clv=div(2*c-h-l,h-l)
    tr=pd.DataFrame(np.maximum.reduce([(h-l).to_numpy(),(h-prev).abs().to_numpy(),(l-prev).abs().to_numpy()]),index=c.index,columns=c.columns)
    b=pd.DataFrame(np.broadcast_to(benchmark.reindex(c.index).to_numpy()[:,None],c.shape),index=c.index,columns=c.columns)
    time=pd.DataFrame(np.broadcast_to(np.arange(len(c))[:,None],c.shape),index=c.index,columns=c.columns)
    def emit(name, value, lookback, family, formula, source='extension', warning='', redundant=''):
        CATALOG[name]={'name':name,'family':family,'lookback':lookback,'formula':formula,'source':source,'warning':warning,'redundant_with':redundant,'recommended':not bool(redundant)}
        return name, value, lookback
    bars={'KMID':(div(c-o,o),'(C-O)/O'), 'KLEN':(div(h-l,o),'(H-L)/O'),
          'KMID2':(div(c-o,h-l),'(C-O)/(H-L)'), 'KUP':(div(h-np.maximum(o,c),o),'(H-max(O,C))/O'),
          'KUP2':(div(h-np.maximum(o,c),h-l),'(H-max(O,C))/(H-L)'),
          'KLOW':(div(np.minimum(o,c)-l,o),'(min(O,C)-L)/O'),
          'KLOW2':(div(np.minimum(o,c)-l,h-l),'(min(O,C)-L)/(H-L)'),
          'KSFT':(div(2*c-h-l,o),'(2C-H-L)/O'), 'KSFT2':(clv,'(2C-H-L)/(H-L)'),
          'OPEN0':(div(o,c),'O/C'),'HIGH0':(div(h,c),'H/C'),'LOW0':(div(l,c),'L/C')}
    for name,(x,formula) in bars.items():
        yield emit('Q158_'+name,x,1,'K线形态',formula,QLIB_URL)
    for n in windows:
        q=lambda name,x,formula,extra=0,vol=False,red='':emit(f'Q158_{name}{n}',x,n+extra,'Qlib窗口特征',formula,QLIB_URL,'原始成交股数，非复权成交量' if vol else '',red)
        rw=c.rolling(n,min_periods=n); rwin=r.rolling(n,min_periods=n)
        ma=rw.mean(); sd=rw.std(ddof=1); hi=h.rolling(n,min_periods=n).max(); lo=l.rolling(n,min_periods=n).min()
        center=time-(n-1)/2; cov=(c*time).rolling(n,min_periods=n).sum()-center*rw.sum()
        ssx=n*(n*n-1)/12
        slope=cov/ssx; ssy=rw.var(ddof=0)*n
        ip=pd.DataFrame(rolling_extreme_position(h.to_numpy(),n,True),index=c.index,columns=c.columns)
        im=pd.DataFrame(rolling_extreme_position(l.to_numpy(),n,False),index=c.index,columns=c.columns)
        pos=delta.clip(lower=0).rolling(n,min_periods=n).sum(); neg=(-delta).clip(lower=0).rolling(n,min_periods=n).sum(); den=delta.abs().rolling(n,min_periods=n).sum()
        vp=vdelta.clip(lower=0).rolling(n,min_periods=n).sum(); vn=(-vdelta).clip(lower=0).rolling(n,min_periods=n).sum(); vd=vdelta.abs().rolling(n,min_periods=n).sum()
        up=r.gt(0).where(r.notna()).rolling(n,min_periods=n).mean(); dn=r.lt(0).where(r.notna()).rolling(n,min_periods=n).mean()
        basic=[('ROC',div(c.shift(n),c),'Ref(C,n)/C',1),('MA',div(ma,c),'Mean(C,n)/C',0),('STD',div(sd,c),'Std(C,n,ddof=1)/C',0),
               ('BETA',div(slope,c),'OLS time slope(C,n)/C',0),('RSQR',div(cov**2,ssy*ssx).clip(0,1),'OLS time regression R2(C,n)',0),
               ('RESI',div(c-ma-slope*(n-1)/2,c),'OLS last residual(C,n)/C',0),('MAX',div(hi,c),'Max(H,n)/C',0),('MIN',div(lo,c),'Min(L,n)/C',0),
               ('QTLU',div(rw.quantile(.8),c),'Quantile(C,n,0.8)/C',0),('QTLD',div(rw.quantile(.2),c),'Quantile(C,n,0.2)/C',0),
               ('RANK',rw.rank(pct=True),'Trailing average-tie rank(C,n)/n',0),('RSV',div(c-lo,hi-lo),'(C-Min(L,n))/(Max(H,n)-Min(L,n))',0),
               ('IMAX',ip,'first argmax(H,n), one-based/n',0),('IMIN',im,'first argmin(L,n), one-based/n',0),('IMXD',ip-im,'IMAX-IMIN',0),
               ('CNTP',up,'Mean(R>0,n)',0),('CNTN',dn,'Mean(R<0,n)',0),('CNTD',up-dn,'CNTP-CNTN',0),
               ('SUMP',div(pos,den),'Sum(max(deltaC,0),n)/Sum(abs(deltaC),n)',1),('SUMN',div(neg,den),'Sum(max(-deltaC,0),n)/Sum(abs(deltaC),n)',1),
               ('SUMD',div(pos-neg,den),'(Sum(up deltaC)-Sum(down deltaC))/Sum(abs(deltaC))',1)]
        for name,x,f,extra in basic:
            red=f'Q158_SUMP{n}' if name in ['SUMN','SUMD'] else (f'Q158_IMAX{n},Q158_IMIN{n}' if name=='IMXD' else (f'Q158_CNTP{n},Q158_CNTN{n}' if name=='CNTD' else ''))
            yield q(name,x,f,extra,False,red)
        volwin=v.rolling(n,min_periods=n); weighted=r.abs()*v
        volume_features=[('CORR',rw.corr(logv),'Corr(C,log(1+V),n)',0),('CORD',r.rolling(n,min_periods=n).corr(va),'Corr(R,log(1+V/Ref(V,1)),n)',1),
                        ('VMA',div(volwin.mean(),v),'Mean(V,n)/V',0),('VSTD',div(volwin.std(),v),'Std(V,n)/V',0),
                        ('WVMA',div(weighted.rolling(n,min_periods=n).std(),weighted.rolling(n,min_periods=n).mean()),'Std(abs(R)*V,n)/Mean(abs(R)*V,n)',0),
                        ('VSUMP',div(vp,vd),'Sum(up deltaV)/Sum(abs(deltaV))',1),('VSUMN',div(vn,vd),'Sum(down deltaV)/Sum(abs(deltaV))',1),('VSUMD',div(vp-vn,vd),'Sum(deltaV)/Sum(abs(deltaV))',1)]
        for name,x,f,extra in volume_features:
            yield q(name,x,f,extra,True,f'Q158_VSUMP{n}' if name in ['VSUMN','VSUMD'] else '')
        ex=lambda name,x,f,extra=0,red='':emit(f'PV_{name}_{n}',x,n+extra,'量价扩展',f,'Own implementation of documented trailing statistics','',red)
        mean=rwin.mean(); std=rwin.std(); am=a.rolling(n,min_periods=n); tu=turn.rolling(n,min_periods=n)
        down=np.sqrt(r.clip(upper=0).pow(2).rolling(n,min_periods=n).mean()); upside=np.sqrt(r.clip(lower=0).pow(2).rolling(n,min_periods=n).mean())
        bm=b.rolling(n,min_periods=n); mvar=bm.var(); beta=div(rwin.cov(b),mvar)
        logreturns=np.log1p(r); logr=logreturns.rolling(n,min_periods=n).sum(); blog=np.log1p(b).rolling(n,min_periods=n).sum()
        additions=[
            ('MOM',np.expm1(logr),'exp(Sum(log(1+R),n))-1',0,f'Q158_ROC{n}'),
            ('RET_MEAN',mean,'Mean(R,n)',0,''),('VOL',std*np.sqrt(252),'Std(R,n)*sqrt(252)',0,''),
            ('SKEW',rwin.skew(),'Sample skewness(R,n)',0,''),('KURT',rwin.kurt(),'Unbiased excess kurtosis(R,n)',0,''),
            ('MAXRET',rwin.max(),'Max(R,n)',0,''),('MINRET',rwin.min(),'Min(R,n)',0,''),
            ('DOWNSIDE',down*np.sqrt(252),'sqrt(Mean(min(R,0)^2,n)*252)',0,''),
            ('UPSIDE',upside*np.sqrt(252),'sqrt(Mean(max(R,0)^2,n)*252)',0,''),
            ('DOWN_UP_RATIO',div(down,upside),'Downside_RMS/Upside_RMS',0,''),
            ('MEAN_VOL',div(mean,std),'Mean(R,n)/Std(R,n)',0,''),
            ('AMIHUD',div(r.abs(),a).rolling(n,min_periods=max(2,int(np.ceil(.8*n)))).mean()*1e8,'Mean(abs(R)/amount, >=ceil(0.8n) positive-amount days)*1e8',0,''),
            ('LOG_AMOUNT',np.log1p(am.mean()),'log(1+Mean(amount,n))',0,''),
            ('AMOUNT_CV',div(am.std(),am.mean()),'Std(amount,n)/Mean(amount,n)',0,''),
            ('AMOUNT_SHOCK',div(a,am.mean()),'amount/Mean(amount,n)',0,''),
            ('AMOUNT_RANK',am.rank(pct=True),'Trailing rank(amount,n)/n',0,''),
            ('TURN_MEAN',tu.mean(),'Mean(volume/total_shares,n)',0,''),
            ('TURN_STD',tu.std(),'Std(volume/total_shares,n)',0,''),
            ('TURN_SHOCK',div(turn,tu.mean()),'turnover_total/Mean(turnover_total,n)',0,''),
            ('TURN_SKEW',tu.skew(),'Skew(turnover_total,n)',0,''),
            ('ATR',div(tr,prev).rolling(n,min_periods=n).mean(),'Mean(max(H-L,abs(H-Cprev),abs(L-Cprev))/Cprev,n)',1,''),
            ('PARKINSON',np.sqrt(hl.pow(2).rolling(n,min_periods=n).mean()/(4*np.log(2))*252),'sqrt(Mean(log(H/L)^2,n)*252/(4log2))',0,''),
            ('GARMAN_KLASS',np.sqrt((.5*hl.pow(2)-(2*np.log(2)-1)*co.pow(2)).rolling(n,min_periods=n).mean().clip(lower=0)*252),'sqrt(252*Mean(0.5log(H/L)^2-(2log2-1)log(C/O)^2,n))',0,''),
            ('INTRADAY_MOM',np.expm1(np.log1p(intr).rolling(n,min_periods=n).sum()),'exp(Sum(log(C/O),n))-1',0,''),
            ('OVERNIGHT_MOM',np.expm1(np.log1p(overnight).rolling(n,min_periods=n).sum()),'exp(Sum(log(O/Cprev),n))-1',1,''),
            ('INTRADAY_VOL',intr.rolling(n,min_periods=n).std()*np.sqrt(252),'Std(C/O-1,n)*sqrt(252)',0,''),
            ('OVERNIGHT_VOL',overnight.rolling(n,min_periods=n).std()*np.sqrt(252),'Std(O/Cprev-1,n)*sqrt(252)',1,''),
            ('AUTOCORR',rwin.corr(r.shift(1)),'Corr(R,Ref(R,1),n)',1,''),
            ('RET_AMOUNT_CORR',rwin.corr(loga),'Corr(R,log(1+amount),n)',0,''),
            ('RET_TURN_CORR',rwin.corr(turn),'Corr(R,turnover_total,n)',0,''),
            ('MONEY_FLOW',div((clv*a).rolling(n,min_periods=n).sum(),am.sum()),'Sum(((2C-H-L)/(H-L))*amount,n)/Sum(amount,n)',0,''),
            ('SIGNED_AMOUNT',div((np.sign(r)*a).rolling(n,min_periods=n).sum(),am.sum()),'Sum(sign(R)*amount,n)/Sum(amount,n)',0,''),
            ('UP_AMOUNT',div((a*r.gt(0)).where(r.notna()).rolling(n,min_periods=n).sum(),am.sum()),'Sum(amount*1[R>0],n)/Sum(amount,n)',0,''),
            ('FROM_HIGH',div(c,rw.max())-1,'C/Max(C,n)-1; not maximum drawdown',0,''),
            ('ZERO_VOLUME',v.eq(0).where(v.notna()).rolling(n,min_periods=n).mean(),'Mean(1[V=0],n)',0,''),
            ('MARKET_BETA',beta,'Cov(R,index_ret,n)/Var(index_ret,n)',0,''),
            ('MARKET_CORR',rwin.corr(b),'Corr(R,index_ret,n)',0,''),
            ('IDIO_VOL',np.sqrt((rwin.var()-beta.pow(2)*mvar).clip(lower=0))*np.sqrt(252),'sqrt(252*(Var(R)-beta^2*Var(index_ret)))',0,''),
            ('REL_MOM',np.expm1(logr-blog),'exp(Sum(log(1+R)-log(1+index_ret),n))-1',0,''),
            ('TRACKING_VOL',(r-b).rolling(n,min_periods=n).std()*np.sqrt(252),'Std(R-index_ret,n)*sqrt(252)',0,''),
            ('CORR_AMOUNT_LEVEL',rw.corr(loga),'Corr(C,log(1+amount),n)',0,''),
            ('VOLUME_RANK',volwin.rank(pct=True),'Trailing rank(raw volume,n)/n',0,''),
        ]
        for name,x,f,extra,red in additions:
            if name=='KURT' and n<4: continue
            item=ex(name,x,f,extra,red)
            if name=='VOLUME_RANK':CATALOG[item[0]]['warning']='原始成交股数，非复权成交量'
            yield item
