import numpy as np
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

def atomic_json(path,value):
    import json
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=str),encoding='utf8');temp.replace(path)
