"""Construct trailing base and corrected report factors from normalized local data.

No vendor data are downloaded. History must include pre-membership observations.
The full history start affects recursive formulas; use the same origin for comparisons.
"""
from pathlib import Path
import argparse,sys,json,gc,warnings,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from factor_library import generate_factors
from preprocessing import attach_industry,industry_records,preprocess
from stable_report_engine import ExactTwoPointEngine
from alpha_report_engine import Parser
REPO=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--universe',choices=['csi300','csi500'],required=True);ap.add_argument('--industry',type=Path,required=True);ap.add_argument('--start-year',type=int,default=2010);ap.add_argument('--end-year',type=int,default=2026);a=ap.parse_args()
    d=pd.read_parquet(a.root/f'data/{a.universe}/market.parquet');d.date=pd.to_datetime(d.date);d=d.sort_values(['date','symbol']);assert not d.duplicated(['date','symbol']).any()
    ix=pd.read_parquet(a.root/f'data/{a.universe}/index.parquet');ix.date=pd.to_datetime(ix.date);ix=ix.set_index('date').sort_index();dates=ix.index;symbols=pd.Index(sorted(d.symbol.unique()))
    assert d.date.isin(dates).all();good=d.return_quality.eq('ok_hfq');member=d.pivot(index='date',columns='symbol',values='is_member').reindex(index=dates,columns=symbols).fillna(False).to_numpy(bool)
    cols=['open_hfq','high_hfq','low_hfq','close_hfq','vwap_hfq','volume','amount','return_1d','turnover_total','market_cap']
    panel={c:d.assign(value=d[c].where(good)).pivot(index='date',columns='symbol',values='value').reindex(index=dates,columns=symbols) for c in cols}
    meta=d.loc[d.is_member,['date','symbol','market_cap','factor_input_valid','history_valid_days','return_quality']].copy();meta['is_member']=True;meta=attach_industry(meta,industry_records(a.industry));meta=meta.sort_values(['date','symbol']).reset_index(drop=True)
    rr=dates.get_indexer(meta.date);cc=symbols.get_indexer(meta.symbol);output=a.root/'factor_store';output.mkdir(exist_ok=True)
    # Base factors use a 255-session trailing warmup, matching the historical builder.
    for year in range(a.start_year,a.end_year+1):
        take=meta.date.dt.year.eq(year).to_numpy()
        if not take.any():continue
        local=meta.loc[take].reset_index(drop=True);begin=max(0,dates.searchsorted(pd.Timestamp(year,1,1))-255);end=dates.searchsorted(pd.Timestamp(year+1,1,1));pp={k:v.iloc[begin:end] for k,v in panel.items()}
        arrays=[];names=[]
        for name,wide,lookback in generate_factors(pp,ix.index_ret):
            v=wide.to_numpy()[rr[take]-begin,cc[take]].astype('float32');v[~local.factor_input_valid.to_numpy()|(local.history_valid_days.to_numpy()<lookback)|~np.isfinite(v)]=np.nan;names.append(name);arrays.append(v)
        raw=np.column_stack(arrays);scaled,_=preprocess(raw,local)
        for stage,values in [('raw',raw),('processed',scaled)]:
            path=output/stage/f'universe={a.universe}/year={year}/part-0.parquet';path.parent.mkdir(parents=True,exist_ok=True);pd.concat([local,pd.DataFrame(values,columns=names)],axis=1).to_parquet(path,index=False)
        print('BASE',year,len(names),flush=True)
    mapping={'OPEN':'open_hfq','HIGH':'high_hfq','LOW':'low_hfq','CLOSE':'close_hfq','VWAP':'vwap_hfq','VOLUME':'volume','AMOUNT':'amount','CAP':'market_cap'}
    fields={k:panel[v].to_numpy().copy() for k,v in mapping.items()}
    for k,x in fields.items():x[x<=0 if k in ['OPEN','HIGH','LOW','CLOSE','VWAP','CAP'] else x<0]=np.nan
    # Formula-internal industry grouping differs from CSI neutralization industry.
    if 'formula_industry' not in d:raise ValueError('Supply historical formula_industry codes; see docs/DATA.md')
    ind=d.pivot(index='date',columns='symbol',values='formula_industry').reindex(index=dates,columns=symbols).fillna(0).to_numpy('int16')
    engine=ExactTwoPointEngine(fields,member,ind)
    if a.universe=='csi500':
        for key,col in [('BANCHMARKINDEXOPEN','open'),('BANCHMARKINDEXCLOSE','close')]:fields[key]=np.broadcast_to(ix[col].to_numpy()[:,None],member.shape)
    derived={'RETURNS':'CLOSE/DELAY(CLOSE,1)-1','DTM':'OPEN<=DELAY(OPEN,1)?0:MAX(HIGH-OPEN,OPEN-DELAY(OPEN,1))','DBM':'OPEN>=DELAY(OPEN,1)?0:MAX(OPEN-LOW,OPEN-DELAY(OPEN,1))','TR':'MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))','HD':'HIGH-DELAY(HIGH,1)','LD':'DELAY(LOW,1)-LOW'}
    for k,expr in derived.items():fields[k]=engine.evaluate(Parser(expr).parse())
    fields['RET']=fields['RETURNS'];ready=[f for f in json.loads((REPO/'factors/report_formulas.json').read_text()) if f['status']=='ready']
    for start in range(0,len(ready),24):
        names=[];arrays=[]
        for f in ready[start:start+24]:
            wide=engine.evaluate(Parser(f['formula']).parse()) if a.universe in f.get('universes',['csi300','csi500']) else np.full(member.shape,np.nan)
            v=wide[rr,cc];v=np.where(meta.factor_input_valid.to_numpy()&np.isfinite(v)&(abs(v)<=np.finfo('float32').max),v,np.nan).astype('float32');names.append(f['name']);arrays.append(v)
        raw=np.column_stack(arrays);scaled=raw.astype(float)
        for _,positions in meta.groupby('date',sort=False).indices.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);unit=np.nanmedian(abs(scaled[positions]),axis=0);fallback=np.nanmax(abs(scaled[positions]),axis=0)
            unit=np.where(np.isfinite(unit)&(unit>0),unit,np.where(np.isfinite(fallback)&(fallback>0),fallback,1));scaled[positions]/=unit
        processed,_=preprocess(scaled,meta)
        for year in range(a.start_year,a.end_year+1):
            take=meta.date.dt.year.eq(year).to_numpy()
            if not take.any():continue
            for stage,values in [('raw',raw),('processed',processed)]:
                path=output/stage/f'universe={a.universe}/year={year}/part-0.parquet';base=pd.read_parquet(path)
                pd.testing.assert_frame_equal(base[['date','symbol']],meta.loc[take,['date','symbol']].reset_index(drop=True))
                pd.concat([base,pd.DataFrame(values[take],columns=names)],axis=1).to_parquet(path,index=False)
        print('REPORT',start+len(names),flush=True);gc.collect()
    shutil.copy2(REPO/'factors/catalog.json',output/'_factor_catalog.json')
if __name__=='__main__':main()
