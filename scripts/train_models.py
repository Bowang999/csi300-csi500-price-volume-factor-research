"""Prediction only: quarterly training, IC/RankIC and native rank R2."""
from pathlib import Path
import argparse,sys,hashlib,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import model_pipeline as m

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--universe',choices=['csi300','csi500'],required=True)
    p.add_argument('--start',default='2015Q1');p.add_argument('--end',default='2024Q4');p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    m.ROOT=a.root.resolve();m.OUT=a.out.resolve();m.U=a.universe;m.OUT.mkdir(parents=True,exist_ok=True)
    paths=list((m.ROOT/f'factor_store/processed/universe={m.U}').rglob('*.parquet'))+list((m.ROOT/f'data/{m.U}').glob('*.parquet'))+list((Path(__file__).resolve().parents[1]/'src').glob('*.py'))
    signature={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths};manifest=m.OUT/'input_hashes.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('Inputs changed: select a fresh output directory')
    m.atomic_json(manifest,signature)
    dates,raw,exits,ctx=m.labels();qs=dates.to_period('Q').unique();qs=qs[(qs>=pd.Period(a.start))&(qs<=pd.Period(a.end))]
    if a.pilot:qs=qs[:1]
    if not len(qs):raise ValueError('No prediction dates in range')
    first=max(2012,qs[0].year-3);stats=m.build_cache(raw,range(first,qs[-1].year+1))
    for q in qs:m.quarter(q,'full',dates,raw,exits,ctx,stats)
    predictions=pd.concat([pd.concat([pd.read_parquet(m.OUT/'full'/str(q)/(name+'.parquet')) for name in m.MODELS],axis=1) for q in qs]).sort_index()
    assert not predictions.index.duplicated().any() and predictions.notna().all().all()
    predictions.to_parquet(m.OUT/'predictions.parquet');daily=m.daily_metrics(predictions,raw);daily.to_parquet(m.OUT/'daily_ic.parquet')
    y=m.target(raw,predictions.index);rows=[]
    for name in m.MODELS:
        ok=y.notna();err=(predictions.loc[ok,name]-y[ok])**2;den=y[ok]**2
        rows.append({'model':name,'IC':daily[name,'IC'].mean(),'RankIC':daily[name,'RankIC'].mean(),'rank_R2':1-err.sum()/den.sum(),'valid_days':daily[name,'RankIC'].notna().sum()})
    table=pd.DataFrame(rows).sort_values('RankIC',ascending=False);table.to_csv(m.OUT/'comparison.csv',index=False)
    m.atomic_json(m.OUT/'status.json',{'state':'completed','quarters':len(qs),'models':15,'portfolio_returns':False});print(table.to_string(index=False))
if __name__=='__main__':main()
