"""Export normalized private inputs from the original research layout.

Run locally with authorized data. Neither source nor outputs belong in git.
Copies corrected factor columns, not the superseded report-factor values.
"""
from pathlib import Path
import argparse,json,shutil
import numpy as np,pandas as pd,pyarrow.parquet as pq
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=a.source.resolve();out=a.out.resolve()
    if out==r or r in out.parents:raise ValueError('Use a separate private output directory outside the source project')
    e=r/'research/quarterly_top100_models/portfolio_construction/extension_audit';h=e/'history_rebuild_v3'
    for u in ['csi300','csi500']:
        dest=out/f'data/{u}';dest.mkdir(parents=True,exist_ok=True)
        old=pd.read_parquet(r/f'data/{u}/parquet/{u}_daily_market_2010_2024.parquet');old.date=pd.to_datetime(old.date)
        new=pd.read_parquet(r/f'data/{u}/parquet/{u}_daily_market_extension_2025_2026.parquet');new.date=pd.to_datetime(new.date)
        new=new[~((new.symbol.eq('300114')&new.date.ge('2025-02-17'))|(new.symbol.eq('302132')&new.date.lt('2025-02-17')))].copy();new.symbol=new.symbol.replace({'302132':'300114'})
        new['return_quality']=np.where(new.has_price_data,'ok_hfq','missing');old['is_member']=old[f'is_{u}']
        market=pd.concat([old,new],ignore_index=True).sort_values(['date','symbol']);assert not market.duplicated(['date','symbol']).any();market.to_parquet(dest/'market.parquet',index=False)
        ix=pd.read_parquet(r/f'data/{u}/parquet/{u}_index_returns_2010_2024.parquet');ix.date=pd.to_datetime(ix.date)
        ext=pd.read_parquet(e/'index_daily_received.parquet');ext=ext[ext.index_code.eq('000300' if u=='csi300' else '000905')&ext.date.ge('2025-01-01')]
        combined=pd.concat([ix[['date','index_ret']],ext[['date','index_ret']]]);assert not combined.date.duplicated().any();combined.sort_values('date').to_parquet(dest/'index.parquet',index=False)
        # Archived single-factor engine consumes historical filenames, explicitly.
        (dest/'parquet').mkdir(exist_ok=True)
        for suffix in ['daily_market_2010_2024','index_returns_2010_2024']:
            shutil.copy2(r/f'data/{u}/parquet/{u}_{suffix}.parquet',dest/f'parquet/{u}_{suffix}.parquet')
        for year in range(2010,2025):
            d=pd.read_parquet(r/f'factor_store/processed/universe={u}/year={year}/part-0.parquet');d.date=pd.to_datetime(d.date)
            folder=h/u/f'processed/year={year}';meta=pd.read_parquet(folder/'metadata.parquet');meta.date=pd.to_datetime(meta.date)
            pd.testing.assert_frame_equal(d[['date','symbol']],meta[['date','symbol']])
            for b in sorted(folder.glob('block_*.parquet')):
                x=pd.read_parquet(b)
                for c in x:d[c]=x[c].to_numpy()
            path=out/f'factor_store/processed/universe={u}/year={year}/part-0.parquet';path.parent.mkdir(parents=True,exist_ok=True);d.to_parquet(path,index=False)
        folder=r/f'research/oos_2025_2026_15_models/{u}/factors';meta=pd.read_parquet(folder/'metadata.parquet');meta['is_member']=True
        blocks=sorted((e/f'factor_extension/{u}').glob('processed_block*'))+sorted(folder.glob('processed_block*'))
        names=[];frames=[meta]
        for b in blocks:
            x=pd.read_parquet(b);assert len(x)==len(meta);names.extend(x.columns);frames.append(x)
        assert len(names)==len(set(names))==857
        d=pd.concat(frames,axis=1)
        for year,g in d.groupby(d.date.dt.year):
            path=out/f'factor_store/processed/universe={u}/year={year}/part-0.parquet';path.parent.mkdir(parents=True,exist_ok=True);g.to_parquet(path,index=False)
        print(u,'private inputs prepared',flush=True)
    shutil.copy2(r/'factor_store/_factor_catalog.json',out/'factor_store/_factor_catalog.json')
if __name__=='__main__':main()
