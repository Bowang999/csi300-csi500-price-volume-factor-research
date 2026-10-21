import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,gc,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import model_helpers as old
from metrics import atomic_json,hac_mean_stats
SEED=20261019
MODELS=['equal_weight','ridge','lasso','elastic_net','pca_ridge','pls','random_forest','xgboost','lightgbm','catboost','mlp','resnet','tabnet','ft_transformer','moe']
NEURAL=['mlp','resnet','ft_transformer','moe']
OUT=None
ROOT=None
U=None
def source(year,cols=None):
    d=pd.read_parquet(ROOT/f'factor_store/processed/universe={U}/year={year}/part-0.parquet',columns=cols)
    d.date=pd.to_datetime(d.date)
    return d.loc[d.is_member.fillna(False)&d.neutralization_eligible.fillna(False)].set_index(['date','symbol']).sort_index()
def load(dates,names):
    cols=['date','symbol','is_member','neutralization_eligible']+list(names)
    a=pd.concat([source(y,cols)[names] for y in sorted(set(dates.year))])
    return a.loc[a.index.get_level_values('date').isin(dates)].replace([np.inf,-np.inf],np.nan).astype('float32')
def labels():
    d=pd.read_parquet(ROOT/f'data/{U}/index.parquet');d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index();dates=d.index
    z=pd.read_parquet(ROOT/f'data/{U}/market.parquet');z.date=pd.to_datetime(z.date)
    v=z.pivot(index='date',columns='symbol',values='vwap_hfq').reindex(dates)
    good=z.pivot(index='date',columns='symbol',values='return_quality').reindex(dates).eq('ok_hfq')
    v=v.where(good&v.gt(0));raw=(v.shift(-2)/v.shift(-1)-1).stack(future_stack=True)
    ctx=pd.DataFrame({'market_momentum20':np.expm1(np.log1p(d.index_ret).rolling(20).sum()),'market_volatility20':d.index_ret.rolling(20).std()})
    return dates,raw,pd.Series(dates,index=dates).shift(-2),ctx
def build_cache(raw,years):
    import pyarrow.parquet as pq
    for year in years:
        path=OUT/'cache'/f'daily_stats_{year}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():continue
        d=source(year);names=[n for n in d if n.startswith(('PV_','Q158_','HT101_','GTJA191_'))]
        x=d[names].replace([np.inf,-np.inf],np.nan);y=raw.reindex(x.index);ics=[];cov=[];ds=[]
        for date,z in x.groupby(level='date',sort=True):
            a=z.to_numpy(float);b=y.reindex(z.index).to_numpy(float)[:,None];valid=np.isfinite(a)&np.isfinite(b);n=valid.sum(axis=0)
            aa=np.where(valid,a,0);bb=np.where(valid,b,0);safe=np.maximum(n,1)
            num=(aa*bb).sum(axis=0)-aa.sum(axis=0)*bb.sum(axis=0)/safe
            den=np.sqrt(np.maximum((aa*aa).sum(axis=0)-aa.sum(axis=0)**2/safe,0)*np.maximum((bb*bb).sum(axis=0)-bb.sum(axis=0)**2/safe,0))
            ic=np.divide(num,den,out=np.full(len(names),np.nan),where=(n>=30)&(den>1e-12))
            ds.append(date);ics.append(ic);cov.append(np.isfinite(a).mean(axis=0))
        pd.concat({'IC':pd.DataFrame(ics,index=pd.DatetimeIndex(ds,name='date'),columns=names),'coverage':pd.DataFrame(cov,index=pd.DatetimeIndex(ds,name='date'),columns=names)},axis=1).to_parquet(path)
    return pd.concat([pd.read_parquet(OUT/'cache'/f'daily_stats_{y}.parquet') for y in years]).sort_index()
def select(stats,train_dates):
 s=stats.reindex(train_dates);means=s['IC'].mean();coverage=s['coverage'].mean();count=s['IC'].notna().sum()
 good=means.notna()&coverage.ge(.8)&count.ge(.8*len(train_dates));order=means[good].abs().sort_values(ascending=False,kind='stable').index.tolist()
 if len(order)<100:raise ValueError(f'Only {len(order)} quality-qualified factors; no silent fill with invalid features')
 sampled=train_dates[::max(1,len(train_dates)//60)];z=load(sampled,order)
 total=np.zeros((len(order),len(order)));n=total.copy()
 for _,g in z.groupby(level='date'):
  a=g.rank(pct=True).to_numpy(float);valid=np.isfinite(a).astype(float);a=np.nan_to_num(a)
  nn=valid.T@valid;safe=np.maximum(nn,1);sums=a.T@valid;sq=(a*a).T@valid
  numer=a.T@a-sums*sums.T/safe;var=np.maximum(sq-sums*sums/safe,0);den=np.sqrt(var*var.T);ok=(nn>=30)&(den>1e-10)
  total+=np.divide(numer,den,out=np.zeros_like(total),where=ok);n+=ok
 corr=np.abs(np.divide(total,n,out=np.zeros_like(total),where=n>0));chosen=[];skipped=[]
 for j in range(len(order)):
  if chosen and np.any(corr[j,chosen]>=.9):skipped.append(j);continue
  chosen.append(j)
  if len(chosen)==100:break
 backfill=[]
 if len(chosen)<100:
  for j in range(len(order)):
   if j not in chosen:chosen.append(j);backfill.append(order[j])
   if len(chosen)==100:break
 names=[order[j] for j in chosen];sign=np.sign(means.reindex(names)).to_numpy();sign[sign==0]=1
 return names,sign,{'selected':names,'direction':dict(zip(names,sign.tolist())),'mean_IC':means.reindex(names).to_dict(),'valid_days':count.reindex(names).to_dict(),'correlated_backfill':backfill,'quality_candidates':len(order),'first_date':str(train_dates.min()),'last_signal_date':str(train_dates.max())}

def target(raw,index):
 y=raw.reindex(index);return (y.groupby(level='date').rank(pct=True)*2-1).where(y.groupby(level='date').transform('count')>=30)

def preparation(a,context):
 x=a.to_numpy();med=np.nanmedian(x,axis=0)
 if not np.isfinite(med).all():raise ValueError('Empty selected factor')
 scale=StandardScaler().fit(np.where(np.isfinite(x),x,med));cs=StandardScaler().fit(context)
 return med,scale,cs

def transform(a,context,p):
 x=a.to_numpy();miss=~np.isfinite(x);med,sc,cs=p
 return np.column_stack([sc.transform(np.where(miss,med,x)),miss.astype(float),cs.transform(context)]).astype('float32')

def model_module(name,candidate):
 import torch
 from torch import nn
 width=[32,64][candidate]
 if name=='mlp':return nn.Sequential(nn.Linear(202,width),nn.ReLU(),nn.Linear(width,32),nn.ReLU(),nn.Linear(32,1))
 if name=='ft_transformer':
  class FT(nn.Module):
   def __init__(self):
    super().__init__();self.base=old.neural_module('ft_transformer',100,candidate);self.context=nn.Linear(2,1)
   def forward(self,x):return self.base(x[:,:200])+self.context(x[:,200:])
  return FT()
 if name=='resnet':
  class Res(nn.Module):
   def __init__(self):
    super().__init__();self.input=nn.Linear(202,width);self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.ReLU(),nn.Dropout(.1),nn.Linear(width*2,width)) for _ in range(2)]);self.out=nn.Linear(width,1)
   def forward(self,x):
    h=self.input(x)
    for b in self.blocks:h=h+b(h)
    return self.out(h)
  return Res()
 class MoE(nn.Module):
  def __init__(self):
   super().__init__();self.experts=nn.ModuleList([nn.Sequential(nn.Linear(202,width),nn.ReLU(),nn.Linear(width,1)) for _ in range(3)]);self.gate=nn.Sequential(nn.Linear(2,8),nn.Tanh(),nn.Linear(8,3))
  def forward(self,x):
   w=torch.softmax(self.gate(x[:,200:]),dim=1);e=torch.cat([m(x) for m in self.experts],dim=1);return (w*e).sum(dim=1,keepdim=True)
 return MoE()

def fit(name,c,x,y):
 if name in NEURAL:
  import torch
  torch.set_num_threads(2);torch.manual_seed(SEED);device='cuda' if torch.cuda.is_available() else 'cpu';m=model_module(name,c).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
  a=torch.tensor(x,device=device);b=torch.tensor(y,dtype=torch.float32,device=device)
  for epoch in range(8):
   m.train()
   for ids in torch.randperm(len(a),device=device).split(512):
    opt.zero_grad();loss=((m(a[ids]).flatten()-b[ids])**2).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  m.eval();return m
 if name=='tabnet':
  from pytorch_tabnet.tab_model import TabNetRegressor
  import torch
  torch.set_num_threads(2)
  m=TabNetRegressor(n_d=[8,16][c],n_a=[8,16][c],n_steps=3,seed=SEED,verbose=0,device_name='auto')
  m.fit(x,y.astype('float32').reshape(-1,1),max_epochs=8,patience=0,batch_size=1024,virtual_batch_size=128,num_workers=0,drop_last=False);return m
 if name=='lasso':m=Lasso(alpha=[.001,.01][c],max_iter=3000)
 elif name=='pca_ridge':m=make_pipeline(PCA(n_components=[20,50][c],svd_solver='full'),Ridge(alpha=100))
 else:m=old.make_model(name,c,x.shape[1])
 m.fit(x,y);return m

def predict(name,m,x):return old.predict(m,x,name in NEURAL)

def quarter(q,mode,dates,raw,exits,context,stats):
 folder=OUT/mode/str(q);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'completed.json').exists():return
 cutoff=dates[dates.to_period('Q')==q][0];td=dates[(dates>=cutoff-pd.DateOffset(years=3))&exits.lt(cutoff)];valstart=td[int(len(td)*.8)]
 inner=td[(td<valstart)&exits.reindex(td).lt(valstart)];vd=td[td>=valstart];testdates=dates[dates.to_period('Q')==q]
 ni,si,ai=select(stats,inner);nf,sf,af=select(stats,td)
 assert len(ni)==len(nf)==100
 ai['latest_label_exit']=str(exits.reindex(inner).max());af['latest_label_exit']=str(exits.reindex(td).max());assert exits.reindex(inner).max()<valstart and exits.reindex(td).max()<cutoff
 atomic_json(folder/'inner_selection.json',ai);atomic_json(folder/'final_selection.json',af)
 def data(ds,names,sign):
  a=load(ds,names).mul(sign,axis=1);a=a.loc[a.notna().sum(axis=1)>=50];y=target(raw,a.index)
  ctx=context.reindex(a.index.get_level_values('date')).to_numpy();assert np.isfinite(ctx).all();return a,y,ctx
 a,ya,ca=data(inner[::5],ni,si);v,yv,cv=data(vd,ni,si);f,yf,cf=data(td[::5],nf,sf);t,yt,ct=data(testdates,nf,sf)
 # Label exclusion only for training rows, never test inference rows.
 mask=ya.notna();a=a.loc[mask];ca=ca[mask];ya=ya.loc[mask]
 mask=yf.notna();f=f.loc[mask];cf=cf[mask];yf=yf.loc[mask]
 pi=preparation(a,ca);pf=preparation(f,cf)
 xi=transform(a,ca,pi);xv=transform(v,cv,pi);xf=transform(f,cf,pf);xt=transform(t,ct,pf)
 joblib.dump({'inner':pi,'final':pf,'inner_names':ni,'final_names':nf},folder/'preprocessing.joblib')
 # Fixed100 equal weights: same train median imputation; no available-count reweighting.
 filled=np.where(np.isfinite(t.to_numpy()),t.to_numpy(),pf[0]);base=filled.mean(axis=1)
 pd.DataFrame({'equal_weight':base},index=t.index).to_parquet(folder/'equal_weight.parquet')
 for name in MODELS[1:]:
  if (folder/(name+'.json')).exists() and (folder/(name+'.parquet')).exists():continue
  atomic_json(OUT/'status.json',{'state':'running','mode':mode,'quarter':str(q),'model':name});start=time.time();scores=[]
  for c in range(2):
   m=fit(name,c,xi,ya.to_numpy());z=pd.Series(predict(name,m,xv),index=v.index);score=float(old.daily_ic(z,raw.reindex(v.index)).mean());scores.append(score);del m;gc.collect()
  if not np.isfinite(scores).all():raise ValueError('Nonfinite validation RankIC')
  best=int(np.argmax(scores));m=fit(name,best,xf,yf.to_numpy());z=predict(name,m,xt);assert np.isfinite(z).all()
  if name in NEURAL:
   import torch
   torch.save({'state_dict':m.cpu().state_dict(),'name':name,'candidate':best},folder/(name+'.pt'))
  elif name=='tabnet':m.save_model(str(folder/'tabnet_model'))
  else:joblib.dump(m,folder/(name+'.joblib'))
  pd.DataFrame({name:z},index=t.index).to_parquet(folder/(name+'.parquet'))
  atomic_json(folder/(name+'.json'),{'candidate':best,'inner_validation_RankIC':scores,'seconds':time.time()-start});print('FIT',mode,q,name,round(time.time()-start,1),flush=True);del m;gc.collect()
 atomic_json(folder/'completed.json',{'methods':15,'factors':100,'prediction_rows':len(t),'cutoff':str(cutoff),'validation_start':str(valstart),'inner_train_rows':len(a),'final_train_rows':len(f)})

def daily_metrics(pred,raw):
 rows={}
 for name in pred:
  d=pd.DataFrame({'p':pred[name],'y':raw.reindex(pred.index)}).dropna();g=d.groupby(level='date');count=g.size()
  centered=d-g.transform('mean');num=(centered.p*centered.y).groupby(level='date').sum();den=np.sqrt((centered.p**2).groupby(level='date').sum()*(centered.y**2).groupby(level='date').sum())
  rows[(name,'IC')]=(num/den.replace(0,np.nan)).where(count>=30);rows[(name,'RankIC')]=old.daily_ic(pred[name],raw)
 return pd.DataFrame(rows)

def tests():
 import torch
 torch.set_num_threads(2)
 for name in NEURAL:
  m=model_module(name,0);x=torch.randn(8,202);z=m(x);assert z.shape==(8,1);z.mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 ix=pd.MultiIndex.from_product([pd.bdate_range('2020-01-01',periods=3),range(40)],names=['date','symbol']);r=pd.Series(np.tile(np.arange(40),3),index=ix);d=daily_metrics(pd.DataFrame({'test':r}),r);assert np.allclose(d,1)
 return {'model_dimensions_and_gradients':True,'IC_and_RankIC_perfect_order':True}
