import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
SEED=20261018
def daily_ic(pred,y):
    d=pd.DataFrame({'p':pred,'y':y}).dropna()
    if not len(d):return pd.Series(dtype=float)
    r=d.groupby(level='date').rank(pct=True)
    r=r-r.groupby(level='date').transform('mean')
    num=(r.p*r.y).groupby(level='date').sum()
    den=np.sqrt((r.p**2).groupby(level='date').sum()*(r.y**2).groupby(level='date').sum())
    n=r.groupby(level='date').size()
    return (num/den.replace(0,np.nan)).where(n>=30)

def make_model(name,candidate,p):
    if name=='ridge':return Ridge(alpha=[100.,1000.][candidate])
    if name=='elastic_net':return ElasticNet(alpha=[.001,.01][candidate],l1_ratio=.5,max_iter=2000,tol=1e-4)
    if name=='pls':return PLSRegression(n_components=min([5,15][candidate],p),scale=False,max_iter=300)
    if name in ['random_forest','extra_trees']:
        cls=RandomForestRegressor if name=='random_forest' else ExtraTreesRegressor
        return cls(n_estimators=100,max_depth=[6,10][candidate],min_samples_leaf=80,max_features=.7,n_jobs=4,random_state=SEED)
    if name=='xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=150,max_depth=[3,5][candidate],learning_rate=.04,min_child_weight=80,subsample=.8,colsample_bytree=.8,reg_lambda=10,n_jobs=4,random_state=SEED,tree_method='hist')
    if name=='lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=150,num_leaves=[7,15][candidate],learning_rate=.04,min_child_samples=100,reg_lambda=10,verbosity=-1,n_jobs=4,random_state=SEED)
    if name=='catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=150,depth=[4,6][candidate],learning_rate=.04,l2_leaf_reg=10,loss_function='RMSE',verbose=False,thread_count=4,random_seed=SEED,allow_writing_files=False)
    raise ValueError(name)

def neural_module(name,n,candidate):
    import torch
    from torch import nn
    if name=='mlp':return nn.Sequential(nn.Linear(2*n,[64,128][candidate]),nn.ReLU(),nn.Dropout(.1),nn.Linear([64,128][candidate],32),nn.ReLU(),nn.Linear(32,1))
    class NumericFT(nn.Module):
        def __init__(self):
            super().__init__();width=[16,32][candidate]
            self.w=nn.Parameter(torch.randn(n,width)*.02);self.b=nn.Parameter(torch.randn(n,width)*.02)
            self.missing=nn.Parameter(torch.randn(n,width)*.02);self.cls=nn.Parameter(torch.randn(1,1,width)*.02)
            layer=nn.TransformerEncoderLayer(width,4,dim_feedforward=width*2,dropout=.1,batch_first=True,norm_first=True,activation='gelu')
            self.blocks=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        def forward(self,x):
            t=x[:,:n,None]*self.w+self.b+x[:,n:,None]*self.missing
            return self.head(self.blocks(torch.cat([self.cls.expand(len(x),-1,-1),t],1))[:,0])
    return NumericFT()

def predict(model,x,neural=False):
    if not neural:return np.asarray(model.predict(x)).ravel()
    import torch
    device=next(model.parameters()).device
    with torch.no_grad():return np.concatenate([model(torch.tensor(a,device=device)).flatten().cpu().numpy() for a in np.array_split(x,max(1,int(np.ceil(len(x)/512))))])
