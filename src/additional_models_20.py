"""Twenty extra model configurations; compact adaptations, not novelty claims."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lars, Ridge
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import torch
from torch import nn

SEED=20261023
CLASSICAL=['ordinary_least_squares','least_angle_regression','pca_knn',
    'polynomial_sketch_ridge','extreme_learning_machine','boosted_stumps',
    'huber_gradient_boosting','median_hist_boosting','catboost_mae','lightgbm_dart']
NEURAL=['factorization_machine','deep_fm_numeric','cross_network','deep_cross_network',
    'gated_mlp','highway_network','neural_additive','low_rank_bilinear',
    'supervised_autoencoder','soft_threshold_network']
MODELS=CLASSICAL+NEURAL

class RandomTanh(BaseEstimator,TransformerMixin):
    def __init__(self,width=128,scale=1.):self.width=width;self.scale=scale
    def fit(self,x,y=None):
        rng=np.random.default_rng(SEED)
        self.w_=rng.normal(size=(x.shape[1],self.width))*self.scale/np.sqrt(x.shape[1])
        self.b_=rng.normal(size=self.width)*.1
        return self
    def transform(self,x):return np.tanh(x@self.w_+self.b_)

def classical(name,c,p):
    if name=='ordinary_least_squares':return LinearRegression(fit_intercept=[False,True][c])
    if name=='least_angle_regression':return Lars(n_nonzero_coefs=[10,30][c])
    if name=='pca_knn':return make_pipeline(PCA(n_components=20,svd_solver='full'),KNeighborsRegressor(n_neighbors=[50,200][c],weights='distance',n_jobs=2))
    if name=='polynomial_sketch_ridge':return make_pipeline(PolynomialCountSketch(degree=[2,3][c],gamma=1/p,coef0=1.,n_components=256,random_state=SEED),Ridge(alpha=100.))
    if name=='extreme_learning_machine':return make_pipeline(RandomTanh(width=[128,256][c]),Ridge(alpha=100.))
    if name=='boosted_stumps':return HistGradientBoostingRegressor(max_leaf_nodes=2,max_iter=[100,200][c],learning_rate=.04,min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=SEED)
    if name=='huber_gradient_boosting':return GradientBoostingRegressor(loss='huber',n_estimators=80,max_depth=[2,3][c],learning_rate=.04,min_samples_leaf=100,max_features=.7,subsample=.8,random_state=SEED)
    if name=='median_hist_boosting':return HistGradientBoostingRegressor(loss='quantile',quantile=.5,max_leaf_nodes=[7,15][c],max_iter=150,learning_rate=.04,min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=SEED)
    if name=='catboost_mae':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function='MAE',iterations=150,depth=[4,6][c],learning_rate=.04,l2_leaf_reg=10,thread_count=2,random_seed=SEED,verbose=False,allow_writing_files=False)
    if name=='lightgbm_dart':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(boosting_type='dart',n_estimators=150,num_leaves=[7,15][c],learning_rate=.04,min_child_samples=100,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=SEED)
    raise ValueError(name)

class NumericNet(nn.Module):
    def __init__(self,name,p,c):
        super().__init__();self.name=name;self.p=p;w=[32,64][c];k=[8,16][c]
        self.linear=nn.Linear(p,1)
        if name in ['factorization_machine','deep_fm_numeric']:
            self.v=nn.Parameter(torch.randn(p,k)*.01)
            if name=='deep_fm_numeric':self.deep=nn.Sequential(nn.Linear(p,w),nn.ReLU(),nn.Linear(w,1))
        elif name in ['cross_network','deep_cross_network']:
            self.cross=nn.ModuleList([nn.Linear(p,1,bias=False) for _ in range([2,3][c])]);self.bias=nn.ParameterList([nn.Parameter(torch.zeros(p)) for _ in self.cross])
            if name=='deep_cross_network':self.deep=nn.Sequential(nn.Linear(p,w),nn.ReLU(),nn.Linear(w,1))
        elif name=='neural_additive':
            self.aw=nn.Parameter(torch.randn(p,k)*.1);self.ab=nn.Parameter(torch.zeros(p,k));self.av=nn.Parameter(torch.randn(p,k)*.01)
        elif name=='low_rank_bilinear':
            self.left=nn.Linear(p,k,bias=False);self.right=nn.Linear(p,k,bias=False);self.head=nn.Linear(k,1)
        else:
            self.input=nn.Linear(p,w);self.head=nn.Linear(w,1)
            if name=='gated_mlp':self.gate=nn.Linear(p,w)
            if name=='highway_network':
                self.blocks=nn.ModuleList([nn.Linear(w,w) for _ in range(2)]);self.gates=nn.ModuleList([nn.Linear(w,w) for _ in range(2)])
            if name=='supervised_autoencoder':self.decoder=nn.Linear(w,p)
            if name=='soft_threshold_network':self.threshold=nn.Parameter(torch.full((w,),-2.))
    def forward(self,x):
        n=self.name
        if n in ['factorization_machine','deep_fm_numeric']:
            fm=.5*((x@self.v)**2-(x*x)@(self.v*self.v)).sum(dim=1,keepdim=True)
            z=self.linear(x)+fm
            return z+(self.deep(x) if n=='deep_fm_numeric' else 0)
        if n in ['cross_network','deep_cross_network']:
            h=x
            for layer,b in zip(self.cross,self.bias):h=h+x*layer(h)+b
            return self.linear(h)+(self.deep(x) if n=='deep_cross_network' else 0)
        if n=='neural_additive':return (torch.tanh(x[:,:,None]*self.aw+self.ab)*self.av).sum(dim=(1,2))[:,None]+self.linear(x)
        if n=='low_rank_bilinear':return self.linear(x)+self.head(self.left(x)*self.right(x))
        h=self.input(x)
        if n=='gated_mlp':h=torch.nn.functional.silu(h)*torch.sigmoid(self.gate(x))
        elif n=='soft_threshold_network':h=torch.sign(h)*torch.relu(h.abs()-torch.nn.functional.softplus(self.threshold))
        else:h=torch.relu(h)
        if n=='highway_network':
            for block,gate in zip(self.blocks,self.gates):
                t=torch.sigmoid(gate(h));h=t*torch.relu(block(h))+(1-t)*h
        return self.head(h)

class TorchRegressor:
    def __init__(self,name,c):self.name=name;self.c=c
    def fit(self,x,y):
        torch.set_num_threads(2);torch.manual_seed(SEED)
        self.net=NumericNet(self.name,x.shape[1],self.c)
        opt=torch.optim.AdamW(self.net.parameters(),lr=.001,weight_decay=.01)
        a=torch.as_tensor(x,dtype=torch.float32);b=torch.as_tensor(y,dtype=torch.float32)
        for _ in range(8):
            self.net.train()
            for ix in torch.randperm(len(a)).split(512):
                xx=a[ix];loss=((self.net(xx).flatten()-b[ix])**2).mean()
                if self.name=='supervised_autoencoder':
                    rec=self.net.decoder(torch.relu(self.net.input(xx)))
                    loss=loss+.05*((rec-xx)**2).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite neural training loss')
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.);opt.step()
        self.net.eval();return self
    def predict(self,x):
        with torch.no_grad():return np.concatenate([self.net(torch.as_tensor(z,dtype=torch.float32)).flatten().numpy() for z in np.array_split(x,max(1,int(np.ceil(len(x)/512))))])

def fit(name,c,x,y):
    model=TorchRegressor(name,c) if name in NEURAL else classical(name,c,x.shape[1])
    model.fit(x,y);return model

def predict(name,model,x):return np.asarray(model.predict(x)).ravel()
