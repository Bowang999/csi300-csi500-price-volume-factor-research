"""Auditable PDF formula interpreter. No eval, no future filling, strict windows."""
import re
import numpy as np
import pandas as pd
from numba import njit

TOKEN=re.compile(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+|[A-Z_][A-Z_0-9]*|<=|>=|==|!=|\|\||&&|[()+\-*/^<>=?:,&|]')
PREC={'?':1,'||':2,'|':2,'&&':3,'&':3,'=':4,'==':4,'!=':4,'<':4,'>':4,'<=':4,'>=':4,'+':5,'-':5,'*':6,'/':6,'^':8}
class Parser:
    def __init__(self,s):
        self.t=TOKEN.findall(s.upper());self.i=0
        if ''.join(self.t)!=re.sub(r'\s+','',s.upper()):raise ValueError('Unknown formula characters')
    def pop(self):
        t=self.t[self.i];self.i+=1;return t
    def peek(self):return self.t[self.i] if self.i<len(self.t) else ''
    def expect(self,x):
        if self.pop()!=x:raise ValueError('Expected '+x)
    def expr(self,p=0):
        t=self.pop()
        if t in ('+','-'):a=('unary',t,self.expr(7))
        elif t=='(':a=self.expr();self.expect(')')
        elif t[0].isdigit() or t[0]=='.':a=('num',float(t))
        else:
            a=('var',t)
            if self.peek()=='(':
                self.pop();args=[]
                if self.peek()!=')':
                    args.append(self.expr())
                    while self.peek()==',':self.pop();args.append(self.expr())
                self.expect(')');a=('call',t,tuple(args))
        while self.peek() in PREC and PREC[self.peek()]>=p:
            op=self.pop();q=PREC[op]
            if op=='?':
                b=self.expr();self.expect(':');a=('if',a,b,self.expr(q))
            else:a=('binary',op,a,self.expr(q if op=='^' else q+1))
        return a

    def parse(self):
        a=self.expr()
        if self.i!=len(self.t):raise ValueError('Unexpected trailing '+str(self.t[self.i:]))
        return a

def lookback(node):
    """Nominal longest finite-window dependency; SMA retains recursive history."""
    kind=node[0]
    if kind=='num':return 0
    if kind=='var':
        name=node[1]
        if name.startswith('ADV') and name[3:].isdigit():return int(name[3:])
        return 2 if name in ('RET','RETURNS','TR','HD','LD','DTM','DBM') else 1
    if kind=='unary':return lookback(node[2])
    if kind=='binary':return max(lookback(node[2]),lookback(node[3]))
    if kind=='if':return max(map(lookback,node[1:]))
    name=node[1];args=node[2];base=max(map(lookback,args),default=0)
    if name=='SEQUENCE':return int(args[0][1])
    if name=='REGBETA':return lookback(args[0])+(int(args[2][1]) if len(args)==3 else int(args[1][2][0][1]))-1
    if name in ('DELAY','DELTA'):return lookback(args[0])+int(args[1][1])
    if name in ('RANK','SCALE','ABS','LOG','SIGN','SIGNEDPOWER','INDNEUTRALIZE'):return base
    if name in ('MIN','MAX') and not (args[1][0]=='num' and args[1][1]>0):return base
    at=2 if name in ('CORR','CORRELATION','COV','COVIANCE','COVARIANCE') else 1
    if len(args)>at and args[at][0]=='num':return base+int(args[at][1])-1
    return base

@njit(cache=True)
def rolling_special(a,n,mode):
    out=np.full(a.shape,np.nan)
    for j in range(a.shape[1]):
        for t in range(n-1,a.shape[0]):
            v=a[t-n+1:t+1,j]
            if not np.isfinite(v).all():continue
            if mode==0:
                out[t,j]=((v<v[-1]).sum()+((v==v[-1]).sum()+1)/2)/n
            elif mode==1:out[t,j]=np.argmax(v)+1
            elif mode==2:out[t,j]=np.argmin(v)+1
            elif mode==3:out[t,j]=n-1-np.argmax(v)
            elif mode==4:out[t,j]=n-1-np.argmin(v)
            elif mode==5:
                s=0.;w=0.
                for k in range(n):s+=(k+1)*v[k];w+=k+1
                out[t,j]=s/w
            elif mode==6:
                s=0.;w=0.
                for k in range(n):q=.9**(n-1-k);s+=q*v[k];w+=q
                out[t,j]=s/w
            elif mode==7:out[t,j]=np.prod(v)
            elif mode==8:
                mean=v.mean();s=0.;den=0.
                for k in range(n):s+=(k+1-(n+1)/2)*(v[k]-mean);den+=(k+1-(n+1)/2)**2
                if den>0:out[t,j]=s/den
    return out

@njit(cache=True)
def sma(a,n,m):
    out=np.full(a.shape,np.nan)
    for j in range(a.shape[1]):
        prev=np.nan;count=0
        for t in range(a.shape[0]):
            v=a[t,j]
            if not np.isfinite(v):prev=np.nan;count=0;continue
            prev=v if not np.isfinite(prev) else (m*v+(n-m)*prev)/n
            count+=1
            if count>=n:out[t,j]=prev
    return out

def divide(a,b):
    with np.errstate(all='ignore'):return np.where(np.asarray(b)!=0,np.asarray(a)/b,np.nan)

class Engine:
    def __init__(self,fields,member,industry=None):
        self.fields=fields;self.member=member;self.industry=industry;self.shape=member.shape
    def arr(self,x):return np.broadcast_to(x,self.shape)
    def roll(self,x,n,method):
        n=max(1,int(n));return getattr(pd.DataFrame(self.arr(x)).rolling(n,min_periods=n),method)().to_numpy()
    def rank(self,x):return pd.DataFrame(np.where(self.member,self.arr(x),np.nan)).rank(axis=1,pct=True,method='average').to_numpy()
    def lag(self,x,n):
        n=int(n)
        if n<0:raise ValueError('Future lag forbidden')
        a=self.arr(x);out=np.full(self.shape,np.nan)
        if n==0:return a.copy()
        if n<len(a):out[n:]=a[:-n]
        return out
    def evaluate(self,node):
        kind=node[0]
        if kind=='num':return node[1]
        if kind=='var':
            name=node[1]
            if name=='INDCLASS':return 'INDCLASS'
            if name=='SEQUENCE':return ('sequence',None)
            if name.startswith('ADV') and name[3:].isdigit():return self.roll(self.fields['VOLUME'],int(name[3:]),'mean')
            return self.fields[name]
        if kind=='unary':
            x=self.evaluate(node[2]);return -x if node[1]=='-' else x
        if kind=='if':
            c,a,b=(self.evaluate(x) for x in node[1:]);return np.where(np.isfinite(c),np.where(c!=0,a,b),np.nan)
        if kind=='binary':
            op=node[1];a=self.evaluate(node[2]);b=self.evaluate(node[3])
            with np.errstate(all='ignore'):
                if op=='+':v=a+b
                elif op=='-':v=a-b
                elif op=='*':v=a*b
                elif op=='/':v=divide(a,b)
                elif op=='^':v=np.power(a,b)
                else:
                    fun={'<':np.less,'>':np.greater,'<=':np.less_equal,'>=':np.greater_equal,'=':np.equal,'==':np.equal,'!=':np.not_equal,'&':np.logical_and,'&&':np.logical_and,'|':np.logical_or,'||':np.logical_or}[op]
                    v=np.where(np.isfinite(a)&np.isfinite(b),fun(a,b).astype(float),np.nan)
            return np.where(np.isfinite(v),v,np.nan)
        name=node[1];args=[self.evaluate(x) for x in node[2]]
        name={'CORRELATION':'CORR','COVARIANCE':'COV','COVIANCE':'COV','STDDEV':'STD','TS_RANK':'TSRANK','TS_MIN':'TSMIN','TS_MAX':'TSMAX','DECAY_LINEAR':'DECAYLINEAR','PRODUCT':'PROD'}.get(name,name)
        if name=='RANK':
            if len(args)!=1:raise ValueError('RANK arity')
            return self.rank(args[0])
        if name=='SEQUENCE':return ('sequence',int(args[0]))
        if name in ('ABS','LOG','SIGN'):
            if len(args)!=1:raise ValueError(name+' arity')
            with np.errstate(all='ignore'):v={'ABS':np.abs,'LOG':np.log,'SIGN':np.sign}[name](args[0])
            return np.where(np.isfinite(v),v,np.nan)
        if name=='SIGNEDPOWER':
            a,b=args
            with np.errstate(all='ignore'):v=np.sign(a)*np.abs(a)**b
            return np.where(np.isfinite(v),v,np.nan)
        if name=='SCALE':
            x=np.where(self.member,self.arr(args[0]),np.nan);return divide(x,np.nansum(np.abs(x),axis=1)[:,None])*(args[1] if len(args)==2 else 1)
        if name=='INDNEUTRALIZE':
            if self.industry is None:raise ValueError('Missing CITIC industry')
            x=np.where(self.member,self.arr(args[0]),np.nan);out=np.full(self.shape,np.nan)
            for group in np.unique(self.industry):
                if group==0:continue
                mask=(self.industry==group)&np.isfinite(x);cnt=mask.sum(axis=1)
                mu=divide(np.where(mask,x,0).sum(axis=1),cnt)
                out=np.where(mask&(cnt[:,None]>=2),x-mu[:,None],out)
            return out
        if name in ('DELAY','DELTA'):
            a,n=args;v=self.lag(a,n);return v if name=='DELAY' else a-v
        if name in ('MIN','MAX'):
            a,b=args
            if np.isscalar(b) and b>0:return self.roll(a,b,'min' if name=='MIN' else 'max')
            return (np.minimum if name=='MIN' else np.maximum)(a,b)
        if name in ('SUM','MEAN','STD','TSMIN','TSMAX','COUNT'):
            a,n=args;return self.roll(a,n,{'SUM':'sum','MEAN':'mean','STD':'std','TSMIN':'min','TSMAX':'max','COUNT':'sum'}[name])
        if name in ('CORR','COV'):
            a,b,n=args;n=int(n);aa=pd.DataFrame(self.arr(a));bb=pd.DataFrame(self.arr(b));r=aa.rolling(n,min_periods=n)
            v=(r.corr(bb,pairwise=False) if name=='CORR' else r.cov(bb,pairwise=False)).to_numpy()
            # A constant series is not a valid correlation; suppress numerical cancellation.
            if name=='CORR':v=np.where((aa.rolling(n).std().to_numpy()>1e-12)&(bb.rolling(n).std().to_numpy()>1e-12),np.clip(v,-1,1),np.nan)
            return v
        if name in ('TSRANK','TS_ARGMAX','TS_ARGMIN','HIGHDAY','LOWDAY','DECAYLINEAR','WMA','PROD'):
            a,n=args;mode={'TSRANK':0,'TS_ARGMAX':1,'TS_ARGMIN':2,'HIGHDAY':3,'LOWDAY':4,'DECAYLINEAR':5,'WMA':6,'PROD':7}[name]
            return rolling_special(np.asarray(self.arr(a)),int(n),mode)
        if name=='SMA':a,n,m=args;return sma(np.asarray(self.arr(a)),int(n),int(m))
        if name=='SUMIF':
            a,n,c=args;return self.roll(np.where(np.isfinite(c),np.where(c!=0,a,0),np.nan),n,'sum')
        if name=='REGBETA':
            a,b=args[:2]
            if isinstance(b,tuple) and b[0]=='sequence':
                n=int(args[2]) if len(args)==3 else b[1];return rolling_special(np.asarray(self.arr(a)),n,8)
            raise ValueError('Only sequence REGBETA supported without benchmark inputs')
        raise ValueError('Unknown function '+name)
