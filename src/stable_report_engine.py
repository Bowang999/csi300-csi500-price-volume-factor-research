"""Isolated candidate: window-local covariance, no changes to production factors."""
import sys
from pathlib import Path
import numpy as np
from numba import njit
ROOT = Path(__file__).resolve().parents[1]

from alpha_report_engine import Engine

@njit(cache=True)
def local_covariance(a, b, n, correlation=False):
    out = np.full(a.shape, np.nan)
    if n < 2:
        return out
    for j in range(a.shape[1]):
        for t in range(n-1, a.shape[0]):
            x = a[t-n+1:t+1, j]
            y = b[t-n+1:t+1, j]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            # Subtract a local anchor before centering to reduce cancellation.
            xx = x-x[0]
            yy = y-y[0]
            xx = xx-xx.mean()
            yy = yy-yy.mean()
            cross = (xx*yy).sum()
            if correlation:
                sx = (xx*xx).sum()
                sy = (yy*yy).sum()
                if sx > 0 and sy > 0:
                    out[t,j] = min(1., max(-1., cross / np.sqrt(sx*sy)))
            else:
                out[t,j] = cross/(n-1)
    return out

class StableEngine(Engine):
    def evaluate(self, node):
        if node[0] == "call" and node[1] in (
            "CORR", "CORRELATION", "COV", "COVIANCE", "COVARIANCE"
        ):
            a, b, n = [self.evaluate(x) for x in node[2]]
            return local_covariance(
                np.asarray(self.arr(a), dtype=float),
                np.asarray(self.arr(b), dtype=float),
                int(n), node[1] in ("CORR", "CORRELATION")
            )
        return super().evaluate(node)

@njit(cache=True)
def local_roll(a,n,mode):
    out=np.full(a.shape,np.nan)
    for j in range(a.shape[1]):
        for t in range(n-1,a.shape[0]):
            x=a[t-n+1:t+1,j]
            if not np.isfinite(x).all():continue
            if mode==0:out[t,j]=x.sum()
            elif mode==1:out[t,j]=x.mean()
            elif mode==2:
                if n>1:
                    z=x-x[0];z=z-z.mean()
                    out[t,j]=np.sqrt((z*z).sum()/(n-1))
            elif mode==3:out[t,j]=x.min()
            elif mode==4:out[t,j]=x.max()
    return out

class WindowLocalEngine(StableEngine):
    """Second candidate: also remove rolling sum/mean/std accumulator history."""
    def roll(self,x,n,method):
        modes={"sum":0,"mean":1,"std":2,"min":3,"max":4}
        return local_roll(np.asarray(self.arr(x),dtype=float),max(1,int(n)),modes[method])

class ExactTwoPointEngine(WindowLocalEngine):
    """V3: preserve mathematical ties for two-observation correlations."""
    def evaluate(self,node):
        if node[0]=="call" and node[1] in ("CORR","CORRELATION"):
            a,b,n=[self.evaluate(x) for x in node[2]]
            if int(n)==2:
                x=np.asarray(self.arr(a),dtype=float)
                y=np.asarray(self.arr(b),dtype=float)
                dx=x[1:]-x[:-1];dy=y[1:]-y[:-1]
                valid=(np.isfinite(x[1:])&np.isfinite(x[:-1])&
                    np.isfinite(y[1:])&np.isfinite(y[:-1])&(dx!=0)&(dy!=0))
                out=np.full(self.shape,np.nan)
                out[1:]=np.where(valid,np.sign(dx)*np.sign(dy),np.nan)
                return out
        return super().evaluate(node)
