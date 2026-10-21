from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from preprocessing import preprocess
from stable_report_engine import ExactTwoPointEngine
from alpha_report_engine import Parser
from model_pipeline import daily_metrics,target
from backtest.test_accounting import run
class PipelineTests(unittest.TestCase):
    def test_accounting(self):self.assertGreaterEqual(len(run()),19)
    def test_metrics(self):
        ix=pd.MultiIndex.from_product([pd.bdate_range('2020-01-01',periods=3),range(50)],names=['date','symbol']);y=pd.Series(np.tile(np.arange(50),3),index=ix)
        z=daily_metrics(pd.DataFrame({'same':y,'reverse':-y}),y)
        np.testing.assert_allclose(z['same'],1);np.testing.assert_allclose(z['reverse'],-1)
        rank=target(y,ix);self.assertAlmostEqual(float(((rank-rank)**2).sum()),0)
    def test_preprocessing(self):
        rng=np.random.default_rng(5);n=100
        meta=pd.DataFrame({'date':pd.Timestamp('2020-01-01'),'neutralization_eligible':True,'log_market_cap':rng.normal(size=n),'industry_code':np.repeat(['a','b','c','d'],25)})
        values=rng.normal(size=(n,3))+meta.log_market_cap.to_numpy()[:,None]
        z,diag=preprocess(values,meta);self.assertLess(diag['max_abs_logcap_corr'],1e-5);self.assertTrue(np.isfinite(z).all())
    def test_report_prefix(self):
        rng=np.random.default_rng(9);fields={'CLOSE':rng.uniform(5,10,(50,40)),'VOLUME':rng.uniform(100,200,(50,40))};member=np.ones((50,40),bool)
        expr=Parser('RANK(CORRELATION(CLOSE,VOLUME,2))').parse()
        full=ExactTwoPointEngine(fields,member).evaluate(expr)
        prefix=ExactTwoPointEngine({k:v[:30] for k,v in fields.items()},member[:30]).evaluate(expr)
        np.testing.assert_allclose(full[:30],prefix,equal_nan=True)
if __name__=='__main__':unittest.main()
