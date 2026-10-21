from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from backtest.research import Config,run_backtest,export_report
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--universe',choices=['csi300','csi500'],required=True);p.add_argument('--factor',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=run_backtest(Config(universe=a.universe,factor=a.factor,rebalance_intervals=(1,5,20)),a.root);a.out.mkdir(parents=True,exist_ok=True);export_report(r,a.out)
    print(r['frequency_comparison'].to_string())
if __name__=='__main__':main()
