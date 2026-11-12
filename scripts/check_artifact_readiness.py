#!/usr/bin/env python3
"""Inspect, without loading pickles, whether article replay input files exist."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lorenzetti_repro.legacy_artifacts import catalog,ARTICLE_METHODS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder',default=str(ROOT/'artifacts/pickles'))
    args=p.parse_args()
    rows=catalog(args.folder,checksum=False)
    by_method={}
    for r in rows:
        if r['method']:
            by_method.setdefault(r['method'],set()).add(r['kind'])
    print(f'{len(rows)} pickle files recognized in {args.folder}')
    for art,methods in ARTICLE_METHODS.items():
        print(f'\n[{art}]')
        any_data=False
        for m in methods:
            available= 'noSEG_Full_DF' in by_method.get(m,set())
            any_data |= available
            print(f'  {m:7s} {"FULL_DF present" if available else "not provided"}')
        if not any_data:
            print('  No event-level data for this article: only reference tables can display')
    print('\nNote: filenames alone do not prove that the winning configuration is inside any file.')
if __name__=='__main__':main()
