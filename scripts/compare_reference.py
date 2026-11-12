#!/usr/bin/env python3
"""Read-only comparison of NEW run with source article numbers (never changes them)."""
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
FILES={'1':'stage1_tns_table7.csv','2':'stage2_measurement_table12.csv','3':'stage3_information_fusion_table17.csv'}

def row_label(m,rep):
    name=m.upper() if m!='bdt-bin' else 'BDT Bin'
    if m=='bdt-bin':prefix='BDT Bin'
    else:prefix=('DNN' if m=='dnn' else 'GNN' if m=='gnn' else 'RFBDT' if m=='rfbdt' else 'BDT' if m=='bdt' else name)+' no-Bin'
    if rep.lower() in ('rings100','full','rings[1:100]'):return prefix+': Rings[1:100]'
    if 'THR' in rep.upper():
        part=rep.upper().replace('_THR_','-').replace('-THR-','-')
        return prefix+': '+part
    return prefix+': '+rep.upper()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('manifest',type=Path);a=ap.parse_args()
    m=json.loads(a.manifest.read_text(encoding='utf-8'))
    stage=str(m['stage']);model=m['model'];rep=m['representation']['name']
    label=row_label(model,rep)
    csv=ROOT/'results'/'reference'/FILES[stage]
    data=pd.read_csv(csv)
    result=data[data['configuration'].str.casefold()==label.casefold()]
    if result.empty:
        raise SystemExit('This configuration has no matching central published row in '+FILES[stage]+'. This is not an error in the new run.')
    published=result.iloc[0]
    metrics=m['metrics_from_current_run']
    factor=1000.0 if stage=='1' else 1.0
    for q in ('rmse','iqr'):
        key=q+('_gev' if stage=='1' else '_mev')
        historical=float(published[key]);new=float(metrics[q])/factor
        print(f'{label} {q}: published={historical:.5f}; NEW rerun={new:.5f}; difference={new-historical:+.5f}')
    print('REPRODUCTION STATUS: comparison only, NOT verified numerical/historical equivalence. Data/splits/environment must be independently checked.')
if __name__=='__main__':main()
