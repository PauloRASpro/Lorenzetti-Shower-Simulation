#!/usr/bin/env python3
"""Diagnostic plots of NEW predictions; not numerical replicas of thesis Figures 35–45."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('--out',type=Path,default=Path('results/local/plots'));a=p.parse_args()
d=np.load(a.predictions);et=d['et_true'];ef=d['et_fast'];ec=ef*d['alpha_pred'];a.out.mkdir(parents=True,exist_ok=True)
for name,arr in [('fast',et-ef),('calibrated',et-ec)]:
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(arr,bins=80,histtype='step',label=name);ax.set_xlabel('Etrue - Eestimate (native energy units)');ax.set_ylabel('Events');ax.legend();fig.tight_layout();fig.savefig(a.out/f'{name}_residual_hist.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5));ax.scatter(et,ec,s=.5,alpha=.25);minv=min(et.min(),ec.min());maxv=max(et.max(),ec.max());ax.plot([minv,maxv],[minv,maxv],ls='--');ax.set_xlabel('Etrue');ax.set_ylabel('Ecalib');fig.tight_layout();fig.savefig(a.out/'new_e_calib_vs_true.png',dpi=160);plt.close(fig)
print('These are NEW diagnostic plots, not thesis figure reproductions. Saved to',a.out)
