"""Input-data diagnostic figures using the plotting instructions of notebook cells 118 and 157.

Only the plotting statements, not screenshots, are carried into this module.
The published benchmark figures require the original Lorenzetti CSV.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def draw_benchmark_diagnostics(dataset_csv, output_dir, label='Lorenzetti Showers Framework'):
    full_data=pd.read_csv(dataset_csv,usecols=['mc_match_et','cluster_truthet'])
    if full_data.empty:raise ValueError('Lorenzetti input dataset has no events')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    paths=[]
    # Original notebook cell 157: the author's 8x7 step histograms.
    fig=plt.figure(figsize=(8,7))
    plt.hist(full_data['mc_match_et'],bins=100,log=False,histtype='step',label=r'$E_{true}$ Monte Carlo')
    plt.hist(full_data['cluster_truthet'],bins=100,log=True,histtype='step',label=r'$E_{fast}$ Standard FastCalo')
    plt.ylabel('Count',fontsize=16);plt.xlabel('Energy (MeV)',fontsize=16)
    plt.xticks(rotation=0,fontsize=12)
    plt.legend(loc='upper right',frameon=False,fontsize=12)
    ax=plt.subplot();xmin,xmax,ymin,ymax=plt.axis();dx=(xmax-xmin)/20;dy=(ymax-ymin)/20
    ax.text(xmin+6*dx,ymax-5*dy,label)
    f=out/'original_energy_histogram.png';fig.savefig(f,dpi=fig.dpi,bbox_inches='tight');paths.append(f);plt.close(fig)
    # Original notebook cell 118: measured-vs-generated energy scatter.
    fig=plt.figure(figsize=(8,7.3))
    plt.scatter(full_data['mc_match_et'],full_data['cluster_truthet'],marker='.')
    plt.ylabel(r'$E_{fast}$: Standard FastCalo (MeV)',fontsize=16)
    plt.xlabel(r'$E_{true}$: Monte Carlo (MeV)',fontsize=16)
    plt.xticks(rotation=0,fontsize=12)
    ax=plt.subplot();xmin,xmax,ymin,ymax=plt.axis();dx=(xmax-xmin)/20;dy=(ymax-ymin)/20
    ax.text(xmin+dx,ymax-dy,label,fontsize=13)
    # Original line x=0..xmax; use the two endpoints for the identical ideal diagonal.
    plt.plot([0,xmax],[0,xmax],'--',color='black')
    f=out/'original_fast_vs_truth.png';fig.savefig(f,dpi=fig.dpi,bbox_inches='tight');paths.append(f);plt.close(fig)
    return paths
