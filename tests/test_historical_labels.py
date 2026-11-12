from pathlib import Path
import numpy as np
import pandas as pd
from lorenzetti_repro.original_replay import replay_original_figures

def frame(label,n=70):
    et=np.linspace(7000,50000,n);ef=et*1.02;alpha=et/ef*1.004
    return pd.DataFrame({'method-selection':[label]*n,'method':[label.split(':')[0]]*n,
            'ypred':alpha,'yte':et/ef,'cluster_truthet':ef,'mc_match_et':et,
            'calib_et':ef*alpha,'cluster_trutheta':np.linspace(0,1.46,n)})

def test_exact_stage1_figure_uses_pca10_not_pca100(tmp_path):
    data={'BDT no-Bin: PCA-100':frame('BDT no-Bin: PCA-100'),
          'BDT no-Bin: PCA-10':frame('BDT no-Bin: PCA-10')}
    report=replay_original_figures('tns',data,{},tmp_path)
    assert report['best']=='BDT no-Bin: PCA-10'
    assert report['figure_paths']
