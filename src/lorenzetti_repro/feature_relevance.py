"""Stage I Fig. 20: representation relevance computed on D70 alone.

CORR and MUTI scores refer to original rings. PCA and ICA scores refer to
transformed components (not original calorimeter layers).
"""
from pathlib import Path
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from .dataset import RINGS,TRUE,read_dataset,split_70_30
from .representations import Representation,_safe_corr


def feature_scores(data_csv,seed=0):
    frame=read_dataset(data_csv)
    dev,_=split_70_30(frame)
    X=frame.loc[dev,RINGS].to_numpy(dtype=float)
    y=frame.loc[dev,TRUE].to_numpy(dtype=float)
    corr=np.asarray([abs(_safe_corr(X[:,i],y)) for i in range(100)])
    mi=mutual_info_regression(X,y,random_state=seed)
    scores={'CORR':corr,'MUTI':mi}
    for kind in ('PCA','ICA'):
        g=Representation(f'{kind}-100',seed=seed).fit(X,y)
        # Order scores by original component index; transformation has shape (N, 100).
        components=g.transform(X)
        scores[kind]=np.asarray([abs(_safe_corr(components[:,i],y)) for i in range(100)])
    return scores


def draw_feature_relevance(data_csv,output):
    from .publication_figures import _plt
    import pandas as pd
    data=feature_scores(data_csv)
    plt=_plt();fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,(name,scores) in zip(axes.flat,data.items()):
        ax.bar(np.arange(1,101),scores,width=1.0)
        ax.set(title=name,xlabel='Ring' if name in ('CORR','MUTI') else 'Component',ylabel='Relevance score')
    fig.tight_layout()
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=125);plt.close(fig)
    return out
