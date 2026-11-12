from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RINGS=[str(i) for i in range(100)]
TRUE='mc_match_et'
FAST='cluster_truthet'
ETA='cluster_trutheta'
# Preserve the public file's actual spelling while accepting historical casing.
KNOWN_NAMES=['Lorenzetti_simulation_raw_20250815.csv','Lorenzetti_electrons_merge_root.csv']
KNOWN_NAME_KEYS={name.casefold() for name in KNOWN_NAMES}

def is_known_dataset_filename(filename):
    """Match the two documented Lorenzetti CSV basenames irrespective of case."""
    return Path(filename).name.casefold() in KNOWN_NAME_KEYS
EXPECTED_EVENTS=302619

def find_dataset(root='data',explicit=None):
    """Never silently pick among multiple datasets (including ATLAS)."""
    if explicit:
        p=Path(explicit)
        if not p.is_file(): raise FileNotFoundError(p)
        return p
    root=Path(root)
    search_roots=[root]
    alternate=root.parent/('Data' if root.name=='data' else 'data')
    if root.name.lower()=='data' and alternate!=root and alternate.exists():
        search_roots.append(alternate)
    # Under Windows, Data/ and data/ may resolve to the same directory.
    candidates=[p for base in search_roots if base.exists() for p in base.rglob('*') if p.is_file() and is_known_dataset_filename(p.name)]
    options=sorted({str(p.resolve()).casefold():p for p in candidates}.values())
    if len(options)!=1:
        raise ValueError(f'Expected exactly one Lorenzetti CSV under {root}, found: {[str(p) for p in options]}. Pass --data explicitly.')
    return options[0]

def read_dataset(path):
    df=pd.read_csv(path)
    df.columns=[str(c) for c in df.columns]
    needed=RINGS+[TRUE,FAST]
    missing=[k for k in needed if k not in df]
    if missing: raise ValueError(f'Missing columns: {missing[:15]}')
    df=df.reset_index(drop=True)
    for name in needed:
        if not pd.api.types.is_numeric_dtype(df[name]): raise ValueError(f'Non-numeric: {name}')
    if not np.isfinite(df[needed].to_numpy(dtype='float64')).all():
        raise ValueError('Missing/Inf/NaN values. No silent dropna or imputation: inspect original data.')
    if not (df[FAST].to_numpy()>0).all(): raise ValueError('Non-positive Efast, target division undefined.')
    return df

def split_70_30(df,random_state=0):
    """Use the documented base seed 0; exact historical index identity still requires saved split indices."""
    ids=np.arange(len(df))
    d70,d30=train_test_split(ids,test_size=0.30,random_state=random_state,shuffle=True)
    assert not set(d70).intersection(d30)
    return np.asarray(d70,dtype=int),np.asarray(d30,dtype=int)

def alpha_target(df):
    return df[TRUE].to_numpy(dtype=float)/df[FAST].to_numpy(dtype=float)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
