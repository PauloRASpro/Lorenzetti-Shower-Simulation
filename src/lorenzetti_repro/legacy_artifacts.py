"""Inventory and explicit, trust-gated recovery of historical notebook DataFrames.

Historical source: notebook cells 303 (save/load), 671 (method selection) and
274 (event-level results). This module does not reconstruct training history.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

KIND_SUFFIXES = (
    "noSEG_Full_DF", "noSEG_erros_data", "noSEG_erros_data_alpha",
    "noSEG_cv_metrics_df", "noSEG_times", "noSEG_performance_metrics",
)
PREFIXED_KINDS = ("result_", "integrated_result_")
KNOWN_METHODS = (
    "MLR", "PREG", "DTREE", "BDT", "RFOR", "BDTRF", "RFBDT", "RFBDTS",
    "DNN", "GNN", "BSPL", "CSPL", "RIDGE", "LASSO", "BDTX", "BDTC",
    "SGNN", "CGNN", "TRAN", "STRAN",
)
ARTICLE_METHODS = {
    "tns": ("MLR", "PREG", "DTREE", "BDT", "BSPL"),
    "measurement": ("DTREE", "BDT", "RFOR", "BDTRF", "RFBDT", "RFBDTS"),
    "information_fusion": ("BDT", "RFBDT", "DNN", "GNN"),
}
EXTENSIONS = (".pkl.gz", ".pickle.gz", ".pkl", ".pickle")

def artifact_stems(method: str) -> tuple[str, ...]:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", method):
        raise ValueError("Invalid method name")
    base = [f"{method}_{kind}" for kind in KIND_SUFFIXES]
    base += [f"{prefix}{method}_noSEG_Full_DF" for prefix in PREFIXED_KINDS]
    return tuple(base)

def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda:f.read(2**20), b""):
            h.update(buf)
    return h.hexdigest()

def recognize_filename(name: str) -> dict:
    stem=name
    ext=next((e for e in EXTENSIONS if name.lower().endswith(e)), None)
    if ext is None:
        return {"method":"", "kind":"", "extension":"", "known":False}
    stem=stem[:-len(ext)]
    for method in KNOWN_METHODS:
        for expected in artifact_stems(method):
            if stem == expected:
                return {"method":method, "kind":expected[len(method)+1:] if expected.startswith(method+'_') else expected.split(method+'_',1)[0]+"noSEG_Full_DF", "extension":ext, "known":True}
    return {"method":"", "kind":stem, "extension":ext, "known":False}

def catalog(root: str|Path, *, checksum:bool=True) -> list[dict]:
    """Safely inspect filenames, sizes, hashes. NO pickle deserialization."""
    base=Path(root).expanduser().resolve()
    if not base.exists():
        return []
    if not base.is_dir():
        raise NotADirectoryError(base)
    records=[]
    for p in sorted(base.rglob('*')):
        if not p.is_file() or not any(p.name.lower().endswith(x) for x in EXTENSIONS):
            continue
        meta=recognize_filename(p.name)
        records.append({"relative_path":p.relative_to(base).as_posix(),
                        "bytes":p.stat().st_size,
                        "sha256":file_sha256(p) if checksum else None, **meta})
    return records

def load_method_bundle(root: str|Path, method: str, *, trust_pickle:bool=False,
                       expected_sha256:Mapping[str,str]|None=None,
                       include_model_objects:bool=False) -> dict:
    """Load notebook's eight named DataFrames; never modify module/global namespace.

    UNPICKLING CAN EXECUTE ARBITRARY CODE. Only load files from a source you trust.
    `trust_pickle=True` is mandatory. SHA-256 validates integrity, not safety.
    """
    if not trust_pickle:
        raise PermissionError('Pickle deserialization disabled. Set trust_pickle=True ONLY for files from a trusted source.')
    import pandas as pd
    base=Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f'Pickle folder not found: {base}')
    result={}
    for stem in artifact_stems(method):
        # Historical *_times DataFrames can contain huge Python estimators (Best Model).
        # Skip them by default: event-level predictions suffice to replay results.
        if stem.endswith("_noSEG_times") and not include_model_objects:
            continue
        matched=[]
        for ext in EXTENSIONS:
            matched.extend(p for p in base.rglob(stem+ext) if p.is_file() and not p.is_symlink())
        if len(matched)>1:
            raise ValueError(f'Multiple candidates for {stem}; select/rename a single archive before loading: {matched}')
        if not matched:
            continue
        p=matched[0]
        rel=p.relative_to(base).as_posix()
        if expected_sha256 is not None:
            want=expected_sha256.get(rel)
            if not want:
                raise ValueError(f'Missing expected SHA-256 for {rel}')
            if file_sha256(p).lower()!=want.lower():
                raise ValueError(f'Checksum mismatch: {rel}')
        # pandas infers gzip compression from .pkl.gz and supports both variants.
        obj=pd.read_pickle(p, compression='infer')
        if not isinstance(obj,pd.DataFrame):
            raise TypeError(f'Expected pandas.DataFrame in {p}, got {type(obj).__name__}')
        result[stem] = obj
    return result

def select_event_frames(bundles:Mapping[str,dict], *, labels:Mapping[str,list[str]]|None=None):
    """Extract event DataFrames from complete noSEG results without training.

    Grouped by `method-selection`, which is the source notebook's label for each
    representation. Rows may repeat across methods but never merge by row order.
    """
    import pandas as pd
    out={}
    labels=labels or {}
    for method, bundle in bundles.items():
        key=f'{method}_noSEG_Full_DF'
        if key not in bundle:continue
        frame=bundle[key]
        required={'method-selection','ypred','cluster_truthet','mc_match_et'}
        if not required.issubset(frame.columns):
            raise ValueError(f'Historical artifact {key} missing {sorted(required-set(frame.columns))}')
        for label,g in frame.groupby('method-selection',sort=True):
            if method in labels and label not in labels[method]:
                continue
            if label in out:
                raise ValueError(f'Duplicate result label: {label}')
            out[str(label)] = g.copy()
    return out

def event_values(frame):
    """Convert the nine-column saved per-event format into physical MeV fields.

    `ypred` is alpha_calib, not E_calib. `yte` is alpha_true, not E_true.
    """
    import numpy as np
    import pandas as pd
    def arr(name):
        v=np.asarray(pd.to_numeric(frame[name],errors='raise'),dtype=float).reshape(-1)
        if not np.isfinite(v).all():
            raise ValueError(f'{name}: non-finite value found; investigate source')
        return v
    ef=arr('cluster_truthet')
    et=arr('mc_match_et')
    alpha=arr('ypred')
    ec=ef*alpha
    if 'calib_et' in frame:
        stored=arr('calib_et')
        if not np.allclose(ec,stored,rtol=1e-7,atol=1e-5):
            raise ValueError('Stored calib_et differs from cluster_truthet * ypred')
    if 'yte' in frame:
        ideal=arr('yte')
        mask=ef!=0
        if not np.allclose(ideal[mask],et[mask]/ef[mask],rtol=1e-5,atol=1e-4):
            raise ValueError('yte is not consistent with alpha_true=E_true/E_fast')
    out=pd.DataFrame({'E_true_MeV':et,'E_fast_MeV':ef,'E_calib_MeV':ec,
                     'alpha_calib':alpha,'residual_fast_MeV':et-ef,
                     'residual_calib_MeV':et-ec})
    if 'cluster_trutheta' in frame:
        out['abs_eta']=abs(arr('cluster_trutheta'))
    return out

def read_manifest(path:str|Path)->dict[str,str]:
    payload=json.loads(Path(path).read_text(encoding='utf8'))
    rows=payload['files'] if isinstance(payload,dict) else payload
    return {row['relative_path']:row['sha256'] for row in rows}
