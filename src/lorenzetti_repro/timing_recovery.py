"""Extract numerical prediction timings from the original model-containing archives.

Only deserialize archives supplied by and trusted by their owner. The output
contains no fitted estimator objects and is compact enough for version control.
"""
from __future__ import annotations
from pathlib import Path
import gc
import warnings
import pandas as pd
from .legacy_artifacts import ARTICLE_METHODS, EXTENSIONS
from .analysis_tables import timing_from_archived_bundles


def recover_timing_csv(article, pickle_root, output, *, trusted=False):
    output=Path(output)
    existing = pd.read_csv(output) if output.is_file() and output.stat().st_size else pd.DataFrame()
    recorded = set(existing['configuration'].astype(str)) if 'configuration' in existing else set()
    required_prefixes = {f'{m} no-Bin:' for m in ARTICLE_METHODS[article]}
    # Do not return merely because one configuration from each method was
    # recovered. A paper may compare 10+ configurations from the same family.
    # Preserve existing measurements, then inspect every available archive.
    if not trusted:
        raise PermissionError('Loading model-containing pickle files requires explicit trust.')
    root=Path(pickle_root)
    entries=[]
    for method in ARTICLE_METHODS[article]:
        candidates=[p for ext in EXTENSIONS for p in root.rglob(f'{method}_noSEG_times{ext}') if p.is_file()]
        if not candidates:continue
        if len(candidates)!=1:raise ValueError(f'Ambiguous timing archive for {method}: {candidates}')
        archive=candidates[0]
        if method == 'GNN':
            # The historical notebook serialized custom model types from
            # __main__. Expose their original definitions when PyG is installed.
            # Only invoked after explicit user opt-in to trusted pickle loading.
            try:
                import __main__
                from .models.gnn_original_cell562 import LayeredRingGraphHybrid, _RegionGraphBlock
                if not hasattr(__main__, 'LayeredRingGraphHybrid'):
                    __main__.LayeredRingGraphHybrid=LayeredRingGraphHybrid
                if not hasattr(__main__, '_RegionGraphBlock'):
                    __main__._RegionGraphBlock=_RegionGraphBlock
            except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
                warnings.warn(f'Optional GNN timing model definitions unavailable: {exc}',RuntimeWarning)
        try:
            # Legacy pickles may contain custom model classes from the original
            # notebook's __main__ namespace. A missing model class must not
            # prevent loading the separately saved event predictions.
            frame=pd.read_pickle(archive,compression='infer')
        except (AttributeError, ImportError, ModuleNotFoundError, TypeError, RuntimeError) as exc:
            warnings.warn(f'Optional training-time model archive skipped ({archive.name}): {exc}. '
                          'Event predictions, statistics and figures remain available; '
                          'export numerical times in the original training environment if needed.',RuntimeWarning)
            continue
        try:
            if not isinstance(frame,pd.DataFrame):raise TypeError(f'{archive}: expected DataFrame')
            times=timing_from_archived_bundles({method:{f'{method}_noSEG_times':frame}})
            if not times.empty:entries.append(times)
        finally:
            del frame
            gc.collect()
    if not entries and existing.empty:
        raise FileNotFoundError('No compatible *_noSEG_times.pkl archives for this article')
    table=pd.concat(([existing] if not existing.empty else [])+entries,ignore_index=True)
    # If the same model is shared across articles, identical entries may occur;
    # conflicting duplicates are never silently removed.
    if table.configuration.duplicated().any():
        duplicates=table[table.configuration.duplicated(keep=False)]
        if duplicates.groupby('configuration').time_ms_per_event.nunique().max()>1:
            raise ValueError('Conflicting archived timings for a configuration')
        table=table.drop_duplicates('configuration')
    output.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(output,index=False)
    return output
