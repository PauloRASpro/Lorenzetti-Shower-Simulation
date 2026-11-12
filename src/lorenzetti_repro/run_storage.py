"""Immutable historical archive and unique per-training run storage."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json

ARTICLES = ("tns", "measurement", "information_fusion")

def make_new_run(root: Path, article: str) -> tuple[Path, Path]:
    root=Path(root).resolve()
    if article not in ARTICLES:raise ValueError(f"Unknown research stage: {article}")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id=f"{stamp}_{uuid4().hex[:8]}"
    base=root/'artifacts'/'new_runs'/article/run_id
    if base.exists():raise FileExistsError(base)
    # train_article creates the pickles directory exclusively, guarding against an existing run.
    return base,base/'pickles'

def choose_replay_dir(root: Path, article: str, source: str='original', explicit: str|None=None) -> Path:
    root=Path(root).resolve()
    if article not in ARTICLES:raise ValueError(article)
    if source == 'original':
        return root/'artifacts'/'pickles'
    if source == 'latest_new_run':
        runs=root/'artifacts'/'new_runs'/article
        paths=sorted([p for p in runs.iterdir() if p.is_dir() and (p/'pickles').is_dir()]) if runs.exists() else []
        if not paths:raise FileNotFoundError(f"No new run for {article} under {runs}")
        return paths[-1]/'pickles'
    if source == 'specific_new_run':
        if not explicit:raise ValueError('NEW_RUN_ID must be supplied for specific_new_run')
        p=(root/'artifacts'/'new_runs'/article/explicit/'pickles').resolve()
        allowed=(root/'artifacts'/'new_runs'/article).resolve()
        if allowed not in p.parents or p.name!='pickles':raise ValueError('Invalid run ID')
        if not p.is_dir():raise FileNotFoundError(p)
        return p
    raise ValueError(f'Unknown replay source: {source}')
