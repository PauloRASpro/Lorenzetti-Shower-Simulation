#!/usr/bin/env python3
"""Choose compact historical Lorenzetti artifacts without opening any pickle.

Example (Windows PowerShell):
  python select_pickles_for_repository.py --source "D:\\Pesquisas\\Lorenzetti_20251203"
  python select_pickles_for_repository.py --source "D:\\Pesquisas\\Lorenzetti_20251203" --write --output "lorenzetti_essentials.zip"
  python select_pickles_for_repository.py --source "D:\\Pesquisas\\Lorenzetti_20251203" --write --all-methods --output "lorenzetti_all_methods.zip"

The source remains untouched. Only filenames/sizes are inspected. No pickle deserialization.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile, ZIP_DEFLATED

DEFAULT_METHODS = ["BDT", "RFBDT", "DNN", "GNN"]
KIND_GROUPS = {
    "noSEG_Full_DF": "essential: event-level predictions",
    "noSEG_performance_metrics": "small: published-value crosscheck",
    "noSEG_cv_metrics_df": "small: validation metrics",
    "integrated_result_": "recommended: original bin summaries",
}
SUFFIXES = [".pkl.gz", ".pickle.gz", ".pkl", ".pickle"]


def parse_stem(name: str):
    low = name.lower()
    suffix = next((s for s in SUFFIXES if low.endswith(s)), None)
    if not suffix:
        return None
    stem = name[:-len(suffix)]
    for prefix, kind in (("integrated_result_", "integrated_result_"), ("result_", "result_"), ("", "")):
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)_(noSEG_[A-Za-z0-9_]+)", rest)
            if match:
                return match.group(1).upper(), (kind if prefix else match.group(2)), match.group(2)
    return None


def select(source: Path, methods: set[str] | None, integrated: bool):
    chosen, excluded = [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        match = parse_stem(path.name)
        if match is None:
            excluded.append((path,"not a recognized pickle file")); continue
        method, group, detail = match
        if methods is not None and method not in methods:
            excluded.append((path,"method not selected"));continue
        if group == "integrated_result_":
            if not integrated:
                excluded.append((path,"integrated results not selected"));continue
            reason=KIND_GROUPS[group]
        elif group in KIND_GROUPS:
            reason=KIND_GROUPS[group]
        else:
            excluded.append((path,"duplicated/derivable, or includes a model object"));continue
        chosen.append((path,method,group,reason))
    return chosen,excluded


def hash_file(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source",required=True,help="folder with original pickle files; scanned recursively")
    ap.add_argument("--methods",nargs="+",default=DEFAULT_METHODS,help="model prefixes; default BDT RFBDT DNN GNN")
    ap.add_argument("--all-methods",action="store_true",help="include all recognized methods found in folder")
    ap.add_argument("--without-integrated",action="store_true",help="omit original integrated/binned results to make the ZIP smaller")
    ap.add_argument("--output",default="lorenzetti_essential_pickles.zip",help="output zip (not written without --write)")
    ap.add_argument("--write",action="store_true",help="actually create ZIP; otherwise preview only")
    args=ap.parse_args()
    src=Path(args.source).expanduser().resolve()
    if not src.is_dir():ap.error(f"Not a folder: {src}")
    methods=None if args.all_methods else {m.upper() for m in args.methods}
    chosen,excluded=select(src,methods,not args.without_integrated)
    size=sum(p.stat().st_size for p,*_ in chosen)
    print(f"Selected {len(chosen)} files; total {size/2**20:.1f} MiB before ZIP. Excluded {len(excluded)} files.")
    for p,method,kind,reason in chosen:
        print(f"  {p.stat().st_size/2**20:8.1f} MiB | {p.relative_to(src)} | {reason}")
    if not chosen:raise SystemExit("No matching files were found; check folder and file suffixes.")
    if not args.write:
        print("PREVIEW ONLY. Run again with --write to build ZIP. Originals were not changed.")
        return
    out=Path(args.output).expanduser().resolve()
    if src in out.parents:raise SystemExit("Write the output ZIP outside the source folder to avoid recursive selection.")
    out.parent.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with ZipFile(out,'w',compression=ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for p,method,kind,reason in chosen:
            # Prefix with artifacts/pickles/ to match the public repository's existing loader.
            arcname=f"artifacts/pickles/{p.relative_to(src).as_posix()}"
            z.write(p,arcname)
            manifest.append(dict(path=arcname,method=method,category=kind,source_size_bytes=p.stat().st_size,
                                 sha256=hash_file(p),purpose=reason))
        z.writestr("artifacts/manifests/selected_pickles_manifest.json",json.dumps({
            "note":"This is a selection of historic pickles; no model re-training or re-calculation was performed.",
            "files":manifest},indent=2,ensure_ascii=False)+"\n")
    print(f"Created {out} ({out.stat().st_size/2**20:.1f} MiB). No source files altered.")

if __name__=="__main__":main()
