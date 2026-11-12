#!/usr/bin/env python3
"""No-deserialization index of the user's historical pickles, including SHA256."""
import argparse
import json
import csv
from pathlib import Path
from lorenzetti_repro.legacy_artifacts import catalog

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--folder',default='artifacts/pickles')
    ap.add_argument('--output',default='artifacts/manifests/pickles_manifest.json')
    ns=ap.parse_args()
    rows=catalog(ns.folder)
    dest=Path(ns.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps({'schema':'lorenzetti-pickle-manifest-v1','root_hint':ns.folder,'files':rows},indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(f'Indexed {len(rows)} files without loading them: {dest}')
    for row in rows:
        print(f"{row['method'] or '?':7s} {row['bytes']/2**20:8.2f} MiB  {row['relative_path']}")
    if not rows:print('No pickle files found; upload/unzip the original folder, then rerun.')
if __name__=='__main__':main()
