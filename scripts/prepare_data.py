#!/usr/bin/env python3
"""Optional safe extractor for ZIP already checked out into GitHub data/."""
import argparse,zipfile
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('zip_file',type=Path)
ap.add_argument('--destination',type=Path,default=Path('data/raw'))
args=ap.parse_args()
if not args.zip_file.is_file():raise SystemExit('ZIP not found')
args.destination.mkdir(parents=True,exist_ok=True)
base=args.destination.resolve()
with zipfile.ZipFile(args.zip_file) as z:
    for info in z.infolist():
        dst=(args.destination/info.filename).resolve()
        if not dst.is_relative_to(base):raise ValueError('Unsafe ZIP member: '+info.filename)
        if not info.is_dir() and info.file_size>50_000_000_000:raise ValueError('Excessive file size: '+info.filename)
    z.extractall(args.destination)
    print('Extracted',len(z.infolist()),'members to',base)
