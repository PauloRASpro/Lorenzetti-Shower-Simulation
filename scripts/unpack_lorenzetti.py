#!/usr/bin/env python3
"""Extract the CSV dataset already distributed in data/ (ZIP/GZip/7z).

Does not download or alter the original compressed archive. Validates the
Lorenzetti 100-ring schema after extraction.
"""
from __future__ import annotations
import argparse
import gzip
import shutil
import zipfile
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lorenzetti_repro.dataset import read_dataset,KNOWN_NAMES,is_known_dataset_filename


def unpack(root=None,archive=None,output=None):
    if root is None:root=ROOT/('Data' if (ROOT/'Data').exists() else 'data')
    root=Path(root)
    if archive: candidate=[Path(archive)]
    else:
        search_roots=[root]
        alternate=root.parent/('Data' if root.name=='data' else 'data')
        if root.name.lower()=='data' and alternate!=root and alternate.exists():search_roots.append(alternate)
        candidate=sorted([x for base in search_roots if base.exists() for x in base.rglob('*')
                          if x.is_file() and x.name.lower().endswith(('.zip','.csv.gz','.7z','.tar.gz','.tar','.tgz','.rar'))])
        if candidate and not any(x.is_file() and x.name.lower().endswith(('.zip','.csv.gz','.7z','.tar.gz','.tar','.tgz','.rar')) for x in root.rglob('*')):
            root=candidate[0].parent
    root.mkdir(parents=True,exist_ok=True)
    for file in candidate:
        if not file.exists():raise FileNotFoundError(file)
        if file.name.lower().endswith('.zip'):
            with zipfile.ZipFile(file) as z:
                names=[x for x in z.infolist() if not x.is_dir() and x.filename.lower().endswith('.csv')]
                names=sorted(names,key=lambda x:(not is_known_dataset_filename(x.filename),x.file_size))
                if not names:continue
                if len(names)>1 and not is_known_dataset_filename(names[0].filename):
                    raise ValueError(f'{file.name} contains multiple CSVs; specify the intended one in a separate archive')
                name=names[0];dest=Path(output) if output else root/Path(name.filename).name
                if dest.exists():return dest
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name,'r') as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
        elif file.name.lower().endswith('.csv.gz'):
            dest=Path(output) if output else root/file.name[:-3]
            if dest.exists():return dest
            with gzip.open(file,'rb') as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
        elif file.name.lower().endswith(('.tar','.tar.gz','.tgz')):
            import tarfile
            with tarfile.open(file,'r:*') as tf:
                members=[m for m in tf.getmembers() if m.isfile() and Path(m.name).suffix.lower()=='.csv']
                members.sort(key=lambda m: (not is_known_dataset_filename(m.name),m.size))
                if len(members)!=1 and (not members or not is_known_dataset_filename(members[0].name)):
                    raise ValueError(f'Archive {file} has no unambiguous Lorenzetti CSV')
                member=members[0]
                dest=Path(output) if output else root/Path(member.name).name
                if dest.exists(): return dest
                dest.parent.mkdir(parents=True,exist_ok=True)
                with tf.extractfile(member) as src,dest.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
        elif file.name.lower().endswith('.rar'):
            try:
                import rarfile
            except ImportError as exc:
                raise RuntimeError('Install the rarfile library and an unrar/bsdtar extractor to open .rar dataset archives') from exc
            with rarfile.RarFile(file) as rf:
                members=[n for n in rf.namelist() if n.lower().endswith('.csv')]
                if not members:raise ValueError(f'No CSV in {file}')
                members.sort(key=lambda n:not is_known_dataset_filename(n))
                if len(members)>1 and not is_known_dataset_filename(members[0]):
                    raise ValueError(f'Ambiguous Lorenzetti CSV members in {file}: {members}')
                dest=Path(output) if output else root/Path(members[0]).name
                if dest.exists(): return dest
                dest.parent.mkdir(parents=True,exist_ok=True)
                with rf.open(members[0]) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
        elif file.name.lower().endswith('.7z'):
            try:import py7zr
            except ImportError as exc:raise RuntimeError('Install py7zr to extract .7z archives: pip install py7zr') from exc
            with py7zr.SevenZipFile(file,'r') as z:
                # For 7z, extract only verified names; reject unsafe paths.
                files=[n for n in z.getnames() if n.lower().endswith('.csv')]
                if len(files)!=1:raise ValueError(f'Expected one CSV in {file}: {files}')
                target=Path(files[0]);
                if '..' in target.parts or target.is_absolute():raise ValueError('Unsafe 7z archive member name')
                z.extract(path=root,targets=files)
                dest=root/target
        else:continue
        df=read_dataset(dest)
        print(f'Extracted {len(df):,} rows from {file.name} -> {dest}')
        return dest
    existing=[x for x in root.rglob('*.csv') if is_known_dataset_filename(x.name)]
    if len(existing)==1:return existing[0]
    available=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]
    raise FileNotFoundError(
        f'No Lorenzetti CSV or supported compressed archive was found in {root}. '
        f'Files present: {available[:25]}. '
        'The original dataset must be present in the local GitHub clone, not only on github.com. '
        'Check Data/ and Git LFS download status.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive');p.add_argument('--dir',default=None);p.add_argument('--output');a=p.parse_args()
    print(unpack(a.dir,a.archive,a.output))
