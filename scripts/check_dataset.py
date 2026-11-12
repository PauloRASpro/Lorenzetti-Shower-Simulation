#!/usr/bin/env python3
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lorenzetti_repro.dataset import find_dataset,read_dataset,sha256_file,EXPECTED_EVENTS,split_70_30
p=find_dataset(explicit=sys.argv[1] if len(sys.argv)>1 else None)
df=read_dataset(p)
a,b=split_70_30(df)
print(json.dumps({'path':str(p),'rows':len(df),'expected_rows':EXPECTED_EVENTS,
                  'size_matches':len(df)==EXPECTED_EVENTS,'D70':len(a),'D30':len(b),
                  'sha256':sha256_file(p)},indent=2))
