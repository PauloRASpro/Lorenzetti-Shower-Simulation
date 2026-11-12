from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from lorenzetti_repro.run_storage import make_new_run,choose_replay_dir
from lorenzetti_repro.article_training import train_article

def test_unique_runs_and_replay_selection(tmp_path):
    root=tmp_path
    original=root/'artifacts'/'pickles'
    original.mkdir(parents=True)
    fixed=original/'BDT_noSEG_Full_DF.pkl'
    fixed.write_bytes(b'original archival bytes must survive')
    base1,run1=make_new_run(root,'tns')
    base2,run2=make_new_run(root,'tns')
    assert run1!=run2
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)
    assert choose_replay_dir(root,'tns') == original
    assert choose_replay_dir(root,'tns','latest_new_run') == max(run1,run2)
    assert choose_replay_dir(root,'tns','specific_new_run',base1.name)==run1
    with pytest.raises((ValueError,FileNotFoundError)):
        choose_replay_dir(root,'tns','specific_new_run','../../bad')
    assert fixed.read_bytes()==b'original archival bytes must survive'

def test_training_rejects_historical_archive_even_with_small_synthetic_input(tmp_path):
    from lorenzetti_repro.dataset import RINGS
    root=tmp_path
    archive=root/'artifacts'/'pickles'
    archive.mkdir(parents=True)
    frame=pd.DataFrame(np.ones((30,100)),columns=RINGS)
    frame['mc_match_et']=np.arange(30)+5000
    frame['cluster_truthet']=np.arange(30)+4500
    frame['cluster_trutheta']=0.2
    csv=root/'dataset.csv';frame.to_csv(csv,index=False)
    with pytest.raises(ValueError,match='historical archive'):
        train_article('tns',root,csv,allow_subset=True,chosen=[],output_dir=archive)
    assert list(archive.iterdir())==[]

def test_trained_output_is_new_and_cannot_be_reused(tmp_path):
    from lorenzetti_repro.dataset import RINGS
    root=tmp_path
    frame=pd.DataFrame(np.ones((45,100)),columns=RINGS)
    frame['mc_match_et']=np.arange(45)+5000
    frame['cluster_truthet']=np.arange(45)+4500
    frame['cluster_trutheta']=0.2
    csv=root/'input.csv';frame.to_csv(csv,index=False)
    base,output=make_new_run(root,'tns')
    got=train_article('tns',root,csv,allow_subset=True,chosen=[],output_dir=output)
    assert got==[]
    assert (base/'manifests'/'training_tns.json').exists()
    with pytest.raises(FileExistsError):
        train_article('tns',root,csv,allow_subset=True,chosen=[],output_dir=output)
