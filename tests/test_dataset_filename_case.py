"""Regression tests for the filenames shipped in the public Lorenzetti Data folder."""
import io
import tarfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from lorenzetti_repro.dataset import find_dataset, read_dataset
from lorenzetti_repro.article_workflow import prepare_data_optional
from scripts.unpack_lorenzetti import unpack


def _csv_bytes(n=32):
    frame = pd.DataFrame({str(i): np.linspace(i+1, i+3, n) for i in range(100)})
    frame['mc_match_et'] = np.linspace(12000, 25000, n)
    frame['cluster_truthet'] = np.linspace(11000, 23000, n)
    frame['cluster_trutheta'] = np.linspace(-1.2, 1.2, n)
    return frame.to_csv(index=False).encode()


def test_real_public_dataset_filename_is_located(tmp_path):
    base = tmp_path/'Data'
    base.mkdir()
    file = base/'Lorenzetti_electrons_merge_root.csv'
    file.write_bytes(_csv_bytes())
    assert find_dataset(base) == file
    assert prepare_data_optional(tmp_path, extract=False) == file
    assert len(read_dataset(file)) == 32


def test_case_variants_are_accepted(tmp_path):
    for name in ['LORENZETTI_electrons_merge_root.csv', 'lorenzetti_electrons_merge_root.CSV']:
        base = tmp_path / name.replace('.', '_')
        base.mkdir()
        file = base/name
        file.write_bytes(_csv_bytes())
        assert find_dataset(base) == file


def test_public_name_inside_zip(tmp_path):
    folder = tmp_path/'Data'
    folder.mkdir()
    with zipfile.ZipFile(folder/'events.zip', 'w') as archive:
        archive.writestr('source/Lorenzetti_electrons_merge_root.csv', _csv_bytes())
    extracted = unpack(folder)
    assert extracted.name == 'Lorenzetti_electrons_merge_root.csv'
    assert find_dataset(folder) == extracted
    assert len(read_dataset(extracted)) == 32


def test_public_name_inside_tar(tmp_path):
    folder = tmp_path/'Data'
    folder.mkdir()
    payload = _csv_bytes()
    with tarfile.open(folder/'events.tar.gz','w:gz') as archive:
        obj=tarfile.TarInfo('source/Lorenzetti_electrons_merge_root.csv')
        obj.size=len(payload)
        archive.addfile(obj, io.BytesIO(payload))
    extracted=unpack(folder)
    assert extracted.name == 'Lorenzetti_electrons_merge_root.csv'
    assert find_dataset(folder) == extracted


def test_other_csv_does_not_override_lorenzetti(tmp_path):
    folder=tmp_path/'Data';folder.mkdir()
    (folder/'Lorenzetti_electrons_merge_root.csv').write_bytes(_csv_bytes())
    (folder/'atlas_electrons.csv').write_bytes(_csv_bytes())
    assert find_dataset(folder).name=='Lorenzetti_electrons_merge_root.csv'
