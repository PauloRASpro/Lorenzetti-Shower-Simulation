# Lorenzetti Dataset and Reproducibility

## Dataset

The Lorenzetti simulation dataset is provided in this repository as a **multi-volume 7z archive**, split into approximately 50 MB parts (`.7z.001`, `.7z.002`, `.7z.003`, etc.) to accommodate GitHub's file-size limits. The dataset is not duplicated in another format in the repository.

After extraction, the source notebooks use the CSV file `Lorenzetti\\\_electrons\\\_merge\\\_root.csv`.

### Download and extract the dataset

1. Download **all** parts of the archive (for example, `filename.7z.001`, `filename.7z.002`, and subsequent parts).
2. Place all parts in the **same directory**, without renaming them.
3. Start extraction **only from the first part**, ending in `.7z.001`. The extraction utility automatically combines the remaining volumes.

**Windows**

1. Install [7-Zip](https://7-zip.org).
2. Right-click the `.7z.001` file and select **7-Zip → Extract Here**, or extract to a selected folder.
3. Wait for 7-Zip to combine the volumes and extract the CSV.

**macOS**

1. Install [Keka](https://keka.io) or [The Unarchiver](https://theunarchiver.com).
2. Open the `.7z.001` file with the extraction utility. All parts must remain together in the same directory.

**Linux**

Run the following command in the directory containing the archive parts:

```bash
7z x filename.7z.001
```

Replace `filename.7z.001` with the name of the first volume. If the `7z` command is not installed, install an appropriate 7-Zip package for your Linux distribution (for example, `p7zip-full` on distributions that provide it).

> \\\*\\\*Important:\\\*\\\* Do not extract `.002`, `.003`, or subsequent parts separately, and do not treat the split 7z volumes as ZIP archives.

### Expected dataset structure

The experimental dataset contains **302,619 events**. The expected CSV schema includes:

* **100 Ringer features** in columns named `0` through `99`;
* `mc\\\_match\\\_et` — Monte Carlo true transverse energy (`E\\\_true`);
* `cluster\\\_truthet` — standard fast transverse-energy estimate (`E\\\_fast`);
* diagnostic observables such as `cluster\\\_trutheta`.

Energy values are expressed in **MeV**, except where the Stage I results explicitly divide by 1,000 to report values in GeV.

If the extracted CSV contains an additional leading index column, retain the data file as is, but select only the 100 named ring columns as ring-based model features.

### Data integrity and experimental conventions

The experiments use the Lorenzetti-generated benchmark, **not ATLAS real data** or collaboration-restricted inputs. To preserve the original experimental protocol:

* Do not automatically apply `dropna()`, remove rows, or rebalance the dataset.
* Do not introduce target information into inference features.
* Do not perform a second global normalization of the ring features.
* Keep the original dataset intact and avoid committing a duplicate CSV.

Once the dataset has been extracted, run the integrity check from the repository root:

```bash
python scripts/check\\\_dataset.py Data/Lorenzetti\\\_electrons\\\_merge\\\_root.csv
```

Adjust the path if the extracted CSV is located elsewhere. The script checks dataset cardinality, schema, and SHA-256 checksum.

Reproducing a historical run byte for byte additionally requires the historical dataset checksum and the original train/test split indices.



