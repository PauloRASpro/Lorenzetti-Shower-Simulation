# Dataset

Use the existing compressed Lorenzetti simulation data in this GitHub repository; it is **not duplicated** in this overlay. Decompress the archive to obtain the CSV used by the source notebook (for example, `Lorenzetti_simulation_raw_20250815.csv`).

Expected: 302,619 rows, 100 ring columns named `0`..`99`, `mc_match_et` (Etrue), `cluster_truthet` (Efast), and diagnostics such as `cluster_trutheta`. All energies used by the manuscript results are in MeV, except where expressly divided by 1000 for Stage I GeV tables.

This repo does not assume ATLAS real data or collab-restricted inputs. Do **not** run automatic `dropna()`, row deletion, balancing, target leaks, or a second global normalization of rings. If the actual CSV contains an extra leading index column, leave it intact but select only the 100 named rings for model features. No CSV is committed twice.

The script `python scripts/check_dataset.py path/to/file.csv` checks cardinality, schema and SHA256. A published run cannot be guaranteed identical without the historical byte-level dataset checksum and split indices.
