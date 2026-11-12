# Historial artifact manifests

After obtaining the trusted original archive, run `python scripts/catalog_pickles.py --folder artifacts/pickles` to create `pickles_manifest.json`, and review paths and checksums before publishing.

The manifest is excluded from commits by default until author review; hashes verify integrity, **not pickle safety or scientific correctness**.
