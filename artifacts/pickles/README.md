# Original experiment archives

Copy the original `*.pkl` or `*.pkl.gz` DataFrames produced by the experiment notebook into this directory without renaming them.

Typical file names are:

```text
BDT_noSEG_Full_DF.pkl.gz
BDT_noSEG_performance_metrics.pkl.gz
BDT_noSEG_cv_metrics_df.pkl.gz
integrated_result_BDT_noSEG_Full_DF.pkl.gz
RFBDT_noSEG_Full_DF.pkl.gz
DNN_noSEG_Full_DF.pkl.gz
GNN_noSEG_Full_DF.pkl.gz
```

The full event tables contain the original prediction and truth columns. The integrated tables preserve the original energy and pseudorapidity intervals. Analysis code loads the relevant model families for each publication and records missing inputs explicitly.

The `*_noSEG_times.pkl` archives may contain trained model objects and can be very large. When these files are available locally, the notebook can extract their numerical timing metadata to `artifacts/metadata/prediction_times.csv`; the model objects are not copied into that CSV.

Never unpickle files from an untrusted source. For a public release, use Git LFS or a versioned external scientific archive when large files exceed normal GitHub limits.
