# Measured inference times

The optional `prediction_times.csv` contains `configuration,time_ms_per_event,F` extracted from the experiment's `*_noSEG_times.pkl[.gz]`. It is read by the notebooks for the engineering cost and D0 figures.

To extract locally from personally trusted archives:

```bash
python scripts/export_timing_metadata.py --folder artifacts/pickles --trust-my-pickles
```

Only numeric times and configuration labels are exported; fitted estimators in the large `*_times.pkl` archives are not included in this CSV.
