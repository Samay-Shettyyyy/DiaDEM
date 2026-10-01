# Dataset storage

Raw and split CSVs are read from the **parent repo** (see `configs/datasets/*.json`).

No large `.npy` files are duplicated here. All processed outputs go to:

`outputs/{dataset}/{split_mode}/processed/`

To add a new dataset later (MP, CSC, …):

1. Add `configs/datasets/yourdataset.json`
2. Point `source_csv` and `split_csv_pattern` at your files
3. Run: `DATASET=yourdataset ./run.sh`
