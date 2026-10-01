# cleanedup_diadem

Streamlined DiADEM workflow: **one folder, one `run.sh`, all datasets and baselines.**

The original messy repo (`dices/`, `voices/`, `Mutlipico/`) is untouched. All new code lives here.

## Layout

```
cleanedup_diadem/
├── run.sh                          ← single entry point (always run from here)
├── configs/
│   ├── datasets/                   ← dices.json, voiced.json (column maps, labels)
│   ├── diadem.json                 ← DiADEM hyperparameters (matches voices_demographic.cfg)
│   ├── disco_baseline.json         ← DisCo baseline (generic)
│   └── disco_baseline_voiced.json  ← DisCo baseline for VOICED (matches voiced_baseline_template.cfg)
├── datasets/                       ← symlinks / pointers to raw CSVs in parent repo
├── outputs/                        ← all artifacts (no hunting across folders)
│   └── {dataset}/{split_mode}/
│       ├── processed/              ← shared npy from preprocess
│       ├── diadem/                 ← DiADEM model + eval_summary.json
│       ├── disco/                  ← DisCo baseline + eval_summary.json
│       └── comparison/             ← results_table.md (paper-ready)
└── diadem/                         ← Python package (models, train, eval, compare)
```

## Quick start

```bash
cd cleanedup_diadem
chmod +x run.sh

# Full pipeline (uses existing split CSVs in parent repo; copies npy → trains both models)
./run.sh

# DICES, item split
DATASET=dices SPLIT_MODE=by_item ./run.sh

# From scratch:
STEPS=split,preprocess,prepare,train,eval,compare ./run.sh

# Bootstrap from existing legacy processed npy (fast first run):
STEPS=seed,prepare,train,eval,compare ./run.sh

# Only regenerate comparison table
STEPS=compare ./run.sh
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `voiced` | `dices` or `voiced` |
| `SPLIT_MODE` | `by_annotator` | `by_annotator` or `by_item` |
| `MODELS` | `diadem,disco` | Comma-separated models to train/eval |
| `STEPS` | `seed,prepare,train,eval,error_detailed,error_in_depth,compare` | Pipeline steps (see below) |
| `GPU_ID` | `0` | GPU index (-1 for CPU if supported) |

## Pipeline steps

1. **split** — group-level train/test CSV (`scripts/split_per_group.py`)
2. **preprocess** — embeddings + design matrices → `outputs/.../processed/`
3. **prepare** — copy npy into per-model dirs (DisCo uses annotator index only)
4. **train** — DiADEM or DisCo baseline
5. **eval** — item-level ensemble eval (`eval_voiced`-style: cmatrix, preds.xlsx, op_*.tsv)
6. **error_detailed** — full paper metrics + plots → `error_analysis/`
7. **error_in_depth** — bootstrap CIs, NLL, Brier → `error_analysis_in_depth/`
8. **compare** — Markdown table with bold best values → `comparison/results_table.md`

## Models

| Key | Class | Annotator input |
|-----|-------|-----------------|
| `diadem` | `DISCO_DemographicWeighted` | Demographic one-hots + optional AnnotatorID (voiced) |
| `disco` | `DISCO` | Annotator index embedding only |

Add more baselines later by extending `configs/` and `diadem/train.py`.

## Data sources

Dataset configs point to CSVs in the **parent repo** (no duplicate raw data):

- DICES: `../dices/datasets/dices/`
- VOICED: `../voices/datasets/voiced/`

Split CSVs: `dices_split_{by_annotator,by_item}.csv`, `voiced_split_*.csv`

## Paper output

After `./run.sh`, open:

```
outputs/voiced/by_annotator/comparison/results_table.md
```

Copy-paste into your paper. Each model also has `eval_summary.json` with full metrics.

## Requirements

Same as parent project: TensorFlow, numpy, pandas, scikit-learn, sentence-transformers, tqdm.

```bash
conda activate lewidi
```

## Notes

- Disagreement loss uses **per-row labels** (`argmax(y)`) — patched vs old repo.
- Preprocess is unified; dataset differences are only in `configs/datasets/*.json`.
- To add a dataset: copy a JSON config, add raw CSV path, run with `DATASET=yourname`.
