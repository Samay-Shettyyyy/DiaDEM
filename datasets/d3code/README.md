# D3CODE in cleanedup_diadem

Source: [google-research-datasets/D3code](https://github.com/google-research-datasets/D3code)

## Files

| File | Role |
|------|------|
| `d3-items.csv` | item text + category |
| `d3-raters.csv` | demographics + moral foundations |
| `d3-ratings.csv` | item×rater labels |
| `d3code_complete.csv` | joined table (built by script) |
| `d3code_split_by_*.csv` | train/test splits |

## Build / run

```bash
cd cleanedup_diadem
python scripts/build_d3code_complete.py   # already done once after clone

DATASET=d3code SPLIT_MODE=by_annotator \
  STEPS=split,preprocess,prepare,train,eval,error_detailed,error_in_depth,compare \
  sinteractive run.sh
```

Notes:
- Label = `rating_binary` (0/1); raw=-1 rows dropped
- AnnotatorID one-hot **off** (~4k annotators)
- Continuous moral scores (care…purity) min-max normalized from [1,5] → [0,1]
- Categorical: Age, Gender, Region, Country, SES
