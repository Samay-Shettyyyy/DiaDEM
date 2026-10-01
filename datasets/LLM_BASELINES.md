# LLM-as-a-judge baselines (zero-shot)

Text-only zero-shot judges. No annotator demographics in the prompt — so we
measure whether pretrained LMs recover human labels/distributions from text alone.

## Models (`configs/llm_models.json`)

| Key | HuggingFace ID |
|-----|----------------|
| `llm_gemma4` | `google/gemma-4-26B-A4B-it` |
| `llm_gpt_oss` | `openai/gpt-oss-20b` |
| `llm_qwen8b` | `Qwen/Qwen3-8B` |
| `llm_deepseek` | `deepseek-ai/DeepSeek-V4-Flash` |

DeepSeek-V4-Flash is a large MoE. Prefer serving with vLLM and:

```bash
export LLM_API_BASE=http://localhost:8000/v1
export LLM_API_KEY=EMPTY
```

## One-run examples

```bash
cd cleanedup_diadem

# Neural + all LLM judges on D3CODE
DATASET=d3code SPLIT_MODE=by_annotator \
  MODELS=diadem,llm_qwen8b,llm_gemma4,llm_gpt_oss,llm_deepseek \
  STEPS=prepare,train,eval,error_detailed,compare \
  sinteractive run.sh

# LLM-only (data already prepared)
DATASET=d3code MODELS=llm_qwen8b,llm_gemma4 \
  STEPS=prepare,eval,compare \
  sinteractive run.sh
```

Outputs land under `outputs/{dataset}/{split}/llm_*/eval_summary.json` and
`error_analysis/analysis_summary.json`, then into `comparison/results_table.md`.
