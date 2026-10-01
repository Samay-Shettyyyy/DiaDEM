#!/bin/bash -l
# Single entry point for cleanedup_diadem — run from THIS directory only.
#
# Examples:
#   sinteractive run.sh                         # from this directory — that's it
#   DATASET=dices SPLIT_MODE=by_item ./run.sh
#   STEPS=compare ./run.sh                      # re-generate paper table only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- knobs (override via env) ---
DATASET="${DATASET:-voiced}"              # dices | voiced | d3code
SPLIT_MODE="${SPLIT_MODE:-by_annotator}"  # by_annotator | by_item
# Models: diadem,disco + majority_similar + LLM judges: llm_gemma4,llm_gpt_oss,llm_qwen8b,llm_deepseek
MODELS="${MODELS:-diadem,disco}"
# For d3code (no legacy npy), use: STEPS=split,preprocess,prepare,train,eval,error_detailed,error_in_depth,compare
STEPS="${STEPS:-seed,prepare,train,eval,error_detailed,error_in_depth,compare}"
GPU_ID="${GPU_ID:-0}"
# Optional OpenAI-compatible endpoint for huge models (DeepSeek etc.):
#   export LLM_API_BASE=http://localhost:8000/v1
#   export LLM_API_KEY=EMPTY


conda activate lewidi 2>/dev/null || true

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# TensorFlow XLA needs CUDA nvvm/libdevice + ptxas (not just the GPU driver).
# Prefer the pip nvidia-cuda-nvcc package; do NOT `module load cuda/6.5` (too old).
_NVCC_ROOT="${CONDA_PREFIX:-$HOME/miniconda3/envs/lewidi}/lib/python3.10/site-packages/nvidia/cuda_nvcc"
if [[ -d "${_NVCC_ROOT}/nvvm/libdevice" ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${_NVCC_ROOT}${XLA_FLAGS:+ ${XLA_FLAGS}}"
  export PATH="${_NVCC_ROOT}/bin:${PATH}"
  echo "[run.sh] XLA CUDA toolkit: ${_NVCC_ROOT}"
fi

echo "=========================================="
echo " cleanedup_diadem pipeline"
echo " dataset=$DATASET split=$SPLIT_MODE"
echo " models=$MODELS"
echo " steps=$STEPS"
echo " gpu_id=$GPU_ID"
echo " cwd=$SCRIPT_DIR"
echo "=========================================="
echo "[run.sh] activating conda env lewidi (if available) ..."
echo "[run.sh] launching: python -m diadem.pipeline ..."
echo ""

python -m diadem.pipeline \
  --dataset "$DATASET" \
  --split-mode "$SPLIT_MODE" \
  --models "$MODELS" \
  --steps "$STEPS" \
  --gpu-id "$GPU_ID"

echo ""
echo "Done. Results:"
echo "  outputs/${DATASET}/${SPLIT_MODE}/diadem/error_analysis/analysis_summary.json"
echo "  outputs/${DATASET}/${SPLIT_MODE}/diadem/error_analysis_in_depth/analysis_summary_in_depth.json"
echo "  outputs/${DATASET}/${SPLIT_MODE}/comparison/results_table.md  (paper-ready)"
