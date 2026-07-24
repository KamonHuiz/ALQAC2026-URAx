#!/usr/bin/env bash
# ============================================================
#  One-shot Colab runner for URAx-LACE (ALQAC 2026).
#  Usage (inside a Colab cell, after `git clone` + `cd ALQAC2026-URAx`):
#     !bash scripts/run_colab.sh --split private --group URAx
#  Any extra args are forwarded to scripts/run_pipeline.py.
# ============================================================
set -e

echo ">>> [1/4] Installing core dependencies (HF backend works with Colab's torch) ..."
pip install -q -U pip
pip install -q -U "transformers>=4.51.0" accelerate \
    FlagEmbedding sentence-transformers faiss-cpu rank-bm25 \
    requests numpy tqdm pyyaml natsort regex orjson

echo ">>> [2/4] Trying to enable vLLM (optional fast path) ..."
# vLLM is much faster but its wheel must match the runtime CUDA. On mismatch we simply
# use the transformers backend instead — the pipeline auto-falls-back either way.
export ALQAC_BACKEND=""
if python -c "import vllm" 2>/dev/null; then
    echo "    vLLM already importable."
elif pip install -q "vllm>=0.6.3" 2>/dev/null && python -c "import vllm" 2>/dev/null; then
    echo "    vLLM installed OK."
else
    echo "    vLLM not usable in this runtime (CUDA/torch mismatch) -> forcing HF backend."
    export ALQAC_BACKEND=hf
fi

echo ">>> [3/4] Environment:"
python - <<'PY'
import torch
print("  torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU: {p.name} | {p.total_memory/1e9:.0f} GB")
try:
    import vllm; print("  vLLM:", vllm.__version__)
except Exception as e:
    print("  vLLM: unavailable ->", type(e).__name__)
PY

echo ">>> [4/4] Running pipeline (ALQAC_BACKEND='${ALQAC_BACKEND}') ..."
python scripts/run_pipeline.py "$@"
echo ">>> Done. Check your Drive: ALQAC_RESULT/run_<split>/[submission] <group>.json"
