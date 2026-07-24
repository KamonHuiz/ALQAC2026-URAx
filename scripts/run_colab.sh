#!/usr/bin/env bash
# ============================================================
#  One-shot Colab runner for URAx-LACE (ALQAC 2026).
#  Usage (inside a Colab cell, after `git clone` + `cd ALQAC2026-URAx`):
#     !bash scripts/run_colab.sh --split private --group URAx
#  Any extra args are forwarded to scripts/run_pipeline.py.
# ============================================================
set -e

echo ">>> [1/3] Installing dependencies (this takes a few minutes on first run) ..."
pip install -q -U pip
# vLLM pulls a compatible torch; install it first, then the rest.
pip install -q "vllm>=0.6.3"
pip install -q -r requirements.txt

echo ">>> [2/3] Environment:"
python - <<'PY'
import torch, os
print("  torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0),
          f"| {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
PY

echo ">>> [3/3] Running pipeline ..."
python scripts/run_pipeline.py "$@"
echo ">>> Done. Check your Drive: ALQAC_RESULT/<run_id>/[submission] <group>.json"
