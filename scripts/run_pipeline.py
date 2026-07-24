#!/usr/bin/env python
"""CLI entry point for the URAx-LACE ALQAC 2026 pipeline.

Examples
--------
# Full private run (the real submission), saving to Drive:
python scripts/run_pipeline.py --split private --group URAx

# Fast offline validation on the labelled public set (no API calls):
python scripts/run_pipeline.py --split public --no-api

# Resume a run and recompute only the prediction stage:
python scripts/run_pipeline.py --run-id 2026-07-24_1830 --force predict
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alqac.config import Config          # noqa: E402
from alqac.pipeline import Pipeline      # noqa: E402
from alqac.utils import LOG              # noqa: E402


def maybe_mount_drive(drive_root: str) -> None:
    """On Colab, mount Google Drive automatically so `bash run_colab.sh` is one-shot."""
    if not drive_root.startswith("/content/drive"):
        return
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore
        LOG.info("Mounting Google Drive ...")
        drive.mount("/content/drive")
    except Exception as e:
        LOG.warning("Could not mount Drive automatically (%s). Mount it manually.", e)


def build_overrides(a) -> dict:
    ov: dict = {"run": {}, "model": {}, "retrieval": {}}
    if a.split:
        ov["run"]["target_split"] = a.split
    if a.run_id:
        ov["run"]["run_id"] = a.run_id
    if a.group:
        ov["run"]["group_name"] = a.group
    if a.drive_root:
        ov["run"]["drive_root"] = a.drive_root
    if a.model:
        ov["model"]["name"] = a.model
    # backend precedence: --backend flag > ALQAC_BACKEND env (set by run_colab.sh when it
    # detects vLLM is unusable) > config default
    backend = a.backend or os.environ.get("ALQAC_BACKEND")
    if backend:
        ov["model"]["backend"] = backend
    if a.no_api:
        ov["retrieval"]["use_api"] = False
    return ov


def main() -> None:
    ap = argparse.ArgumentParser(description="URAx-LACE ALQAC 2026 pipeline")
    ap.add_argument("--config", default=None, help="path to a yaml config (default: configs/default.yaml)")
    ap.add_argument("--split", choices=["private", "public"], default=None)
    ap.add_argument("--run-id", default=None, help="reuse to resume a run")
    ap.add_argument("--group", default=None, help="team/group name for the submission file")
    ap.add_argument("--drive-root", default=None)
    ap.add_argument("--model", default=None, help="override model.name (any <10B open-weight)")
    ap.add_argument("--backend", choices=["vllm", "hf"], default=None)
    ap.add_argument("--no-api", action="store_true", help="offline mode (public split only)")
    ap.add_argument("--force", nargs="*", default=[],
                    help="stages to recompute: queries understand predict law")
    args = ap.parse_args()

    cfg = Config.load(args.config, overrides=build_overrides(args))
    maybe_mount_drive(cfg.run.drive_root)

    pipe = Pipeline(cfg)
    result = pipe.run(force_stages=set(args.force))

    rep = result["report"]
    print("\n" + "=" * 60)
    print(f"DONE. Submission + artifacts in: {result['run_dir']}")
    if rep.get("outcome"):
        print(f"  Public outcome accuracy : {rep['outcome'].get('accuracy'):.3f}")
    if rep.get("law"):
        print(f"  Public law micro-F1     : {rep['law'].get('micro_f1'):.3f}")
    print(f"  FinalScore estimate     : {rep.get('final_score_estimate')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
