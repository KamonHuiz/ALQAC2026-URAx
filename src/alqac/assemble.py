"""Stage 5 — Assemble the submission.json.

Format (one object per test case):
    {"case_id", "prediction", "case_evidence": [chunk_id...], "law_evidence": [{law_id, aid}]}
case_evidence has no precision penalty in scoring, so we submit ALL cached chunk_ids to
maximise recall; the API-efficiency factor only depends on the calls already spent.
"""
from __future__ import annotations

from natsort import natsorted

from .prompts import LABELS


def build_submission(cases, predictions: dict, caches: dict, law_evidence: dict,
                     submit_all_chunks: bool = True) -> list[dict]:
    out = []
    for c in cases:
        pred = predictions.get(c.case_id, "PARTIAL_A_WIN")
        if pred not in LABELS:
            pred = "PARTIAL_A_WIN"
        chunks = caches.get(c.case_id, {}).get("chunks", {})
        case_ev = natsorted(set(chunks.keys())) if submit_all_chunks else []
        laws = law_evidence.get(c.case_id, [])
        # dedup law_evidence, ensure int aid
        seen, laws_clean = set(), []
        for le in laws:
            key = (le["law_id"], int(le["aid"]))
            if key not in seen:
                seen.add(key)
                laws_clean.append({"law_id": le["law_id"], "aid": int(le["aid"])})
        out.append({
            "case_id": c.case_id,
            "prediction": pred,
            "case_evidence": list(case_ev),
            "law_evidence": laws_clean,
        })
    return out
