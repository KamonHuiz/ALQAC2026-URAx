"""Validation harness for the labelled public split.

Computes the two components we can measure offline:
  * Outcome Accuracy (exact match over the 4 labels)          -> 70% of FinalScore
  * Law micro-F1 over (case, law_id, aid) pairs               -> 10% of FinalScore
Penalized Case Recall (20%) needs the hidden gold chunk ids, so we report distinct-chunk
coverage instead. A FinalScore *estimate* uses accuracy + lawF1 with a placeholder recall.
"""
from __future__ import annotations

from collections import Counter

from .prompts import LABELS
from .utils import LOG


def evaluate_outcome(cases, predictions: dict) -> dict:
    labelled = [c for c in cases if c.has_gold]
    if not labelled:
        return {}
    correct = 0
    conf = Counter()
    per_class = {lab: [0, 0] for lab in LABELS}  # [correct, total]
    for c in labelled:
        gold = c.verdict_label
        pred = predictions.get(c.case_id)
        per_class.setdefault(gold, [0, 0])[1] += 1
        if pred == gold:
            correct += 1
            per_class[gold][0] += 1
        else:
            conf[(gold, pred)] += 1
    acc = correct / len(labelled)
    return {
        "n": len(labelled),
        "accuracy": acc,
        "per_class_recall": {k: (v[0] / v[1] if v[1] else 0.0) for k, v in per_class.items()},
        "top_confusions": conf.most_common(6),
    }


def evaluate_law(cases, law_evidence: dict, corpus) -> dict:
    labelled = [c for c in cases if c.has_gold and c.related_law_provisions]
    if not labelled:
        return {}
    tp = fp = fn = 0
    for c in labelled:
        gold = set(corpus.parse_gold_provisions(c.related_law_provisions))
        pred = set((le["law_id"], int(le["aid"])) for le in law_evidence.get(c.case_id, []))
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": len(labelled), "micro_precision": prec, "micro_recall": rec,
            "micro_f1": f1, "tp": tp, "fp": fp, "fn": fn}


def report(cases, predictions: dict, law_evidence: dict, caches: dict, corpus) -> dict:
    out = {"outcome": evaluate_outcome(cases, predictions),
           "law": evaluate_law(cases, law_evidence, corpus)}
    # distinct-chunk coverage (proxy for case recall)
    chunk_counts = [len(caches.get(c.case_id, {}).get("chunks", {})) for c in cases]
    call_counts = [caches.get(c.case_id, {}).get("calls", 0) for c in cases]
    if chunk_counts:
        out["retrieval"] = {
            "avg_chunks": sum(chunk_counts) / len(chunk_counts),
            "avg_calls": sum(call_counts) / len(call_counts),
        }
    acc = out["outcome"].get("accuracy", 0.0)
    lawf1 = out["law"].get("micro_f1", 0.0)
    # FinalScore estimate assuming full case-recall credit (E_i=1, recall placeholder=0.8)
    out["final_score_estimate"] = round(0.70 * acc + 0.20 * 0.8 + 0.10 * lawf1, 4)
    _log(out)
    return out


def _log(out: dict) -> None:
    o, l = out.get("outcome", {}), out.get("law", {})
    if o:
        LOG.info("== Outcome: acc=%.3f (n=%d) | per-class=%s",
                 o["accuracy"], o["n"], {k: round(v, 2) for k, v in o["per_class_recall"].items()})
        LOG.info("   confusions (gold->pred): %s", o["top_confusions"])
    if l:
        LOG.info("== Law: microF1=%.3f (P=%.3f R=%.3f) tp=%d fp=%d fn=%d",
                 l["micro_f1"], l["micro_precision"], l["micro_recall"], l["tp"], l["fp"], l["fn"])
    if "retrieval" in out:
        LOG.info("== Retrieval: avg_chunks=%.1f avg_calls=%.1f",
                 out["retrieval"]["avg_chunks"], out["retrieval"]["avg_calls"])
    LOG.info("== FinalScore estimate (recall placeholder 0.8): %.4f", out["final_score_estimate"])
