"""Stage 3b — Outcome prediction.

Combines several signals into a single 4-way label:
  * Dual-Advocate Debate: an LLM argues for A, another for B, an adjudicator decides.
  * Self-Consistency: the adjudicator is sampled N times; labels are majority-voted.
  * VKS-prior fusion: the prosecutor's recommendation (a strong predictor of Vietnamese
    civil verdicts) is mapped to a label and injected as weighted votes.
  * Class-prior tie-break: public-set base rates settle ties.

Everything is batched across cases for throughput.
"""
from __future__ import annotations

import re
from collections import Counter

from . import prompts
from .prompts import LABELS
from .utils import LOG, strip_think

_VKS_TO_LABEL = {
    "ACCEPT_FULL": "A_WIN",
    "ACCEPT_PARTIAL": "PARTIAL_A_WIN",
    "REJECT": "B_WIN",
}
_CONCLUSION_RE = re.compile(r"K[ẾE]T\s*LU[ẬA]N\s*[:\-]?\s*([A-Z_]+)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Z_]+")


def parse_label(text: str) -> str:
    """Extract a canonical label from an adjudicator completion.

    Careful: 'B_WIN' is a substring of 'PARTIAL_B_WIN' (and 'A_WIN' of 'PARTIAL_A_WIN'),
    so we match on whole [A-Z_] tokens rather than substrings.
    """
    t = strip_think(text).upper()
    m = _CONCLUSION_RE.search(t)
    if m and m.group(1) in LABELS:          # explicit "KẾT LUẬN: <label>"
        return m.group(1)
    # else: last whole-token label appearing anywhere (the conclusion is usually last)
    valid = [tok for tok in _TOKEN_RE.findall(t) if tok in LABELS]
    return valid[-1] if valid else ""


class OutcomePredictor:
    def __init__(self, cfg, llm, precedents):
        self.cfg = cfg.outcome
        self.llm = llm
        self.precedents = precedents
        self.class_prior = dict(cfg.outcome.class_prior.to_dict()
                                if hasattr(cfg.outcome.class_prior, "to_dict")
                                else cfg.outcome.class_prior)

    # ------------------------------------------------------------------ #
    def predict(self, cases, summaries: list[str], records: list[dict],
                exclude_self: bool = False) -> list[dict]:
        # precedent blocks
        prec_blocks = []
        for c in cases:
            prec = (self.precedents.retrieve(c.case_query,
                    exclude_case_id=c.case_id if exclude_self else None)
                    if self.cfg.use_precedents else [])
            prec_blocks.append(self.precedents.format_block(prec) if self.cfg.use_precedents
                               else "(tắt)")

        vks_hints = []
        for rec in records:
            hint = rec.get("vks_de_nghi", "") or f"(stance: {rec.get('vks_stance','UNKNOWN')})"
            vks_hints.append(hint[:600])

        # ---- optional dual-advocate debate ----
        if self.cfg.use_debate:
            args_a = self.llm.chat_batch1(
                [prompts.advocate("A", s, pb) for s, pb in zip(summaries, prec_blocks)],
                thinking=False, max_tokens=400)
            args_b = self.llm.chat_batch1(
                [prompts.advocate("B", s, pb) for s, pb in zip(summaries, prec_blocks)],
                thinking=False, max_tokens=400)
        else:
            args_a = ["(tắt)"] * len(cases)
            args_b = ["(tắt)"] * len(cases)

        # ---- adjudicator with self-consistency ----
        if self.cfg.use_debate:
            adj_prompts = [prompts.adjudicator(s, pb, aa, ab, vh)
                           for s, pb, aa, ab, vh in
                           zip(summaries, prec_blocks, args_a, args_b, vks_hints)]
        else:
            adj_prompts = [prompts.direct_predict(s, pb, vh)
                           for s, pb, vh in zip(summaries, prec_blocks, vks_hints)]

        n = max(1, int(self.cfg.self_consistency_samples))
        # thinking follows the engine default (model.enable_thinking); budget generous so
        # a <think> trace never truncates the final "KẾT LUẬN:" line.
        samples = self.llm.chat_batch(adj_prompts, n=n, temperature=0.7, max_tokens=2048)

        results = []
        for c, group, rec in zip(cases, samples, records):
            votes = Counter()
            for comp in group:
                lab = parse_label(comp)
                if lab:
                    votes[lab] += 1
            self._fuse_vks(votes, rec, n)
            label = self._decide(votes)
            results.append({
                "case_id": c.case_id,
                "prediction": label,
                "llm_votes": dict(votes),
                "vks_stance": rec.get("vks_stance", "UNKNOWN"),
            })
        return results

    # ------------------------------------------------------------------ #
    def _fuse_vks(self, votes: Counter, rec: dict, n: int) -> None:
        if not self.cfg.use_vks_fusion:
            return
        stance = rec.get("vks_stance", "UNKNOWN")
        target = _VKS_TO_LABEL.get(stance)
        if not target:
            return
        w = int(round(float(self.cfg.vks_prior_weight) * n))
        if w > 0:
            votes[target] += w

    def _decide(self, votes: Counter) -> str:
        if not votes:
            # nothing parsed -> fall back to the most likely class a priori
            return max(self.class_prior, key=self.class_prior.get)
        top = max(votes.values())
        tied = [lab for lab, v in votes.items() if v == top]
        if len(tied) == 1:
            return tied[0]
        return max(tied, key=lambda lab: self.class_prior.get(lab, 0.0))
