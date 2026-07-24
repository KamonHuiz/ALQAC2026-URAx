"""Stage 3a — Precedent memory bank for Case-Based Reasoning.

Indexes the 50 labelled public cases by their case_query embedding. For each target
case we retrieve the most similar precedents and surface their GOLD outcome + a snippet
of the court's reasoning, giving the predictor grounded analogies ("how were similar
disputes actually decided, and why").
"""
from __future__ import annotations

from typing import Optional

from .embedder import DenseIndex
from .utils import normalize_ws


class PrecedentBank:
    def __init__(self, cfg, embedder, public_cases):
        self.k = int(cfg.outcome.num_precedents)
        payloads = []
        texts = []
        for c in public_cases:
            texts.append(c.case_query)
            payloads.append({
                "case_id": c.case_id,
                "label": c.verdict_label,
                "query": normalize_ws(c.case_query)[:500],
                "reasoning": normalize_ws(c.court_reasoning)[:400],
            })
        self.index = DenseIndex(embedder, texts, payloads)

    def retrieve(self, query: str, exclude_case_id: Optional[str] = None) -> list[dict]:
        hits = self.index.search(query, top_k=self.k + 1)
        out = []
        for _score, p in hits:
            if exclude_case_id and p["case_id"] == exclude_case_id:
                continue
            out.append(p)
            if len(out) >= self.k:
                break
        return out

    @staticmethod
    def format_block(precedents: list[dict]) -> str:
        if not precedents:
            return "(không có án lệ tham khảo)"
        lines = []
        for i, p in enumerate(precedents, 1):
            lines.append(f"[Án lệ {i}] {p['query']}\n   -> Kết quả THỰC TẾ: {p['label']}"
                         f"\n   -> Lý do Tòa: {p['reasoning']}")
        return "\n".join(lines)
