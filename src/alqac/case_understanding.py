"""Stage 2 — Structured Case Understanding.

Condenses the (long, noisy) retrieved segments into a compact structured record and,
crucially, extracts the Viện kiểm sát (VKS) recommendation + a normalised stance, which
Stage 3 fuses as a strong prior. Runs batched across all cases for throughput.
"""
from __future__ import annotations

from natsort import natsorted

from .prompts import case_understanding as build_prompt
from .utils import extract_json, normalize_ws

_STANCES = {"ACCEPT_FULL", "ACCEPT_PARTIAL", "REJECT", "UNKNOWN"}


def evidence_from_chunks(chunks: dict, max_chars: int) -> str:
    """Concatenate cached segment texts in stable chunk order, capped to max_chars."""
    parts = [chunks[cid]["text"] for cid in natsorted(chunks.keys())]
    text = normalize_ws(" ".join(parts))
    return text[:max_chars]


class CaseUnderstander:
    def __init__(self, cfg, llm):
        self.cfg = cfg.understanding
        self.llm = llm

    def run(self, cases, evidence_texts: list[str]) -> list[dict]:
        prompts = [
            build_prompt(c.case_query, c.A_role, c.B_role, c.A_description,
                         c.B_description, ev)
            for c, ev in zip(cases, evidence_texts)
        ]
        raw = self.llm.chat_batch1(prompts, thinking=False, max_tokens=1024)
        out = []
        for c, txt in zip(cases, raw):
            rec = extract_json(txt) or {}
            if not isinstance(rec, dict):
                rec = {}
            rec.setdefault("tom_tat", c.case_query[:400])
            rec.setdefault("nguyen_don_yeu_cau", "")
            rec.setdefault("bi_don_phan_hoi", "")
            rec.setdefault("tranh_chap_chinh", [])
            rec.setdefault("chung_cu_quan_trong", [])
            rec.setdefault("so_tien_tai_san", "")
            rec.setdefault("vks_de_nghi", "")
            rec.setdefault("van_de_phap_ly", [])
            st = str(rec.get("vks_stance", "UNKNOWN")).upper().strip()
            rec["vks_stance"] = st if st in _STANCES else "UNKNOWN"
            rec["case_id"] = c.case_id
            out.append(rec)
        return out

    @staticmethod
    def summary_text(rec: dict) -> str:
        """Compact human-readable summary used by downstream reasoning prompts."""
        def _join(x):
            return "; ".join(x) if isinstance(x, list) else str(x)
        lines = [
            f"Tóm tắt: {rec.get('tom_tat','')}",
            f"Yêu cầu của nguyên đơn (A): {rec.get('nguyen_don_yeu_cau','')}",
            f"Phản hồi của bị đơn (B): {rec.get('bi_don_phan_hoi','')}",
            f"Tranh chấp chính: {_join(rec.get('tranh_chap_chinh',[]))}",
            f"Chứng cứ: {_join(rec.get('chung_cu_quan_trong',[]))}",
            f"Số tiền/tài sản: {rec.get('so_tien_tai_san','')}",
        ]
        return "\n".join(lines)
