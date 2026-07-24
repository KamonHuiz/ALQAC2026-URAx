"""Stage 4 — Law Provision Retrieval.

Produces the precise set of {law_id, aid} the court would cite, optimising micro-F1
(precision matters). Three complementary signals:
  1. Citation extraction: regex "Điều N ... của <luật>" over retrieved segments, resolved
     to (law_id, aid) exactly via the corpus article-order map. High precision when present.
  2. Hybrid semantic (bge-m3) + lexical (BM25) retrieval over the merged corpus, seeded by
     the extracted legal issues.
  3. LLM rerank/prune of the candidate pool to the final set, capped for precision.
Optional procedural priors (án phí / kháng cáo / xét xử vắng mặt) are offered as candidates.
"""
from __future__ import annotations

import re

import regex  # better unicode support

from .prompts import law_select
from .utils import LOG, extract_json, normalize_ws

# "Điều 584, Điều 590 và Điều 603 của Bộ luật Dân sự năm 2015"
_CIT_RE = regex.compile(
    r"((?:Điều\s+\d{1,3}[\s,;\.và]*)+)"
    r"[^\.]{0,25}?"
    r"(Bộ luật[^\.,;\n]{0,45}|Luật[^\.,;\n]{0,45}|Nghị\s+(?:định|quyết)[^\.,;\n]{0,45}"
    r"|Thông\s+tư[^\.,;\n]{0,45}|\d{1,3}/\d{4}/[A-Za-zĐ\-]+)",
    regex.IGNORECASE)
_DIEU_RE = re.compile(r"Điều\s+(\d{1,3})", re.IGNORECASE)


class LawRetriever:
    def __init__(self, cfg, llm, corpus, dense_index, bm25_index):
        self.cfg = cfg.law
        self.llm = llm
        self.corpus = corpus
        self.dense = dense_index
        self.bm25 = bm25_index
        self.priors = [tuple(p) for p in (cfg.law.procedural_prior.to_dict()
                       if hasattr(cfg.law.procedural_prior, "to_dict")
                       else cfg.law.procedural_prior)] if cfg.law.use_procedural_priors else []

    # ------------------------------------------------------------------ #
    def extract_citations(self, text: str) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for m in _CIT_RE.finditer(text or ""):
            nums = [int(x) for x in _DIEU_RE.findall(m.group(1))]
            law_ref = m.group(2)
            lid = self.corpus.match_law_id(law_ref)
            if not lid:
                continue
            for n in nums:
                aid = self.corpus.article_num_to_aid(lid, n)
                if aid and (lid, aid) not in out:
                    out.append((lid, aid))
        return out

    # ------------------------------------------------------------------ #
    def _candidates(self, query: str) -> list[dict]:
        pool: dict[tuple[str, int], dict] = {}
        if self.cfg.use_semantic_retrieval:
            for _s, art in self.dense.search(query, self.cfg.semantic_top_k):
                pool[(art["law_id"], art["aid"])] = art
            for _s, art in self.bm25.search(query, self.cfg.bm25_top_k):
                pool.setdefault((art["law_id"], art["aid"]), art)
        # procedural priors as candidates
        for lid, num in self.priors:
            aid = self.corpus.article_num_to_aid(lid, num)
            if aid:
                key = (lid, aid)
                if key not in pool:
                    pool[key] = {"law_id": lid, "aid": aid, "num": num,
                                 "text": self.corpus.text_of(lid, aid)}
        return list(pool.values())

    @staticmethod
    def _query_from_record(rec: dict) -> str:
        issues = rec.get("van_de_phap_ly", [])
        issues = "; ".join(issues) if isinstance(issues, list) else str(issues)
        return normalize_ws(f"{issues}. {rec.get('tom_tat','')} "
                            f"{rec.get('nguyen_don_yeu_cau','')}")[:500]

    # ------------------------------------------------------------------ #
    def retrieve(self, cases, records: list[dict], chunks_list: list[dict]) -> list[list[dict]]:
        # 1) citation extraction per case
        cited: list[list[tuple[str, int]]] = []
        for chunks in chunks_list:
            text = " ".join(v["text"] for v in chunks.values()) if chunks else ""
            cited.append(self.extract_citations(text) if self.cfg.use_citation_extraction else [])

        # 2) candidate pools + optional LLM rerank
        cand_lists = [self._candidates(self._query_from_record(r)) for r in records]
        chosen_sets: list[list[tuple[str, int]]] = []
        if self.cfg.llm_rerank:
            prompts_batch, meta = [], []
            for c, rec, cands in zip(cases, records, cand_lists):
                block = self._render_candidates(cands)
                prompts_batch.append(law_select(rec.get("tom_tat", c.case_query),
                                                "; ".join(rec.get("van_de_phap_ly", [])
                                                if isinstance(rec.get("van_de_phap_ly"), list)
                                                else [str(rec.get("van_de_phap_ly", ""))]),
                                                block))
                meta.append(cands)
            raws = self.llm.chat_batch1(prompts_batch, thinking=False, max_tokens=256)
            for raw, cands in zip(raws, meta):
                chosen_sets.append(self._parse_chosen(raw, cands))
        else:
            chosen_sets = [[(a["law_id"], a["aid"]) for a in cands[:self.cfg.max_articles]]
                           for cands in cand_lists]

        # 3) merge citations (priority) + chosen, cap
        final: list[list[dict]] = []
        for cit, chosen in zip(cited, chosen_sets):
            merged: list[tuple[str, int]] = []
            for pair in cit + chosen:
                if pair not in merged:
                    merged.append(pair)
            merged = merged[:self.cfg.max_articles]
            final.append([{"law_id": lid, "aid": aid} for lid, aid in merged])
        return final

    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_candidates(cands: list[dict]) -> str:
        lines = []
        for i, a in enumerate(cands):
            snippet = a["text"][:140].replace("\n", " ")
            lines.append(f"[{i}] {a['law_id']} | Điều {a['num']} | {snippet}")
        return "\n".join(lines) if lines else "(không có ứng viên)"

    @staticmethod
    def _parse_chosen(raw: str, cands: list[dict]) -> list[tuple[str, int]]:
        obj = extract_json(raw) or {}
        idxs = obj.get("chosen", []) if isinstance(obj, dict) else []
        out = []
        for i in idxs:
            try:
                a = cands[int(i)]
            except (ValueError, IndexError, TypeError):
                continue
            pair = (a["law_id"], a["aid"])
            if pair not in out:
                out.append(pair)
        return out
