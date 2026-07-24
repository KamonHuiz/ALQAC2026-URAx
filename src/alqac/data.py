"""Data layer: test cases, merged law corpus, article<->aid mapping, law-name resolution.

Key facts this module encodes (verified against the provided data):
  * Articles inside a law are stored in order Điều 1, 2, 3 ... with contiguous aids,
    so the N-th article (1-indexed) is "Điều N" and its aid is content[N-1]['aid'].
  * The public corpus is reused for the private test; we merge both corpora and
    de-duplicate by (law_id, aid).
  * Gold `related_law_provisions` use inconsistent free-text law names; we resolve
    them to law_ids with an explicit-code-first, keyword-fallback matcher.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import Config
from .utils import (LOG, read_json, norm_key, find_law_codes, find_short_codes,
                    normalize_ws)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


# --------------------------------------------------------------------------- #
# Case container
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    case_id: str
    case_query: str = ""
    A_role: str = "Nguyên đơn"
    B_role: str = "Bị đơn"
    A_description: str = ""
    B_description: str = ""
    court: str = ""
    case_type: str = ""
    # gold fields — present only on the public (labelled) split
    verdict_label: Optional[str] = None
    related_law_provisions: str = ""
    case_fact: str = ""
    court_reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_gold(self) -> bool:
        return self.verdict_label is not None

    @classmethod
    def from_raw(cls, d: dict[str, Any]) -> "Case":
        return cls(
            case_id=d.get("case_id") or d.get("id"),
            case_query=d.get("case_query", "") or "",
            A_role=d.get("A_role", "Nguyên đơn") or "Nguyên đơn",
            B_role=d.get("B_role", "Bị đơn") or "Bị đơn",
            A_description=d.get("A_description", "") or "",
            B_description=d.get("B_description", "") or "",
            court=d.get("court", "") or "",
            case_type=d.get("case_type", "") or "",
            verdict_label=d.get("verdict_label"),
            related_law_provisions=d.get("related_law_provisions", "") or "",
            case_fact=d.get("case_fact", "") or "",
            court_reasoning=d.get("court_reasoning", "") or "",
            raw=d,
        )


# --------------------------------------------------------------------------- #
# Law corpus
# --------------------------------------------------------------------------- #
class LawCorpus:
    def __init__(self, laws: list[dict], law_meta: dict[str, dict]):
        # by_law[law_id] = list of {'aid', 'content_Article'} in article order
        self.by_law: dict[str, list[dict]] = {}
        for law in laws:
            lid = law["law_id"]
            self.by_law.setdefault(lid, law["content"])
        self.law_meta = law_meta

        # flat list + indices
        self.articles: list[dict] = []          # {law_id, aid, num, text}
        self.aid_text: dict[tuple[str, int], str] = {}
        self.num_to_aid: dict[str, dict[int, int]] = {}
        for lid, content in self.by_law.items():
            num_map: dict[int, int] = {}
            for i, art in enumerate(content):
                aid = int(art["aid"])
                num = i + 1                      # position => "Điều num"
                text = normalize_ws(art["content_Article"])
                self.articles.append({"law_id": lid, "aid": aid, "num": num, "text": text})
                self.aid_text[(lid, aid)] = text
                num_map[num] = aid
            self.num_to_aid[lid] = num_map
        LOG.info("LawCorpus: %d laws, %d articles", len(self.by_law), len(self.articles))

    # ---- article number <-> aid -----------------------------------------
    def article_num_to_aid(self, law_id: str, num: int) -> Optional[int]:
        return self.num_to_aid.get(law_id, {}).get(int(num))

    def text_of(self, law_id: str, aid: int) -> str:
        return self.aid_text.get((law_id, int(aid)), "")

    # ---- law-name resolution --------------------------------------------
    def match_law_id(self, name_str: str) -> Optional[str]:
        """Resolve a free-text law name to a corpus law_id (or None)."""
        # 1) explicit full code present in the string
        for code in find_law_codes(name_str):
            if code in self.by_law:
                return code
        # 2) short code like '91/2015'
        for sc in find_short_codes(name_str):
            for lid in self.by_law:
                if lid.startswith(sc + "/"):
                    return lid
        # 3) keyword rules
        norm = norm_key(name_str)
        ym = _YEAR_RE.search(norm)
        year = int(ym.group(0)) if ym else None
        best = None
        best_score = -1
        for lid, meta in self.law_meta.items():
            if lid not in self.by_law:
                continue
            musts = meta.get("must", []) or []
            if any(m not in norm for m in musts):
                continue
            any_kw = meta.get("must_any", []) or []
            if any_kw and not any(a in norm for a in any_kw):
                continue
            if any(mn in norm for mn in (meta.get("must_not", []) or [])):
                continue
            meta_year = meta.get("year")
            # if the citation states a year, require it to match (prevents 2005<->2015 mixups)
            if year and meta_year and year != meta_year:
                continue
            score = len(musts) + len(any_kw)
            if year and meta_year == year:
                score += 5
            if score > best_score:
                best_score, best = score, lid
        return best

    def resolve_citation(self, law_name: str, article_num: int) -> Optional[tuple[str, int]]:
        lid = self.match_law_id(law_name)
        if lid is None:
            return None
        aid = self.article_num_to_aid(lid, article_num)
        if aid is None:
            return None
        return (lid, aid)

    def parse_gold_provisions(self, blob: str) -> list[tuple[str, int]]:
        """'Bộ luật Dân sự năm 2015 | Điều 584\n...' -> [(law_id, aid), ...] (mappable only)."""
        out: list[tuple[str, int]] = []
        for line in (blob or "").splitlines():
            if "|" not in line:
                continue
            name, art = line.split("|", 1)
            m = re.search(r"(\d{1,3})", art)
            if not m:
                continue
            res = self.resolve_citation(name.strip(), int(m.group(1)))
            if res and res not in out:
                out.append(res)
        return out


# --------------------------------------------------------------------------- #
# Data manager — one object holding everything the pipeline needs
# --------------------------------------------------------------------------- #
class DataManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        d = cfg.data

        # --- law-name metadata + merged corpus ---
        law_meta = read_json(cfg.resolve_path(d.law_name_map_file))["laws"]
        pub_corpus = read_json(cfg.resolve_path(d.public_corpus_file))
        priv_corpus = read_json(cfg.resolve_path(d.private_corpus_file))
        merged = self._merge_corpora(pub_corpus, priv_corpus)
        self.corpus = LawCorpus(merged, law_meta)

        # --- query bank ---
        self.query_bank = read_json(cfg.resolve_path(d.query_bank_file))

        # --- public (labelled) set: precedents + validation ---
        self.public_cases: list[Case] = [
            Case.from_raw(x) for x in read_json(cfg.resolve_path(d.public_test_file))
        ]
        LOG.info("Loaded %d public (labelled) cases", len(self.public_cases))

        # --- target split (what we produce a submission for) ---
        self.target_cases: list[Case] = self._load_target_split()
        LOG.info("Target split '%s': %d cases", cfg.run.target_split, len(self.target_cases))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_corpora(pub: list[dict], priv: list[dict]) -> list[dict]:
        """Union both corpora keyed by law_id; within a law, union articles by aid."""
        by_law: dict[str, dict[int, dict]] = {}
        order: list[str] = []
        for corpus in (pub, priv):
            for law in corpus:
                lid = law["law_id"]
                if lid not in by_law:
                    by_law[lid] = {}
                    order.append(lid)
                for art in law["content"]:
                    aid = int(art["aid"])
                    by_law[lid].setdefault(aid, {"aid": aid,
                                                 "content_Article": art["content_Article"]})
        merged = []
        for lid in order:
            arts = sorted(by_law[lid].values(), key=lambda a: a["aid"])
            merged.append({"law_id": lid, "content": arts})
        return merged

    # ------------------------------------------------------------------ #
    def _load_target_split(self) -> list[Case]:
        split = self.cfg.run.target_split
        if split == "public":
            return list(self.public_cases)
        # private -> configured path (resolved against repo root), else auto-detect in
        # <drive_root>/input/, else fall back to the public split.
        from pathlib import Path
        path = self.cfg.get("data.private_test_file")
        if path:
            rp = self.cfg.resolve_path(path)
            path = rp if Path(rp).exists() else self._autodetect_private_test()
        else:
            path = self._autodetect_private_test()
        if not path or not Path(path).exists():
            LOG.warning("No private test file found; falling back to PUBLIC split. "
                        "Place the 60-case JSON in <drive_root>/input/ or set data.private_test_file.")
            return list(self.public_cases)
        raw = read_json(path)
        cases = [Case.from_raw(x) for x in raw]
        LOG.info("Loaded private test from %s (%d cases)", path, len(cases))
        return cases

    def _autodetect_private_test(self) -> Optional[str]:
        from pathlib import Path
        root = Path(self.cfg.run.drive_root)
        for folder in (root / "input", root):
            if not folder.exists():
                continue
            for p in sorted(folder.glob("*.json")):
                try:
                    data = read_json(p)
                except Exception:
                    continue
                if (isinstance(data, list) and data and isinstance(data[0], dict)
                        and "case_query" in data[0]):
                    return str(p)
        return None
