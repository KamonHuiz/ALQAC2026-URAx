"""Stage 1 — Economical Agentic Evidence Retrieval.

For each case we issue a carefully ordered set of BM25 queries to the Case Content API:
    VKS-focus queries  ->  party descriptions  ->  structural bank  ->  LLM-generated
and stop early once results saturate (no new chunk for `saturation_patience` calls),
capped at `max_calls_per_case`. This keeps the per-case call count c_i low so the
API-efficiency factor E_i stays at 1.0, while still covering the judgment's structure.

Everything is cached on Drive per case; a query already spent is never re-issued, so
re-running (after a Colab disconnect, or for a new experiment) costs ZERO extra API calls.
"""
from __future__ import annotations

from typing import Optional

from .utils import LOG


class RetrievalAgent:
    def __init__(self, cfg, api, data):
        self.cfg = cfg.retrieval
        self.api = api
        self.qbank = data.query_bank

    # ------------------------------------------------------------------ #
    def _ordered_queries(self, case, generated: list[str]) -> list[str]:
        qs: list[str] = []
        if self.cfg.use_query_bank:
            qs += list(self.qbank.get("vks_focus_queries", []))
            for slot in self.qbank.get("party_slots", []):
                if slot == "__A_DESCRIPTION__" and case.A_description:
                    qs.append(case.A_description)
                elif slot == "__B_DESCRIPTION__" and case.B_description:
                    qs.append(case.B_description)
            qs += list(self.qbank.get("structural_queries", []))
        if self.cfg.use_generated_queries:
            qs += list(generated or [])
        # de-dup preserving order
        seen, out = set(), []
        for q in qs:
            q = (q or "").strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out

    # ------------------------------------------------------------------ #
    def retrieve_case(self, case, generated: Optional[list[str]] = None) -> dict:
        cache = self.api.load_cache(case.case_id)
        if cache.get("complete"):
            LOG.info("[retrieve] %s cached (%d chunks, %d calls) — skip",
                     case.case_id, len(cache["chunks"]), cache["calls"])
            return cache

        done = set(cache.get("queries_done", []))
        queries = [q for q in self._ordered_queries(case, generated or []) if q not in done]

        no_new = 0
        min_calls = len(self.qbank.get("vks_focus_queries", [])) + 2  # ensure VKS ran
        for q in queries:
            if cache["calls"] >= self.cfg.max_calls_per_case:
                LOG.info("[retrieve] %s hit call budget (%d)", case.case_id, cache["calls"])
                break
            hit = self.api.call(q, case.case_id)
            cache["queries_done"].append(q)
            cache["calls"] += 1
            if hit and hit["chunk_id"] not in cache["chunks"]:
                cache["chunks"][hit["chunk_id"]] = {"text": hit["text"], "score": hit["score"]}
                no_new = 0
            else:
                no_new += 1
            if cache["calls"] % 5 == 0:
                self.api.save_cache(case.case_id, cache)
            if no_new >= self.cfg.saturation_patience and cache["calls"] >= min_calls:
                LOG.info("[retrieve] %s saturated after %d calls (%d chunks)",
                         case.case_id, cache["calls"], len(cache["chunks"]))
                break

        cache["complete"] = True
        self.api.save_cache(case.case_id, cache)
        LOG.info("[retrieve] %s done: %d chunks in %d calls",
                 case.case_id, len(cache["chunks"]), cache["calls"])
        return cache
