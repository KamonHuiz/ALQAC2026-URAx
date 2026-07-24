"""Rate-limited, retrying client for the ALQAC Case Content Retrieval API.

Design principles (driven by the scoring rules):
  * The 1-request-per-5-seconds limit is PER TEAM, so pacing is GLOBAL across all cases.
  * Every call is expensive: c_i (calls per case) accumulates forever across runs and
    feeds the API-efficiency penalty E_i. So the retrieval agent caches every result on
    Drive and never re-issues a query already spent (see retrieval_agent.py).
  * We keep an append-only call log on Drive mirroring the organiser's server logs, so you
    can audit total spend per case at any time.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import requests

from .utils import LOG, ensure_dir, read_json, write_json


class RetrievalAPI:
    def __init__(self, cfg, token: str, run_dir: str):
        r = cfg.retrieval
        self.url = r.api_url
        self.wait = float(r.wait_seconds)
        self.max_retries = int(r.max_retries)
        self.timeout = int(r.timeout)
        self.headers = {"X-API-Key": token, "Content-Type": "application/json"}

        self.cache_dir = ensure_dir(Path(run_dir) / "case_cache")
        self.log_path = Path(run_dir) / "api_call_log.jsonl"
        self._last_call = 0.0
        self.total_calls = 0

    # ------------------------------------------------------------------ #
    # pacing
    # ------------------------------------------------------------------ #
    def _pace(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.wait:
            time.sleep(self.wait - elapsed)

    # ------------------------------------------------------------------ #
    # single call with retries
    # ------------------------------------------------------------------ #
    def call(self, query: str, case_id: str) -> Optional[dict]:
        """Return the top-1 hit {chunk_id, text, score} or None. Paced + retried."""
        payload = {"query": query, "case_id": case_id}
        for attempt in range(1, self.max_retries + 1):
            self._pace()
            try:
                resp = requests.post(self.url, headers=self.headers,
                                     json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                LOG.warning("API network error (%s), retry %d/%d", e, attempt, self.max_retries)
                self._last_call = time.time()
                time.sleep(self.wait)
                continue
            self._last_call = time.time()

            if resp.status_code == 200:
                self.total_calls += 1
                hit = self._parse(resp)
                self._log(case_id, query, resp.status_code, hit)
                return hit
            if resp.status_code == 429:                # rate limited -> wait extra
                LOG.warning("429 rate-limited on %s, backing off", case_id)
                time.sleep(self.wait * 1.5)
                continue
            if resp.status_code == 503:                # team db temporarily down
                LOG.warning("503 on %s, retrying", case_id)
                time.sleep(self.wait * 2)
                continue
            if resp.status_code == 403:
                raise RuntimeError("403 Forbidden — invalid ALQAC_TOKEN.")
            if resp.status_code == 422:
                LOG.error("422 malformed request for %s query=%r", case_id, query[:60])
                self._log(case_id, query, 422, None)
                return None
            LOG.warning("Unexpected status %d on %s", resp.status_code, case_id)
            time.sleep(self.wait)
        LOG.error("Giving up on query for %s after %d retries", case_id, self.max_retries)
        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse(resp: requests.Response) -> Optional[dict]:
        try:
            results = resp.json().get("results", [])
        except Exception:
            return None
        if not results:
            return None
        top = results[0]
        return {"chunk_id": top["chunk_id"],
                "text": top.get("text", ""),
                "score": float(top.get("score", 0.0))}

    def _log(self, case_id: str, query: str, status: int, hit: Optional[dict]) -> None:
        rec = {"t": time.time(), "case_id": case_id, "query": query[:120],
               "status": status, "chunk_id": (hit or {}).get("chunk_id")}
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # per-case cache (resumable, never re-spend a query)
    # ------------------------------------------------------------------ #
    def cache_path(self, case_id: str) -> Path:
        return self.cache_dir / f"{case_id}.json"

    def load_cache(self, case_id: str) -> dict:
        p = self.cache_path(case_id)
        if p.exists():
            try:
                return read_json(p)
            except Exception:
                pass
        return {"case_id": case_id, "chunks": {}, "queries_done": [], "calls": 0}

    def save_cache(self, case_id: str, cache: dict) -> None:
        write_json(cache, self.cache_path(case_id))
