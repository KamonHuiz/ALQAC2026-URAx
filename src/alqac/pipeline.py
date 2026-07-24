"""Orchestrator — wires the 5 stages, resumable with per-phase checkpoints on Drive.

Phase layout (chosen so the slow, sequential API phase is isolated and fully resumable,
while every GPU stage is batched across all cases for throughput):

  0. setup: load model + embedder, build precedent bank & law indices
  1. query-gen        (GPU, batched)   -> queries.json
  2. evidence retrieval (API, serial)  -> case_cache/<id>.json  [resumable, never re-spends]
  3. case understanding (GPU, batched) -> understanding.json
  4. outcome prediction (GPU, batched) -> predictions.json
  5. law retrieval      (GPU, batched) -> law_evidence.json
  6. assemble + validate               -> submission*.json + report.json
"""
from __future__ import annotations

import json
from pathlib import Path

from . import prompts
from .assemble import build_submission
from .api_client import RetrievalAPI
from .case_understanding import CaseUnderstander, evidence_from_chunks
from .config import Config, get_token
from .data import DataManager
from .embedder import BM25Index, DenseIndex, Embedder
from .evaluate import report
from .law_retriever import LawRetriever
from .llm import LLMEngine
from .outcome_predictor import OutcomePredictor
from .precedent import PrecedentBank
from .retrieval_agent import RetrievalAgent
from .utils import LOG, ensure_dir, extract_json, read_json, timestamp, write_json


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Stable per-split run_id by default so ALL checkpoints (and thus a 2-3h retrieval)
        # resume across sessions instead of starting a fresh timestamped dir each time.
        run_id = cfg.get("run.run_id") or f"run_{cfg.run.target_split}"
        self.run_dir = ensure_dir(Path(cfg.run.drive_root) / run_id)
        LOG.info("Run dir: %s", self.run_dir)
        write_json(cfg.to_dict(), self.run_dir / "config_used.json")
        self.data = DataManager(cfg)
        self.cases = self.data.target_cases
        self.use_api = bool(cfg.retrieval.use_api)
        self.chunk_size = int(cfg.get("exec.chunk_size", 12))
        # components filled by setup_models()
        self.llm = self.embedder = None

    # ------------------------------------------------------------------ #
    def _ckpt(self, name: str):
        return self.run_dir / name

    def _chunks(self, seq):
        for i in range(0, len(seq), self.chunk_size):
            yield seq[i:i + self.chunk_size]

    def _load_ckpt(self, name: str, force: bool):
        p = self._ckpt(name)
        if p.exists() and not force:
            LOG.info("[resume] loading %s", name)
            return read_json(p)
        return None

    # ------------------------------------------------------------------ #
    def setup_models(self):
        self.llm = LLMEngine(self.cfg)
        self.embedder = Embedder(self.cfg)
        self.precedents = PrecedentBank(self.cfg, self.embedder, self.data.public_cases)
        # law indices over the candidate corpus (truncate text for speed).
        # Embeddings are cached on Drive; restricting the pool changes the signature so a
        # fresh .npy is built automatically for the new corpus.
        arts = self.data.corpus.articles
        if self.cfg.get("law.candidate_corpus", "split") == "split":
            ids = self.data.retrieval_law_ids
            arts = [a for a in arts if a["law_id"] in ids]
            LOG.info("Law candidate pool = split corpus: %d laws, %d articles (was %d)",
                     len(ids), len(arts), len(self.data.corpus.articles))
        texts = [a["text"][:1000] for a in arts]
        from hashlib import md5
        sig = md5(("|".join([self.cfg.embedder.name]
                            + [f"{a['law_id']}:{a['aid']}" for a in arts])).encode()).hexdigest()[:10]
        cache_path = str(Path(self.cfg.run.drive_root) / "_emb_cache" / f"corpus_{sig}.npy")
        LOG.info("Building law index over %d corpus articles (cache: %s)", len(texts), cache_path)
        self.dense = DenseIndex(self.embedder, texts, arts, cache_path=cache_path)
        self.bm25 = BM25Index(texts, arts)
        self.understander = CaseUnderstander(self.cfg, self.llm)
        self.predictor = OutcomePredictor(self.cfg, self.llm, self.precedents)
        self.law = LawRetriever(self.cfg, self.llm, self.data.corpus, self.dense, self.bm25)

    # ------------------------------------------------------------------ #
    # Phase 1 — retrieval query generation
    # ------------------------------------------------------------------ #
    def gen_queries(self, force=False) -> dict:
        cached = self._load_ckpt("queries.json", force)
        if cached is not None:
            return cached
        out = {}
        if self.use_api and self.cfg.retrieval.use_generated_queries:
            n = int(self.cfg.retrieval.num_generated_queries)
            batch = [prompts.query_generation(c.case_query, c.A_description, c.B_description, n)
                     for c in self.cases]
            raws = self.llm.chat_batch1(batch, thinking=False, max_tokens=512)
            for c, raw in zip(self.cases, raws):
                obj = extract_json(raw) or {}
                qs = obj.get("queries", []) if isinstance(obj, dict) else []
                out[c.case_id] = [q for q in qs if isinstance(q, str)][:n]
        write_json(out, self._ckpt("queries.json"))
        return out

    # ------------------------------------------------------------------ #
    # Phase 2 — evidence retrieval (API, serial, resumable)
    # ------------------------------------------------------------------ #
    def retrieve(self, queries: dict) -> dict:
        caches = {}
        if not self.use_api:
            LOG.info("[retrieve] API disabled — using offline evidence (case_fact)")
            for c in self.cases:
                caches[c.case_id] = {"chunks": {}, "calls": 0, "complete": True,
                                     "offline_text": c.case_fact or c.case_query}
            return caches
        token = get_token()
        # IMPORTANT: the API cache lives in a STABLE, split-scoped location shared by every
        # run/experiment — never under a timestamped dir — so we never re-spend calls (c_i).
        cache_dir = ensure_dir(Path(self.cfg.run.drive_root) / "case_cache"
                               / self.cfg.run.target_split)
        api = RetrievalAPI(self.cfg, token, str(cache_dir))
        agent = RetrievalAgent(self.cfg, api, self.data)
        for i, c in enumerate(self.cases, 1):
            LOG.info("[retrieve] (%d/%d) %s", i, len(self.cases), c.case_id)
            caches[c.case_id] = agent.retrieve_case(c, queries.get(c.case_id, []))
        LOG.info("[retrieve] total API calls this run: %d", api.total_calls)
        return caches

    # ------------------------------------------------------------------ #
    # Phase 3 — understanding  (chunked + checkpointed + resumable)
    # ------------------------------------------------------------------ #
    def understand(self, caches: dict, force=False) -> list[dict]:
        ckpt = self._ckpt("understanding.json")
        done: dict[str, dict] = {}
        if not force and ckpt.exists():
            prev = read_json(ckpt)
            done = {r["case_id"]: r for r in prev if isinstance(r, dict) and "case_id" in r}
        max_chars = int(self.cfg.understanding.max_context_chars)
        todo = [c for c in self.cases if c.case_id not in done]
        if todo:
            LOG.info("[understand] %d/%d cases to process", len(todo), len(self.cases))
            for chunk in self._chunks(todo):
                evid = []
                for c in chunk:
                    cache = caches.get(c.case_id, {})
                    if cache.get("chunks"):
                        evid.append(evidence_from_chunks(cache["chunks"], max_chars))
                    else:
                        evid.append((cache.get("offline_text") or c.case_query)[:max_chars])
                for rec in self.understander.run(chunk, evid):
                    done[rec["case_id"]] = rec
                write_json([done[c.case_id] for c in self.cases if c.case_id in done], ckpt)
                LOG.info("[understand] %d/%d done", len(done), len(self.cases))
        return [done[c.case_id] for c in self.cases]

    # ------------------------------------------------------------------ #
    # Phase 4 — outcome  (chunked + checkpointed + resumable)
    # ------------------------------------------------------------------ #
    def predict(self, recs: list[dict], force=False) -> dict:
        ckpt = self._ckpt("predictions.json")
        done: dict = {} if force or not ckpt.exists() else read_json(ckpt)
        rec_by_id = {r["case_id"]: r for r in recs}
        summ_by_id = {r["case_id"]: self.understander.summary_text(r) for r in recs}
        exclude_self = (self.cfg.run.target_split == "public")
        todo = [c for c in self.cases if c.case_id not in done]
        if todo:
            LOG.info("[predict] %d/%d cases to process", len(todo), len(self.cases))
            for chunk in self._chunks(todo):
                summaries = [summ_by_id[c.case_id] for c in chunk]
                crecs = [rec_by_id[c.case_id] for c in chunk]
                for p in self.predictor.predict(chunk, summaries, crecs, exclude_self=exclude_self):
                    done[p["case_id"]] = p
                write_json(done, ckpt)
                LOG.info("[predict] %d/%d done", len(done), len(self.cases))
        return done

    # ------------------------------------------------------------------ #
    # Phase 5 — law retrieval  (chunked + checkpointed + resumable)
    # ------------------------------------------------------------------ #
    def retrieve_law(self, recs: list[dict], caches: dict, force=False) -> dict:
        ckpt = self._ckpt("law_evidence.json")
        done: dict = {} if force or not ckpt.exists() else read_json(ckpt)
        rec_by_id = {r["case_id"]: r for r in recs}
        todo = [c for c in self.cases if c.case_id not in done]
        if todo:
            LOG.info("[law] %d/%d cases to process", len(todo), len(self.cases))
            for chunk in self._chunks(todo):
                crecs = [rec_by_id[c.case_id] for c in chunk]
                chunks_list = [caches.get(c.case_id, {}).get("chunks", {}) for c in chunk]
                for c, le in zip(chunk, self.law.retrieve(chunk, crecs, chunks_list)):
                    done[c.case_id] = le
                write_json(done, ckpt)
                LOG.info("[law] %d/%d done", len(done), len(self.cases))
        return done

    # ------------------------------------------------------------------ #
    def run(self, force_stages: set | None = None) -> dict:
        force = force_stages or set()
        self.setup_models()

        queries = self.gen_queries("queries" in force)
        caches = self.retrieve(queries)
        recs = self.understand(caches, "understand" in force)
        preds = self.predict(recs, "predict" in force)
        pred_labels = {cid: p["prediction"] for cid, p in preds.items()}
        law_ev = self.retrieve_law(recs, caches, "law" in force)

        submission = build_submission(
            self.cases, pred_labels, caches, law_ev,
            submit_all_chunks=bool(self.cfg.retrieval.submit_all_cached_chunks))
        # organiser naming: "[submission] YOUR_GROUP_NAME"
        group = self.cfg.run.group_name
        write_json(submission, self.run_dir / "submission.json")
        write_json(submission, self.run_dir / f"[submission] {group}.json")
        LOG.info("Wrote submission (%d cases) to %s", len(submission), self.run_dir)

        rep = report(self.cases, pred_labels, law_ev, caches, self.data.corpus)
        write_json(rep, self.run_dir / "report.json")
        return {"submission": submission, "report": rep, "run_dir": str(self.run_dir)}
