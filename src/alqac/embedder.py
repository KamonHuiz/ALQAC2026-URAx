"""Dense (bge-m3) + lexical (BM25) retrieval helpers.

Corpora here are small (≤ ~4.3k articles, 50 precedents) so we keep dense indices as
plain normalised matrices and do exact cosine top-k with numpy — no FAISS server needed.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .utils import LOG, strip_diacritics


class Embedder:
    def __init__(self, cfg):
        self.name = cfg.embedder.name
        self.batch_size = int(cfg.embedder.batch_size)
        self.max_length = int(cfg.embedder.max_length)
        self._backend = None
        self._load()

    def _load(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
            LOG.info("Loading embedder %s (FlagEmbedding) ...", self.name)
            self.model = BGEM3FlagModel(self.name, use_fp16=True)
            self._backend = "flag"
        except Exception as e:
            LOG.warning("FlagEmbedding unavailable (%s); using sentence-transformers", e)
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name)
            self._backend = "st"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        if self._backend == "flag":
            out = self.model.encode(texts, batch_size=self.batch_size,
                                    max_length=self.max_length)["dense_vecs"]
            vecs = np.asarray(out, dtype=np.float32)
        else:
            vecs = self.model.encode(texts, batch_size=self.batch_size,
                                     convert_to_numpy=True, normalize_embeddings=False)
            vecs = np.asarray(vecs, dtype=np.float32)
        # L2-normalise so dot product == cosine
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class DenseIndex:
    """Exact cosine index over a fixed set of items."""
    def __init__(self, embedder: Embedder, texts: list[str], payloads: list):
        self.embedder = embedder
        self.payloads = payloads
        self.matrix = embedder.encode(texts) if texts else np.zeros((0, 1024), np.float32)

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, object]]:
        if self.matrix.shape[0] == 0:
            return []
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        k = min(top_k, len(self.payloads))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(float(scores[i]), self.payloads[i]) for i in idx]


_TOK_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOK_RE.findall(strip_diacritics(text))


class BM25Index:
    def __init__(self, texts: list[str], payloads: list):
        from rank_bm25 import BM25Okapi
        self.payloads = payloads
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts]) if texts else None

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, object]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        k = min(top_k, len(self.payloads))
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.payloads[i]) for i in idx]
