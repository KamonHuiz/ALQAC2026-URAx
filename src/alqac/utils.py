"""Small shared utilities: logging, (fast) JSON IO, text normalisation, checkpoints."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

try:
    import orjson  # fast path for the big corpus files
    _HAS_ORJSON = True
except Exception:  # pragma: no cover
    _HAS_ORJSON = False


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "alqac", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


LOG = get_logger()


# --------------------------------------------------------------------------- #
# JSON IO  (UTF-8, ensure_ascii=False so Vietnamese stays readable on disk)
# --------------------------------------------------------------------------- #
def read_json(path: str | os.PathLike) -> Any:
    path = str(path)
    if _HAS_ORJSON:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | os.PathLike, *, indent: bool = True) -> None:
    """Atomic write (write to .tmp then replace) so a crash never corrupts a checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if _HAS_ORJSON:
        opt = orjson.OPT_INDENT_2 if indent else 0
        with open(tmp, "wb") as f:
            f.write(orjson.dumps(obj, option=opt | orjson.OPT_NON_STR_KEYS))
    else:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2 if indent else None)
    os.replace(tmp, path)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M")


# --------------------------------------------------------------------------- #
# Vietnamese text normalisation
# --------------------------------------------------------------------------- #
def strip_diacritics(text: str) -> str:
    """'Bộ luật Tố tụng Dân sự' -> 'bo luat to tung dan su' (lower, no accents)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    out = "".join(c for c in nfkd if not unicodedata.combining(c))
    return out.lower()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def norm_key(text: str) -> str:
    """Aggressive normalisation for fuzzy law-name matching."""
    return normalize_ws(strip_diacritics(text))


_ARTICLE_RE = re.compile(r"\bĐiều\s+(\d{1,3})", re.IGNORECASE)


def extract_article_numbers(text: str) -> list[int]:
    """All 'Điều N' article numbers appearing in a text, in order (deduplicated)."""
    seen: list[int] = []
    for m in _ARTICLE_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


# codes like 91/2015/QH13, 326/2016/UBTVQH14, 24/2012/NĐ-CP, 39/2016/TT-NHNN
_CODE_RE = re.compile(r"\b\d{1,3}/\d{4}/[A-Za-zĐ\-]+\b")
# partial codes like 91/2015 or 326/2016
_CODE_SHORT_RE = re.compile(r"\b(\d{1,3}/\d{4})\b")


def find_law_codes(text: str) -> list[str]:
    return _CODE_RE.findall(text or "")


def find_short_codes(text: str) -> list[str]:
    return _CODE_SHORT_RE.findall(text or "")


# --------------------------------------------------------------------------- #
# JSON extraction from LLM output (robust to ```json fences and think traces)
# --------------------------------------------------------------------------- #
def strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning traces, keeping the final answer."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def extract_json(text: str) -> Any | None:
    """Best-effort extraction of the first JSON object/array from an LLM response."""
    if not text:
        return None
    text = strip_think(text)
    # strip code fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        obj = _try_load(candidate)
        if obj is not None:
            return obj
    # find first balanced { } or [ ]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    obj = _try_load(text[start:i + 1])
                    if obj is not None:
                        return obj
                    break
    return _try_load(text.strip())


def _try_load(s: str) -> Any | None:
    try:
        return json.loads(s)
    except Exception:
        # tolerate trailing commas
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", s))
        except Exception:
            return None
