"""Configuration loading with dot-access and dict-merge overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config:
    """Recursive dot-accessible view over a nested dict.

    >>> cfg = Config.load()
    >>> cfg.model.name
    'Qwen/Qwen3-8B'
    >>> cfg.get("retrieval.wait_seconds")
    5.2
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # -- access -------------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            val = self._data[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}'") from e
        return Config(val) if isinstance(val, dict) else val

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return Config(node) if isinstance(node, dict) else node

    def to_dict(self) -> dict[str, Any]:
        return self._data

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike | None = None,
             overrides: dict[str, Any] | None = None) -> "Config":
        base_path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
        with open(base_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if overrides:
            data = _deep_merge(data, overrides)
        return cls(data)

    def resolve_path(self, rel: str) -> str:
        """Resolve a possibly-relative data path against the repo root."""
        p = Path(rel)
        return str(p if p.is_absolute() else (REPO_ROOT / p))


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_token() -> str:
    """ALQAC API token, resolved securely (never hard-coded in the repo).

    Order: env ALQAC_TOKEN -> Colab secret 'ALQAC_TOKEN' -> <drive>/ALQAC_RESULT/token.txt
    """
    tok = os.environ.get("ALQAC_TOKEN", "").strip()
    if tok:
        return tok
    try:  # Colab Secrets (Settings -> Secrets -> ALQAC_TOKEN)
        from google.colab import userdata  # type: ignore
        tok = (userdata.get("ALQAC_TOKEN") or "").strip()
        if tok:
            return tok
    except Exception:
        pass
    for cand in ("/content/drive/MyDrive/ALQAC_RESULT/token.txt", "token.txt"):
        p = Path(cand)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "ALQAC_TOKEN not found. Set it via `os.environ['ALQAC_TOKEN']=...`, "
        "a Colab Secret named ALQAC_TOKEN, or a token.txt in your Drive ALQAC_RESULT folder."
    )
