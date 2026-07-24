"""LLM engine — batched inference over vLLM (default) or transformers (fallback).

Exposes a single `chat_batch` that all pipeline stages use, with optional:
  * `choices`  -> constrained decoding to a fixed label set (guided/regex), for the
                  4-way outcome classification and yes/no gates.
  * `n`        -> multiple samples per prompt (self-consistency, debate).
  * thinking   -> Qwen3 <think> chain-of-thought toggle.
Outputs have <think>...</think> traces stripped by callers via utils.strip_think.
"""
from __future__ import annotations

from typing import Optional

from .utils import LOG, strip_think


class LLMEngine:
    def __init__(self, cfg):
        m = cfg.model
        self.cfg = m
        self.name = m.name
        self.backend = m.backend
        self.enable_thinking = bool(m.enable_thinking)
        self.default_temp = float(m.temperature)
        self.default_top_p = float(m.top_p)
        self.default_max_new = int(m.max_new_tokens)
        self._guided_cls = None
        if self.backend == "vllm":
            self._init_vllm(m)
        else:
            self._init_hf(m)

    # ------------------------------------------------------------------ #
    def _init_vllm(self, m):
        from vllm import LLM, SamplingParams  # noqa
        self._SamplingParams = SamplingParams
        try:  # guided decoding lives in different places across versions
            from vllm.sampling_params import GuidedDecodingParams
            self._guided_cls = GuidedDecodingParams
        except Exception:
            LOG.warning("vLLM GuidedDecodingParams unavailable; using free decoding + parsing")
        LOG.info("Loading vLLM model %s ...", self.name)
        self.llm = LLM(
            model=self.name,
            dtype=m.dtype,
            max_model_len=int(m.max_model_len),
            gpu_memory_utilization=float(m.gpu_memory_utilization),
            tensor_parallel_size=int(m.tensor_parallel_size),
            trust_remote_code=True,
            enforce_eager=False,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _init_hf(self, m):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        LOG.info("Loading HF model %s ...", self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self._torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype=getattr(torch, m.dtype, torch.bfloat16),
            device_map="auto", trust_remote_code=True,
        )
        self.model.eval()

    # ------------------------------------------------------------------ #
    def _render(self, messages: list[dict], thinking: bool) -> str:
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=thinking, **kw)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kw)

    # ------------------------------------------------------------------ #
    def chat_batch(self, batch_messages: list[list[dict]], *,
                   temperature: Optional[float] = None, top_p: Optional[float] = None,
                   max_tokens: Optional[int] = None, n: int = 1,
                   choices: Optional[list[str]] = None,
                   thinking: Optional[bool] = None,
                   strip: bool = True) -> list[list[str]]:
        """Return, for each input message list, a list of `n` completion strings."""
        temperature = self.default_temp if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p
        max_tokens = self.default_max_new if max_tokens is None else max_tokens
        think = self.enable_thinking if thinking is None else thinking
        # constrained decoding is short & deterministic — no thinking, low temp
        if choices is not None:
            think, temperature, max_tokens = False, 0.0, min(max_tokens, 16)

        prompts = [self._render(m, think) for m in batch_messages]
        if self.backend == "vllm":
            outs = self._gen_vllm(prompts, temperature, top_p, max_tokens, n, choices)
        else:
            outs = self._gen_hf(prompts, temperature, top_p, max_tokens, n, choices)
        if strip:
            outs = [[strip_think(t) for t in group] for group in outs]
        return outs

    def chat(self, messages: list[dict], **kw) -> str:
        return self.chat_batch([messages], **kw)[0][0]

    def chat_batch1(self, batch_messages: list[list[dict]], **kw) -> list[str]:
        kw.pop("n", None)
        return [g[0] for g in self.chat_batch(batch_messages, n=1, **kw)]

    # ------------------------------------------------------------------ #
    def _gen_vllm(self, prompts, temperature, top_p, max_tokens, n, choices):
        sp_kw = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
        if temperature <= 0:
            sp_kw["top_p"] = 1.0
        if choices is not None and self._guided_cls is not None:
            sp_kw["guided_decoding"] = self._guided_cls(choice=list(choices))
        sp = self._SamplingParams(**sp_kw)
        results = self.llm.generate(prompts, sp)
        out = []
        for r in results:
            comps = [o.text for o in r.outputs]
            if choices is not None:
                comps = [self._snap_choice(c, choices) for c in comps]
            out.append(comps)
        return out

    def _gen_hf(self, prompts, temperature, top_p, max_tokens, n, choices):
        import torch
        outs: list[list[str]] = []
        bs = 8
        do_sample = temperature > 0
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=int(self.cfg.max_model_len)
                                 ).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_tokens, do_sample=do_sample,
                    temperature=max(temperature, 1e-5), top_p=top_p,
                    num_return_sequences=n,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            gen = gen[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            for j in range(len(chunk)):
                comps = texts[j * n:(j + 1) * n]
                if choices is not None:
                    comps = [self._snap_choice(c, choices) for c in comps]
                outs.append(comps)
        return outs

    @staticmethod
    def _snap_choice(text: str, choices: list[str]) -> str:
        """Map a free-form completion onto the nearest allowed choice.

        Longest choice first so 'PARTIAL_B_WIN' wins over its substring 'B_WIN'.
        """
        t = strip_think(text).strip()
        for c in choices:
            if t == c:
                return c
        up = t.upper()
        for c in sorted(choices, key=len, reverse=True):
            if c.upper() in up:
                return c
        return choices[0]
