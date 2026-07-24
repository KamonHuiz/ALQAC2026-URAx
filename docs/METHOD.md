# URAx-LACE — Method notes (for the paper)

This document expands the method behind the system in `src/alqac/`, framed for a workshop
paper. It is deliberately written so results can be **modest but the ideas novel and
reproducible**; every component is toggle-able for ablations (`configs/default.yaml`).

## 1. Problem & metric

ALQAC 2026 scores a submission as

```
FinalScore = 0.70·OutcomeAccuracy + 0.20·PenalizedCaseRecall + 0.10·LawF1micro
PenalizedCaseRecall = CaseRecall · E_i ,   E_i = max(0, 1 − max(0, c_i − 2n_i)/(3n_i))
```

`c_i` = total Case-Content-API calls ever made for case *i* (append-only, never reset);
`n_i` = number of segments in case *i*. The system is designed **around** this metric:
accuracy dominates (70%), and the retrieval component is explicitly efficiency-constrained.

## 2. Data observations that shape the design

- The **public test is fully labelled** (verdict, court reasoning, cited provisions) → we
  reuse it as a *precedent memory bank*, not just few-shot filler.
- Corpus articles are stored **in article order with contiguous aids**, so
  `Điều N ≡ content[N-1]`, giving an exact `article_number → aid` map (verified: Điều 603
  BLDS ⇒ "bồi thường thiệt hại do súc vật gây ra", matching a dog-bite gold case).
- The **public law corpus is reused for the private test**; we merge both and dedupe by
  `(law_id, aid)` (25 laws, ~4.2k articles).
- API-probing the served content shows it contains the **Viện kiểm sát (VKS) recommendation**
  (*"Đề nghị Hội đồng xét xử: 1. Chấp nhận một phần yêu cầu…"*) but **not** the court's
  decision/reasoning — consistent with an outcome-*prediction* task. The VKS stance is a
  strong, near-leak-free predictor we exploit.
- Class prior (public): PARTIAL_A_WIN .38, A_WIN .32, B_WIN .20, PARTIAL_B_WIN .10 — the
  plaintiff prevails (fully or partially) ~70% of the time.

## 3. Pipeline

### Stage 1 — Efficiency-constrained agentic retrieval
Queries are issued in a coverage-oriented order (VKS-focus → party descriptions →
structural bank mirroring a Vietnamese judgment's sections → LLM-generated case-specific
queries) with **saturation early-stop** (stop after `patience` consecutive no-new-chunk
calls) and a hard budget, keeping `c_i` well under `2n_i` so `E_i = 1`. All results are
cached per case on Drive; a query is **never re-issued**, so repeated runs cost zero calls.
Because case-evidence has *no precision penalty*, we submit **all** cached chunk ids.

### Stage 2 — Structured case understanding
An LLM compresses the retrieved segments into a structured record and extracts the VKS
recommendation text + a normalised stance ∈ {ACCEPT_FULL, ACCEPT_PARTIAL, REJECT, UNKNOWN}.

### Stage 3 — Precedent-augmented outcome prediction
1. **Case-based reasoning:** retrieve *k* nearest labelled public precedents (bge-m3 over
   `case_query`) and surface their gold outcome + reasoning snippet.
2. **Dual-advocate debate:** one LLM argues for the plaintiff, one for the defendant.
3. **Adjudicator + self-consistency:** an adjudicator LLM decides; sampled *N* times and
   majority-voted.
4. **VKS-prior fusion:** the VKS stance is mapped to a label and injected as
   `round(w·N)` weighted votes (`ACCEPT_FULL→A_WIN`, `ACCEPT_PARTIAL→PARTIAL_A_WIN`,
   `REJECT→B_WIN`).
5. **Calibration:** ties broken by the public class prior.

### Stage 4 — Hybrid law retrieval
Union of (a) regex **citation extraction** (`Điều N … của <luật>` → exact `aid`),
(b) hybrid **bge-m3 + BM25** retrieval seeded by the extracted legal issues, and
(c) **procedural priors** (frequently-cited BLTTDS + án-phí articles). An LLM prunes the
pool to an F1-optimal set capped at `max_articles` (precision guard).

## 4. Ablations to report

| Toggle (config) | Tests |
|---|---|
| `outcome.use_vks_fusion` | value of the VKS signal |
| `outcome.use_debate` | debate vs. direct prediction |
| `outcome.self_consistency_samples` | 1 vs N |
| `outcome.use_precedents` | precedent CBR contribution |
| `law.use_citation_extraction` / `use_semantic_retrieval` / `llm_rerank` | law-retrieval sources |
| `retrieval.saturation_patience` / `max_calls_per_case` | recall vs. `E_i` trade-off |
| `model.name` | Qwen3-8B vs SeaLLMs-v3-7B vs Gemma-2-9B (all < 10B) |

## 5. Reproducibility
Single config file, fixed seed, per-stage checkpoints on Drive, deterministic
article→aid mapping, and an offline public-set evaluation harness
(`--split public --no-api`) that reports accuracy and law micro-F1.
