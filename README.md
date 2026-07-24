# URAx-LACE — Legal Agentic Case-outcome Engine (ALQAC 2026)

A multi-module **Agentic-RAG** system for the ALQAC 2026 shared task:
*Legal Case Outcome Prediction with Evidence Retrieval* (Vietnamese civil cases).

Given a short Vietnamese case query it (1) predicts the outcome
(`A_WIN` / `PARTIAL_A_WIN` / `PARTIAL_B_WIN` / `B_WIN`), (2) retrieves supporting
**case-evidence** segments via the organiser API, and (3) retrieves the relevant
**law provisions** `{law_id, aid}` — optimised directly for the official metric:

```
FinalScore = 0.70·OutcomeAccuracy + 0.20·PenalizedCaseRecall + 0.10·LawF1micro
```

---

## TL;DR — run it on Colab

1. Open **`notebooks/ALQAC2026_Colab.ipynb`** in Google Colab (A100/L4, High-RAM).
2. Run the cells: clone → mount Drive + token → install → (optional) offline public
   validation → **private run**.
3. Collect `ALQAC_RESULT/<run_id>/[submission] URAx.json` from your Drive.

Or from any Colab cell:

```bash
!git clone https://github.com/KamonHuiz/ALQAC2026-URAx.git
%cd ALQAC2026-URAx
!bash scripts/run_colab.sh --split private --group URAx
```

The token is read from a **Colab Secret** `ALQAC_TOKEN`, the `ALQAC_TOKEN` env var, or
`ALQAC_RESULT/token.txt` on Drive — **never** committed to the repo.

---

## Method (5 stages)

| Stage | Module | File | Idea |
|------:|--------|------|------|
| 0 | Offline prep | `data.py`, `precedent.py`, `embedder.py` | Merge public+private corpora (dedupe `law_id,aid`), build the exact `Điều N → aid` map, a **precedent memory bank** over the 50 labelled public cases, and dense (bge-m3) + BM25 law indices. |
| 1 | **Economical Agentic Retrieval** | `retrieval_agent.py`, `api_client.py` | Ordered query set (VKS-focus → party descriptions → structural bank → LLM-generated), **saturation early-stop** and a per-case call budget so the API-efficiency factor `E_i` stays 1.0. Everything cached on Drive → re-runs cost **0** API calls. |
| 2 | **Structured Case Understanding** | `case_understanding.py` | Condense noisy segments into a structured record and extract the **Viện kiểm sát (VKS) recommendation + stance**. |
| 3 | **Precedent-Augmented Prediction** | `outcome_predictor.py`, `precedent.py` | Case-based reasoning over similar labelled precedents + **dual-advocate debate** + **self-consistency** voting + **VKS-prior fusion** + class-prior calibration. |
| 4 | **Hybrid Law Retrieval** | `law_retriever.py` | Regex **citation extraction** (`Điều N … của <luật>` → exact `aid`) + hybrid semantic/BM25 retrieval + LLM prune to an F1-optimal set + procedural priors. |
| 5 | Assemble + Validate | `assemble.py`, `evaluate.py` | Build `submission.json`; measure accuracy & law-F1 on the public set. |

### Why these choices (paper contributions)

- **VKS-recommendation-aware prediction.** We verified (by probing the API) that the
  served content includes the prosecutor's *"Đề nghị Hội đồng xét xử…"* segment, which
  strongly predicts Vietnamese civil verdicts. We exploit it as a high-precision prior.
- **Efficiency-constrained retrieval.** The scoring penalises API calls (`c_i`); we
  formulate retrieval to maximise distinct-segment coverage under `c_i ≤ 2·n_i`.
- **Precedent CBR** turns the labelled public set into an analogical memory instead of
  throwing it away after few-shot.
- **Exact article-id resolution** from the corpus's article ordering (`Điều N = content[N-1].aid`),
  verified against gold citations (e.g. Điều 603 BLDS ⇒ "bồi thường thiệt hại do súc vật gây ra").

Every module has an on/off flag in `configs/default.yaml` for ablations.

---

## Repository layout

```
configs/default.yaml        all hyper-parameters + ablation flags
data/                       bundled: public test (labelled), both law corpora, query bank, law-name map
src/alqac/                  the pipeline package (one file per stage)
scripts/run_pipeline.py     CLI entry
scripts/run_colab.sh        one-shot Colab install+run
scripts/push_to_github.sh   push helper (token stays out of git)
notebooks/ALQAC2026_Colab.ipynb
docs/METHOD.md              extended method notes for the paper
```

## Configuration highlights

- `model.name` — any open-weight instruct model **< 10B** (default `Qwen/Qwen3-8B`;
  alternatives: `Qwen/Qwen2.5-7B-Instruct`, `SeaLLMs/SeaLLMs-v3-7B-Chat`, `google/gemma-2-9b-it`).
- `retrieval.saturation_patience`, `max_calls_per_case` — API-budget control.
- `outcome.self_consistency_samples`, `use_debate`, `use_vks_fusion`, `vks_prior_weight`.
- `law.max_articles`, `use_citation_extraction`, `use_procedural_priors`.

## Data & privacy note

Bundled in `data/` are the **labelled public test set** and the **law corpora** (public +
private-extracted, merged) so the repo runs out-of-the-box. The 60 private **case queries**
are **not** bundled. If your GitHub repo is public and the organisers' terms restrict
redistributing provided data, keep this repo **private**, or delete `data/*.json` and place
those files in `ALQAC_RESULT/input/` on Drive instead.

## Providing the private test cases

The 60 private **cases** (with `case_query`, `A_description`, `B_description`, `case_id`)
are downloaded from the organisers' *Private Test Folder* into
`ALQAC_RESULT/input/` on your Drive — the pipeline auto-detects them. (The private **law
corpus** is already bundled and merged with the public corpus.)

## Important notes on API budget

`c_i` (calls per case) accumulates **forever** across all runs and feeds the efficiency
penalty. This system retrieves each case **once**, caches it on Drive, and never re-queries.
Do **not** delete `ALQAC_RESULT/<run_id>/case_cache/` between runs.

## Local development

```bash
pip install -r requirements.txt
# offline logic (no GPU, no API):
python scripts/run_pipeline.py --split public --no-api --drive-root ./_local_out --backend hf
```

See `docs/METHOD.md` for the full write-up.
