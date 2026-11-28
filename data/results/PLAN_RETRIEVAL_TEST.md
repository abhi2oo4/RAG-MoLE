# Query-only vs decomposed-step retrieval (RWPL plan dataset v3)

**Date:** 2026-11-03
**Script:** `src/benchmark_plan_retrieval.py`
**Raw numbers:** `plan_retrieval_summary.json`

## Setup

- **Corpus:** 47,075 endpoints in `master_registry.json`, searched by dense cosine similarity.
- **Data:** 2000 samples, 10,292 API steps.
- **Gold:** each API step's `acceptable_tools`. A step counts as found if any acceptable endpoint is retrieved.
- **Retrievers:** all-MiniLM-L6-v2 is the primary retriever. BGE-M3 is reported as a check.

### Methods

| method | what is embedded |
|---|---|
| query | the raw user query, one ranking per sample |
| steps | each gold API step's intent, one ranking per step |
| steps_fused | all intents including LOCAL, merged into one ranking (max similarity) |
| joined | all intents concatenated into one string |
| step+query | `<intent>. Context: <query>`, one ranking per step |

### Metrics

- **Hit@k:** share of API steps whose gold endpoint is in the method's top k.
- **Cov@k:** share of a plan's API steps found within an equal result budget of k × (number of API steps).
- **Full@5:** share of samples where every API step is found within the Cov@5 budget.

## Results (all 2000 samples)

| method | MiniLM Hit@1 | MiniLM Hit@5 | MiniLM MRR | MiniLM Cov@5 | BGE-M3 Hit@1 | BGE-M3 Hit@5 | BGE-M3 MRR | BGE-M3 Cov@5 |
|---|---|---|---|---|---|---|---|---|
| query | 0.8 | 2.3 | 0.018 | 5.8 | 1.6 | 5.2 | 0.038 | 13.6 |
| **steps** | **25.4** | **52.7** | **0.380** | **53.2** | **21.2** | **51.6** | **0.351** | **51.9** |
| steps_fused | 6.5 | 18.0 | 0.126 | 37.1 | 5.5 | 16.2 | 0.112 | 33.5 |
| joined | 0.9 | 2.7 | 0.023 | 7.8 | 1.8 | 6.2 | 0.045 | 16.3 |
| step+query | 14.2 | 32.9 | 0.237 | 33.5 | 17.0 | 44.1 | 0.293 | 44.7 |

The steps-minus-query difference is far outside noise (paired bootstrap over samples, 95% CI):

| retriever | Hit@5 difference | CI | MRR difference | CI |
|---|---|---|---|---|
| MiniLM | +50.4 pts | [49.4, 51.4] | +0.362 | [0.354, 0.369] |
| BGE-M3 | +46.4 pts | [45.4, 47.4] | +0.313 | [0.306, 0.320] |

## Reading

1. **Decomposed steps beat the raw query by a wide margin.**
   - A multi-API query embeds as its dominant topic, so the other APIs it needs rank in the hundreds or thousands.
   - In one example, the spreadsheet and currency-conversion steps ranked about 9,000th.
   - The gap is widest for WRITE steps (MiniLM Hit@5: query 0.6 vs steps 55.7). Delivery actions are rarely what the query is "about".
2. **Each intent needs its own search.**
   - Concatenating intents (`joined`) is as poor as the raw query.
   - Fusing every intent into one ranking (`steps_fused`) recovers less than half of the gain, partly because LOCAL intents add noise.
   - Labelling which steps need an API and searching each one separately is what makes the gain.
3. **Adding the query as context dilutes the intent.**
   - It hurts on the authored data.
   - Treat it as a tunable option, not a default.

## Caveats

- **Oracle decomposition.** The steps are gold intents, so this is an upper bound for the Discovery pass. A trained decomposer will land below it.
- **Selection bias.** Gold endpoints were chosen from a BGE-M3 and TF-IDF candidate pool built on the intent text. That is why MiniLM is the primary retriever.
  - The exact size of the gap is likely inflated.
- **No real-query test set.** All queries come from the same authoring process; a ToolBench-derived set was removed on 2026-11-23 because its queries were not phrased like ordinary user requests.
- **Numbers predate the removal.** The "all 2000 samples" figures above include the 12 removed ToolBench plans (48 API steps); rerun `benchmark_plan_retrieval.py` for exact figures.
