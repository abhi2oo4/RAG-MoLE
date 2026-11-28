# RAG-MoLE Discovery: trained experts vs raw query vs gold steps

**Date:** 2026-11-13
**Plans:** `mole_plans_test.jsonl`
**Scores:** `mole_plans_*_eval_{minilm,bge-m3}.json`

## Setup

- **Base model:** Qwen2.5-3B-Instruct.
- **Adapters:** five LoRA adapters (rank 16) in `models/mole/`, trained in bf16 on an AWS g5.xlarge (A10G) from the 1,600 training plans of plan_v3.
- **Experts (2 epochs each):**

  | Expert | Validation loss |
  |---|---|
  | READ | 0.228 |
  | WRITE | 0.182 |
  | PROCESS | 0.181 |
  | LOCAL | 0.744 |

- **Router (1 epoch):** 92.8% next-step-type accuracy on validation.
- **Planning:** at each step the router picks READ, WRITE, PROCESS, LOCAL or END, and that type's expert writes the step (`mole_discovery.py`).
- **Precision mismatch:** the plans were generated with a 4-bit copy of the base model on the laptop, while training was in bf16. Results in bf16 may differ slightly.

## Retrieval

Scored with `eval_mole.py`. For each gold API step, the score is the best rank its endpoint reaches across the plan's own API-step searches.

| test (193 plans, 1008 API steps) | MiniLM Hit@1 | MiniLM Hit@5 | MiniLM MRR | BGE-M3 Hit@5 |
|---|---|---|---|---|
| raw query | 1.1 | 2.2 | 0.021 | 4.8 |
| **MoLE plan** | **23.5** | **51.3** | **0.362** | **49.1** |
| gold steps (oracle) | 26.1 | 55.2 | 0.390 | 53.9 |

The generated plans recover about 93% of the oracle's Hit@5 under MiniLM.

## Plan structure (test)

- Mean steps: 10.07 generated vs 10.14 gold.
- Mean API steps: 5.10 generated vs 5.22 gold. The API-step count is exact for 85% of plans.
- Step-type sequence similarity is 0.90, with an exact full-sequence match in 31% of plans.
- No empty or malformed steps.

## Caveats

1. **In-distribution test.** The test queries come from the same authoring process as the training data, so this measures in-distribution performance. Transfer to differently written queries is untested: a 12-plan ToolBench set was removed on 2026-11-23 because its queries were not phrased like ordinary user requests.
2. **Bias inherited from gold-endpoint selection.** Gold endpoints were chosen from intent-based candidate pools, and the experts learn to write intents in the gold style. Some of the gain over the raw query comes from that, as in `PLAN_RETRIEVAL_TEST.md`.
3. **Single-adapter baseline now exists, and it matches MoLE.** See `ABLATION_RESULTS.md`: one LoRA reaches Hit@5 53.3 against MoLE's 52.0 on the same plans, so this run does not show that splitting by step type helps.
