# RAG-MoLE ablation: MoLE Discovery vs a single-LoRA Discovery, each followed by one Composition LoRA

**Date:** 2026-11-23
**Hardware:** one AWS g5.xlarge (A10G, 24 GB), bf16, base model Qwen2.5-3B-Instruct, rank-16 LoRA on all projections.
**Data:** plan_v3 test split, 193 plans / 1,008 API steps (authored queries only, so in-distribution).
**Raw files:** `pipeline_{mole,single}_{bench,plans,composed,discovery_minilm,compose_eval}.*`, `pipeline_oracle_compose_eval.json`.

## Systems compared

| | Discovery | Composition |
|---|---|---|
| **MoLE** | router + 4 type experts (READ, WRITE, PROCESS, LOCAL): 5 adapters | one Composition LoRA |
| **Single** | ONE LoRA writes `[TYPE] step` or END: 1 adapter (`models/mole/single`) | the same Composition LoRA |

Composition (`models/mole/compose`): input is the query, the Discovery plan, and the top-5 MiniLM candidates for each API step.
It picks a candidate per API step and writes the arguments. Training forces the gold endpoint into the candidates when
retrieval misses it; at inference the candidates are exactly what retrieval returned.

## Training

| adapter(s) | examples | epochs | minutes (A10G) | final val loss |
|---|---|---|---|---|
| MoLE experts + router | 16.3k + 17.9k | 2 + 1 | 156 | experts 0.18 to 0.74, router acc 92.8% |
| **Single Discovery** | 17.9k | 2 | 120 | 0.393 (includes the type token, so not comparable to expert losses) |
| **Composition** | 1,600 plans | 2 | 72 | 0.088 |

## Discovery quality (MiniLM retrieval from each plan's own API-step intents)

| | raw query | MoLE | Single | gold steps (oracle) |
|---|---|---|---|---|
| Hit@1 | 1.09 | 23.91 | **25.30** | 26.09 |
| Hit@5 | 2.18 | 51.98 | **53.27** | 55.16 |
| Hit@10 | 3.08 | 62.00 | **62.70** | 64.88 |
| MRR | 0.021 | 0.365 | **0.381** | 0.390 |
| API-step count exact | | 86.0% | **89.6%** | |
| step-type sequence exact | | 28.5% | **40.9%** | |
| step-type sequence similarity | | 0.896 | **0.920** | |

## Composition quality (end to end, and with oracle inputs)

| | MoLE pipeline | Single pipeline | oracle plan + gold-forced candidates |
|---|---|---|---|
| well-formed and executable | 99.48% | 99.48% | 100% |
| gold API steps recovered (api_recall) | 42.06% | 41.87% | 95.83% |
| API precision | 43.10% | 42.08% | 95.83% |
| every gold API step recovered | 3.11% | 3.11% | 81.35% |
| argument-name F1 (matched steps) | 80.02% | 80.40% | 98.35% |
| argument-value exact (matched steps) | 32.44% | 33.42% | 59.75% |
| step count matches gold | 87.05% | 86.53% | 100% |

With perfect inputs Composition recovers 95.8% of API steps and writes well-formed plans every time.
End to end it is capped by retrieval: MiniLM puts only about 52% of the gold endpoints in the top 5.

## Cost (A10G, bf16)

| | MoLE | Single |
|---|---|---|
| Discovery adapter size on disk | 571.5 MB | **114.3 MB** |
| Composition adapter size | 114.3 MB | 114.3 MB |
| Online latency, batch 1, mean (n=30) | **34.4 s** (p50 33.6, p95 42.5) | 36.4 s (p50 35.7, p95 43.9) |
|   of which Discovery | **12.8 s** (p95 14.2) | 14.4 s (p95 18.2) |
|   of which Composition | 21.6 s | 22.0 s |
|   of which retrieval | 0.01 s | 0.01 s |
| Throughput, batch 16, 193 queries | 798 s (14.5 q/min) | **742 s (15.6 q/min)** |
|   Discovery only | 391 s | **329 s** |
| Peak GPU memory, batch 1 / batch 16 | 6.8 / 11.2 GB | 6.4 / 11.1 GB |

## Reading

1. **The ablation does not support splitting Discovery into per-type experts.** On this test set the single LoRA is
   as good or slightly better on every Discovery metric and matches MoLE on Composition. The gaps are small
   (Hit@5 +1.3 points, 13 of 1,008 steps) and I did not run a significance test, so read them as "no
   measurable benefit from MoLE", not "single is better".
2. **MoLE's one edge is online latency.** About 2 s lower per query at batch size 1 (n=30, p95s overlap), and the
   single adapter is faster at batch size 16. I did not profile why; a likely cause is that the single adapter also
   generates the type token for every step while MoLE decides it in one router forward pass, and that MoLE's
   per-expert grouping issues more, smaller generations when batched.
3. **MoLE costs 5x the Discovery adapter storage and about 36 more minutes of training.**
4. **Composition dominates latency** (60% of the time at batch 1) and is the same in both systems.
5. **The end-to-end ceiling is retrieval,** not Discovery or Composition: oracle Composition gets 95.8% of API steps
   when the right endpoint is in the candidate list, and the pipeline gets 42% because it often is not.

## Caveats

- One seed per system, no confidence intervals. Differences under about 2 points should not be over-read.
- All queries come from the same authoring process as the training data. There is no real-query test set: the
  12-plan ToolBench set was removed on 2026-11-23 because its queries were not phrased like ordinary requests.
- Gold endpoints were chosen from intent-based candidate pools, so retrieval numbers favour intent-style plans (see
  `PLAN_RETRIEVAL_TEST.md`). The raw-query row is therefore a floor that is probably too low.
- The MoLE experts and router were trained in an earlier run (also a g5.xlarge, bf16, same recipe); the single adapter
  and Composition were trained in this run. MoLE Discovery plans were regenerated in bf16 for this comparison.
- Latency was measured on 30 queries at batch size 1 on one GPU, with greedy decoding and no serving optimisations.
- The single adapter's validation loss rose slightly in epoch 2 (0.3917 to 0.3934), so one epoch may have been enough.

## Base-model baselines (no LoRA, few-shot prompting only)

Run on 2026-11-23 on the same 193 test plans, bf16 on an A10G (`base_baselines.py`, results in `base_*`).

**One pass:** the base Qwen2.5-3B-Instruct writes the whole final plan from the query in one generation and must name
the tool and API itself (one worked example in the prompt, no retrieval).

| | one-pass base |
|---|---|
| names the exact gold tool + API | 2.08% |
| names the right tool | 4.17% |
| API names that do not exist in the registry | 91.0% |
| tool names that do not exist | 74.2% |
| Hit@5 searching the registry with its API-step text | 13.5% (21.6% using all its steps) |
| plan size | 7.1 steps vs 10.1 gold; 4.3 API steps vs 5.2 gold |

**Two pass by prompting:** prompted Discovery (two worked examples) -> the same MiniLM retrieval -> prompted Composition
over the same top-5 candidates (one worked example).

| | base two-pass | MoLE pipeline | single pipeline |
|---|---|---|---|
| Discovery Hit@5 | 25.0 | 52.0 | 53.3 |
| Discovery Hit@1 / MRR | 10.5 / 0.177 | 23.9 / 0.365 | 25.3 / 0.381 |
| plans well-formed and executable | 13.5% | 99.5% | 99.5% |
| gold API steps recovered | 10.6% | 42.1% | 41.9% |
| API precision | 11.7% | 43.1% | 42.1% |
| argument-name F1 (matched steps) | 41.7% | 80.0% | 80.4% |
| argument values exact (matched steps) | 0.0% | 32.4% | 33.4% |
| step count matches gold | 3.6% | 87.1% | 86.5% |
| wall time, 193 queries, batch 16 (Discovery + Composition) | 847 s | 798 s | 742 s |
| peak GPU memory | 9.9 GB | 11.2 GB | 11.1 GB |

Because the base Discovery puts fewer gold endpoints into the candidate lists (25% against 52%), raw recall understates its
Composition skill. Dividing gold API steps recovered by Hit@5 gives the share of reachable endpoints that Composition selected:
about 43% for the base model against 81% (MoLE) and 79% (single). This ratio is derived, not measured separately.

Why the base model scores so low on well-formedness (strict check; any bad line fails the plan): it puts API letters on LOCAL
steps, picks letters outside the five candidates, repeats `api ... | args` fields on a line, and writes JSON objects with
duplicate keys. These are genuine failures of the base model, not scoring artefacts.

Note: `base_onepass_base.json` holds the first parser's name counts (947 API names, 90.4% invented), which skipped names on lines
without `args`. The 91.0% of 1,006 above comes from a more tolerant re-parse of `base_onepass_raw.jsonl`.

Caveats: a single few-shot prompt per stage, not tuned; a better prompt might narrow the gap, though no prompt can give
the base model the registry's API names (see the one-pass result).
