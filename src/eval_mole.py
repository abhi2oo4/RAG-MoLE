#!/usr/bin/env python3
"""
eval_mole.py
------------
Scores plans written by mole_discovery.py against the gold RWPL plans.

Structure:  step count, API-step count, step-type sequence agreement.
Retrieval:  for each gold API step, the best rank its gold endpoint reaches in any generated API
            step's own search (Hit@k, MRR), next to the raw query and the gold steps scored the
            same way. Cov@5 gives each plan a budget of 5 results per API step it searches with.

  python3 src/eval_mole.py --plans data/results/mole_plans_test.jsonl --split test
"""

import argparse
import json
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer

from benchmark_plan_retrieval import MODELS
from mole_common import ROOT, load_split

KS = (1, 5, 10)


def best_ranks(S, gold):
    """S: (N, m) similarity columns. Best 1-based rank of each gold set across the m columns."""
    if S.shape[1] == 0:
        return [10 ** 9] * len(gold)
    return [int(min((S[:, j] > S[g, j].max()).sum() for j in range(S.shape[1]))) + 1 for g in gold]


def covered(S, gold, k):
    if S.shape[1] == 0:
        return [False] * len(gold)
    got = set().union(*(np.argpartition(-S[:, j], k)[:k] for j in range(S.shape[1])))
    return [any(t in got for t in g) for g in gold]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="minilm", choices=list(MODELS))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    gold_rows = {r["sample_id"]: r for r in load_split(args.split)}
    preds = [json.loads(l) for l in open(args.plans)]
    reg = json.load(open(f"{ROOT}/data/processed/master_registry.json"))
    idx_of = {int(r["registry_id"]): i for i, r in enumerate(reg)}
    emb = np.load(MODELS[args.model][1]).astype(np.float32)
    enc = SentenceTransformer(MODELS[args.model][0], device="cuda")
    if args.model == "bge-m3":
        enc.half()
    E = lambda t: enc.encode(t, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False).astype(np.float32).reshape(len(t), -1)

    st = {"n_steps_gold": [], "n_steps_pred": [], "n_api_gold": [], "n_api_pred": [],
          "label_seq_exact": [], "label_seq_ratio": [], "empty_intent": 0, "n_pred_steps": 0}
    ranks = {"query": [], "mole": [], "mole_all_steps": [], "gold_steps": []}
    cov = {m: [] for m in ranks}
    for p in preds:
        g = gold_rows[p["sample_id"]]
        gplan, pplan = g["canonical_plan"], p["pred_plan"]
        gl = [s["primitive_function"] for s in gplan]
        pl = [s["primitive_function"] for s in pplan]
        st["n_steps_gold"].append(len(gl)); st["n_steps_pred"].append(len(pl))
        st["n_api_gold"].append(sum(l != "LOCAL" for l in gl)); st["n_api_pred"].append(sum(l != "LOCAL" for l in pl))
        st["label_seq_exact"].append(gl == pl)
        st["label_seq_ratio"].append(SequenceMatcher(None, gl, pl).ratio())
        st["empty_intent"] += sum(not s["canonical_intent"] for s in pplan)
        st["n_pred_steps"] += len(pplan)

        gold = [[idx_of[t] for t in s["acceptable_tools"] if t in idx_of] for s in gplan if s["requires_api"]]
        texts = {"query": [g["query"]],
                 "mole": [s["canonical_intent"] for s in pplan if s["requires_api"] and s["canonical_intent"]],
                 "mole_all_steps": [s["canonical_intent"] for s in pplan if s["canonical_intent"]],
                 "gold_steps": [s["canonical_intent"] for s in gplan if s["requires_api"]]}
        for m, t in texts.items():
            S = emb @ E(t).T if t else np.zeros((len(emb), 0), dtype=np.float32)
            ranks[m] += best_ranks(S, gold)
            k = 5 * len(gold) if m == "query" else 5
            cov[m] += covered(S, gold, k)

    n = len(preds)
    res = {"split": args.split, "model": args.model, "n_samples": n,
           "structure": {
               "mean_steps_gold": round(float(np.mean(st["n_steps_gold"])), 2),
               "mean_steps_pred": round(float(np.mean(st["n_steps_pred"])), 2),
               "mean_api_steps_gold": round(float(np.mean(st["n_api_gold"])), 2),
               "mean_api_steps_pred": round(float(np.mean(st["n_api_pred"])), 2),
               "api_count_exact_%": round(100 * float(np.mean(np.array(st["n_api_gold"]) == np.array(st["n_api_pred"]))), 2),
               "label_seq_exact_%": round(100 * float(np.mean(st["label_seq_exact"])), 2),
               "label_seq_similarity": round(float(np.mean(st["label_seq_ratio"])), 3),
               "empty_intents": st["empty_intent"], "pred_steps": st["n_pred_steps"]},
           "retrieval": {}}
    for m, rk in ranks.items():
        rk = np.array(rk)
        res["retrieval"][m] = {**{f"Hit@{k}": round(100 * float((rk <= k).mean()), 2) for k in KS},
                               "MRR": round(float((1 / rk).mean()), 4),
                               "Cov@5": round(100 * float(np.mean(cov[m])), 2), "n_gold_steps": int(len(rk))}

    print(json.dumps(res["structure"], indent=1))
    print(f"\n{'method':<16}" + "".join(f"{c:>8}" for c in ("Hit@1", "Hit@5", "Hit@10", "MRR", "Cov@5")))
    for m, v in res["retrieval"].items():
        print(f"{m:<16}" + "".join(f"{v[c]:>8}" for c in ("Hit@1", "Hit@5", "Hit@10", "MRR", "Cov@5")))
    out = args.output or args.plans.replace(".jsonl", f"_eval_{args.model}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
