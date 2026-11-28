#!/usr/bin/env python3
"""
benchmark_plan_retrieval.py
---------------------------
Query-only vs decomposed-step retrieval on the RWPL plan dataset (v3).

Every API step in plan_dataset_v3 has a gold endpoint set (`acceptable_tools`) in
master_registry.json. A step is found when any acceptable endpoint is retrieved.

Retrieval methods:
  query        - the raw user query, one ranking per sample
  steps        - each gold API step's intent, one ranking per step (oracle decomposition)
  steps_fused  - every intent (LOCAL steps too), one ranking per sample: score = max over intents
  joined       - all intents concatenated into one string, one ranking per sample
  step+query   - "<intent>. Context: <query>", one ranking per step

Metrics:
  per-step   Hit@1/5/10 and MRR of the best-ranked acceptable endpoint
  per-plan   Cov@k: share of a sample's API steps whose endpoint lands in a budget of k * n_api
             results (top k*n_api of a single ranking, or the union of each step's top k);
             Full@k: share of samples where every API step is covered

Caveat: the gold endpoint for each step was chosen from a candidate pool built with BGE-M3 and
TF-IDF on the step intent (ground_plan.py). That favours intent-based retrieval under BGE-M3, so
all-MiniLM-L6-v2 is the primary retriever.
"""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.environ.get("RAG_MOLE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", f"{ROOT}/data/processed/registry_emb_minilm.npy"),
    "bge-m3": ("BAAI/bge-m3", f"{ROOT}/data/processed/registry_emb_bge-m3.npy"),
}
METHODS = ["query", "steps", "steps_fused", "joined", "step+query"]
PER_STEP = {"steps", "step+query"}
KS = (1, 5, 10)


def registry_texts(reg):
    # Same endpoint text as ground_plan.CandidateIndex, so the BGE-M3 cache is reusable.
    return [f"{r['tool_name']} | {r['api_name']}: {r.get('api_description') or ''} "
            f"{(r.get('tool_description') or '')[:200]}" for r in reg]


def load_samples(dataset, splits_dir):
    split_of = {}
    for name in ("train", "val", "test"):
        for line in open(f"{splits_dir}/{name}.jsonl"):
            split_of[json.loads(line)["sample_id"]] = name
    rows = [json.loads(l) for l in open(dataset)]
    for r in rows:
        r["split"] = split_of.get(r["sample_id"], "train")
    return rows


def evaluate(rows, model, emb, idx_of):
    """Returns per-step records and per-sample coverage records for every method."""
    texts = {m: [] for m in METHODS}
    for r in rows:
        plan = r["canonical_plan"]
        api = [s for s in plan if s["requires_api"]]
        texts["query"].append(r["query"])
        texts["joined"].append(". ".join(s["canonical_intent"] for s in plan))
        texts["steps"] += [s["canonical_intent"] for s in api]
        texts["step+query"] += [f"{s['canonical_intent']}. Context: {r['query']}" for s in api]
        texts["steps_fused"] += [s["canonical_intent"] for s in plan]
    vecs = {m: model.encode(t, batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32) for m, t in texts.items()}

    steps, plans = [], []
    cursor = {m: 0 for m in METHODS}
    for i, r in enumerate(rows):
        plan = r["canonical_plan"]
        api = [s for s in plan if s["requires_api"]]
        gold = [[idx_of[t] for t in s["acceptable_tools"] if t in idx_of] for s in api]
        n = len(api)
        for m in METHODS:
            if m in PER_STEP:
                S = emb @ vecs[m][cursor[m]:cursor[m] + n].T          # (N, n)
                cursor[m] += n
                cols = [S[:, j] for j in range(n)]
            else:
                if m == "steps_fused":
                    S = (emb @ vecs[m][cursor[m]:cursor[m] + len(plan)].T).max(axis=1)
                    cursor[m] += len(plan)
                else:
                    S = emb @ vecs[m][i]
                cols = [S] * n
            for j, (s, g) in enumerate(zip(api, gold)):
                if not g:
                    continue
                rank = int((cols[j] > cols[j][g].max()).sum()) + 1
                steps.append({"sample": i, "split": r["split"], "method": m, "rank": rank,
                              "prim": s["primitive_function"]})
            for k in KS:
                if m in PER_STEP:
                    got = set().union(*(np.argpartition(-c, k)[:k] for c in cols))
                else:
                    got = set(np.argpartition(-cols[0], k * n)[:k * n])
                hit = [any(t in got for t in g) for g in gold if g]
                plans.append({"sample": i, "split": r["split"], "method": m, "k": k,
                              "cov": float(np.mean(hit)), "full": all(hit)})
    return steps, plans


def summarise(steps, plans, split=None):
    out = {}
    for m in METHODS:
        rk = np.array([s["rank"] for s in steps if s["method"] == m and split in (None, s["split"])])
        res = {f"Hit@{k}": round(100 * float((rk <= k).mean()), 2) for k in KS}
        res["MRR"] = round(float((1 / rk).mean()), 4)
        res["n_steps"] = int(len(rk))
        for k in KS:
            p = [p for p in plans if p["method"] == m and p["k"] == k and split in (None, p["split"])]
            res[f"Cov@{k}"] = round(100 * float(np.mean([x["cov"] for x in p])), 2)
            res[f"Full@{k}"] = round(100 * float(np.mean([x["full"] for x in p])), 2)
        out[m] = res
    return out


def paired_bootstrap(steps, a, b, metric, n_boot=2000, seed=0):
    """95% CI of (b - a) on a per-step metric, resampling whole samples."""
    f = (lambda r: 1 / r) if metric == "MRR" else (lambda r: float(r <= int(metric.split("@")[1])))
    per = {}
    for s in steps:
        if s["method"] in (a, b):
            per.setdefault(s["sample"], {a: [], b: []})[s["method"]].append(f(s["rank"]))
    ids = list(per)
    da = np.array([sum(per[i][a]) for i in ids])
    db = np.array([sum(per[i][b]) for i in ids])
    cnt = np.array([len(per[i][a]) for i in ids])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        boots.append((db[pick].sum() - da[pick].sum()) / cnt[pick].sum())
    diff = (db.sum() - da.sum()) / cnt.sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    scale = 1 if metric == "MRR" else 100
    return {"diff": round(float(scale * diff), 3),
            "ci95": [round(float(scale * lo), 3), round(float(scale * hi), 3)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=f"{ROOT}/data/processed/master_registry.json")
    ap.add_argument("--dataset", default=f"{ROOT}/data/processed/plan_dataset_v3.jsonl")
    ap.add_argument("--splits_dir", default=f"{ROOT}/data/processed/plan_v3")
    ap.add_argument("--models", default="minilm,bge-m3")
    ap.add_argument("--output", default=f"{ROOT}/data/results/plan_retrieval_summary.json")
    args = ap.parse_args()

    reg = json.load(open(args.registry))
    idx_of = {int(r["registry_id"]): i for i, r in enumerate(reg)}
    rows = load_samples(args.dataset, args.splits_dir)
    print(f"{len(rows)} samples, {len(reg)} registry endpoints")

    summary = {"n_samples": len(rows), "n_registry": len(reg), "methods": METHODS, "results": {}}
    for key in args.models.split(","):
        name, cache = MODELS[key]
        model = SentenceTransformer(name, device="cuda")
        if key == "bge-m3":
            model.half()
            model.max_seq_length = 256
        if os.path.exists(cache):
            emb = np.load(cache).astype(np.float32)
        else:
            emb = model.encode(registry_texts(reg), batch_size=128, normalize_embeddings=True,
                               show_progress_bar=True).astype(np.float16)
            np.save(cache, emb)
            emb = emb.astype(np.float32)
        assert emb.shape[0] == len(reg), f"{cache} is stale: {emb.shape[0]} rows vs {len(reg)}"

        steps, plans = evaluate(rows, model, emb, idx_of)
        res = {split or "all": summarise(steps, plans, split)
               for split in (None, "test")}
        res["by_primitive_Hit@5"] = {
            prim: {m: round(100 * float(np.mean([s["rank"] <= 5 for s in steps
                                                 if s["method"] == m and s["prim"] == prim])), 2)
                   for m in METHODS}
            for prim in ("READ", "WRITE", "PROCESS")}
        res["steps_minus_query"] = {met: paired_bootstrap(steps, "query", "steps", met)
                                    for met in ("Hit@1", "Hit@5", "MRR")}
        summary["results"][key] = res

        for split in ("all", "test"):
            print(f"\n[{key}] {split}  (n_steps={res[split]['query']['n_steps']})")
            print(f"{'method':<12}" + "".join(f"{c:>9}" for c in
                  ("Hit@1", "Hit@5", "Hit@10", "MRR", "Cov@1", "Cov@5", "Cov@10", "Full@5")))
            for m in METHODS:
                v = res[split][m]
                print(f"{m:<12}" + "".join(f"{v[c]:>9}" for c in
                      ("Hit@1", "Hit@5", "Hit@10", "MRR", "Cov@1", "Cov@5", "Cov@10", "Full@5")))
        print(f"\n[{key}] Hit@5 by primitive: {res['by_primitive_Hit@5']}")
        print(f"[{key}] steps - query (paired bootstrap): {res['steps_minus_query']}")
        del model

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
