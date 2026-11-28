#!/usr/bin/env python3
"""
bench_pipeline.py
-----------------
Runs the full RAG-MoLE pipeline (Discovery -> retrieval -> Composition) on a split and measures it.

  --config mole     router + 4 type experts for Discovery, then the Composition adapter
  --config single   ONE Discovery adapter (ablation), then the same Composition adapter
  --oracle          also compose from the gold plan with gold endpoints forced into the candidates
                    (isolates Composition skill from Discovery and retrieval errors)

Writes <out>_plans.jsonl (Discovery plans, scored by eval_mole.py), <out>_composed.jsonl (scored by
eval_compose.py) and <out>_bench.json: throughput run (batched) plus an online run (batch size 1) with
per-stage latency, GPU peak memory and adapter sizes.
"""
import argparse
import json
import os
import statistics as st
import time

import torch

from mole_common import ADAPTER_DIR, load_split
from mole_compose import Composer, Retriever, gold_cands, parse_composition, retrieve_for_plan
from mole_discovery import Discovery, SingleDiscovery


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def dir_mb(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors")) / 2 ** 20


def run(disc, comp, ret, rows, bs):
    """Discovery -> retrieval -> Composition on rows; returns plans, composed outputs and stage seconds."""
    q = [r["query"] for r in rows]
    t0 = sync(); plans = disc.plan(q, bs=bs); t1 = sync()
    cands = [retrieve_for_plan(p, ret) for p in plans]; t2 = sync()
    raw = comp.compose(q, plans, cands, bs=bs); t3 = sync()
    composed = [parse_composition(t, c, ret) for t, c in zip(raw, cands)]
    return plans, composed, raw, {"discovery": t1 - t0, "retrieval": t2 - t1, "composition": t3 - t2, "total": t3 - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=["mole", "single"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, help="output prefix, e.g. data/results/pipeline_mole")
    ap.add_argument("--adapter_dir", default=ADAPTER_DIR)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lat_n", type=int, default=30, help="queries in the batch-size-1 latency run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()

    rows = load_split(args.split)[:args.limit or None]
    disc = (Discovery(args.adapter_dir, four_bit=False) if args.config == "mole"
            else SingleDiscovery(args.adapter_dir, four_bit=False))
    disc.model.load_adapter(os.path.join(args.adapter_dir, "compose"), adapter_name="compose")
    ret = Retriever()
    comp = Composer(disc.model, disc.tok, ret)
    res = {"config": args.config, "split": args.split, "n": len(rows),
           "adapter_mb": {a: round(dir_mb(os.path.join(args.adapter_dir, a)), 1)
                          for a in (["router", "read", "write", "process", "local"] if args.config == "mole"
                                    else ["single"]) + ["compose"]}}
    res["discovery_adapter_mb_total"] = round(sum(v for k, v in res["adapter_mb"].items() if k != "compose"), 1)

    run(disc, comp, ret, rows[:2], 2)  # warm-up (CUDA kernels, adapter swaps)
    torch.cuda.reset_peak_memory_stats()
    plans, composed, raw, t = run(disc, comp, ret, rows, args.bs)
    res["throughput"] = {"batch_size": args.bs, "seconds": {k: round(v, 1) for k, v in t.items()},
                         "queries_per_min": round(60 * len(rows) / t["total"], 2)}
    res["peak_gpu_gb_batched"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    with open(f"{args.out}_plans.jsonl", "w") as f:
        for r, p in zip(rows, plans):
            f.write(json.dumps({"sample_id": r["sample_id"], "query": r["query"], "pred_plan": p}) + "\n")
    with open(f"{args.out}_composed.jsonl", "w") as f:
        for r, (steps, ok), rw in zip(rows, composed, raw):
            f.write(json.dumps({"sample_id": r["sample_id"], "cond": "pipeline", "ok": ok, "steps": steps, "raw": rw}) + "\n")
    print("throughput", res["throughput"], flush=True)

    if args.oracle:
        gc = [gold_cands(r, ret) for r in rows]
        t0 = sync()
        raw = comp.compose([r["query"] for r in rows], [r["canonical_plan"] for r in rows], gc, bs=args.bs)
        res["oracle_composition_seconds"] = round(sync() - t0, 1)
        with open(f"{args.out}_composed_oracle.jsonl", "w") as f:
            for r, c, rw in zip(rows, gc, raw):
                steps, ok = parse_composition(rw, c, ret)
                f.write(json.dumps({"sample_id": r["sample_id"], "cond": "oracle", "ok": ok, "steps": steps, "raw": rw}) + "\n")

    torch.cuda.reset_peak_memory_stats()
    lat = {"discovery": [], "retrieval": [], "composition": [], "total": []}
    for r in rows[:args.lat_n]:
        _, _, _, t = run(disc, comp, ret, [r], 1)
        for k in lat:
            lat[k].append(t[k])
    res["online_bs1"] = {k: {"mean_s": round(st.mean(v), 2), "p50_s": round(pct(v, .5), 2), "p95_s": round(pct(v, .95), 2)}
                         for k, v in lat.items()}
    res["online_bs1"]["n"] = len(lat["total"])
    res["peak_gpu_gb_bs1"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    json.dump(res, open(f"{args.out}_bench.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
