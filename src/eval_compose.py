#!/usr/bin/env python3
"""
eval_compose.py
---------------
Scores composed plans (bench_pipeline.py output) against the gold final plans.

  format_ok        every line parsed, every API step resolved to a candidate with JSON arguments
  executable       format_ok and every $stepN reference points at an earlier step
  api_recall       gold API steps whose acceptable endpoint appears anywhere in the composed plan
  api_precision    composed API steps whose endpoint is acceptable for some gold API step
  full_api_plan    samples where every gold API step is recovered
  arg_name_f1      argument-name F1 on API steps matched to a gold step (same acceptable endpoint)
  arg_value_exact  share of gold argument values reproduced exactly on matched steps
  step_count_match composed step count equals the gold count

  python3 src/eval_compose.py --composed data/results/pipeline_mole_composed.jsonl [--output x.json]
"""
import argparse
import json
import re
import statistics as st

from mole_common import load_split

REF = re.compile(r"\$step(\d+)")


def score(gold_rows, outs):
    m = {k: [] for k in ("format_ok", "executable", "api_recall", "api_precision", "full_api_plan",
                         "step_count_match", "arg_name_f1", "arg_value_exact")}
    hits = tot = prec_hit = prec_tot = 0
    for o in outs:
        g = gold_rows[o["sample_id"]]
        steps = o["steps"]
        gapi = [(s, c) for s, c in zip(g["final_plan"], g["canonical_plan"]) if s["primitive_function"] != "LOCAL"]
        accept = [set(c["acceptable_tools"]) for _, c in gapi]
        union = set().union(*accept) if accept else set()
        capi = [s for s in steps if s.get("registry_id") is not None]
        m["format_ok"].append(bool(o["ok"]))
        refs_ok = all(int(n) < s["step_num"] for s in steps for v in (s.get("arguments") or {}).values()
                      for n in REF.findall(json.dumps(v)))
        m["executable"].append(bool(o["ok"]) and refs_ok)
        m["step_count_match"].append(len(steps) == len(g["final_plan"]))
        got = [any(s["registry_id"] in a for s in capi) for a in accept]
        hits += sum(got); tot += len(got)
        prec_hit += sum(s["registry_id"] in union for s in capi); prec_tot += len(capi)
        m["full_api_plan"].append(all(got) if got else True)
        for (gs, _), a in zip(gapi, accept):
            for s in capi:
                if s["registry_id"] in a:
                    gk, ck = set((gs.get("arguments") or {})), set(s.get("arguments") or {})
                    tp = len(gk & ck)
                    m["arg_name_f1"].append(2 * tp / (len(gk) + len(ck)) if gk or ck else 1.0)
                    gv = gs.get("arguments") or {}
                    if gv:
                        m["arg_value_exact"].append(sum(s["arguments"].get(k) == v for k, v in gv.items()) / len(gv))
                    break
    res = {k: round(100 * st.mean(v), 2) for k, v in m.items() if v}
    res["api_recall"] = round(100 * hits / max(tot, 1), 2)
    res["api_precision"] = round(100 * prec_hit / max(prec_tot, 1), 2)
    res["n_samples"], res["n_gold_api_steps"] = len(outs), tot
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--composed", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    gold = {r["sample_id"]: r for r in load_split(args.split)}
    outs = [json.loads(l) for l in open(args.composed)]
    res = score(gold, outs)
    print(json.dumps(res, indent=2))
    if args.output:
        json.dump(res, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
