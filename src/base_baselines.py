#!/usr/bin/env python3
"""
base_baselines.py
-----------------
Baselines that use the BASE model only (no LoRA), driven purely by few-shot prompting.

  --mode onepass   query -> final plan in ONE generation. The model must name tool/API itself (no retrieval,
                   no candidates). Scores whether it names the gold endpoints, how often it invents APIs
                   that do not exist in the registry, and whether searching its step text recovers them.
  --mode twopass   prompted Discovery -> the same MiniLM retrieval -> prompted Composition over the same
                   candidate lists. Outputs are scored by eval_mole.py / eval_compose.py like the trained
                   pipelines.

Writes <out>_plans.jsonl (pred_plan format, for eval_mole.py), <out>_composed.jsonl (twopass only, for
eval_compose.py) and <out>_base.json (timings, plus name-level scores for onepass).

  python3 src/base_baselines.py --mode twopass --out data/results/base_twopass
"""
import argparse
import json
import re
import time

import torch
from transformers import AutoTokenizer

from mole_common import BASE_MODEL, MAX_STEPS, SYSTEM, load_base_model, load_split, parse_step, step_line
from mole_compose import (COMPOSE_SYSTEM, Retriever, compose_messages, gold_cands, parse_composition,
                          retrieve_for_plan, target_text)

LINE = re.compile(r"^\s*(\d+)\.\s*\[(READ|WRITE|PROCESS|LOCAL)\]\s*(.*)$")
ONEPASS_SYSTEM = ("You are an API task planner. Given a user query, write the final executable plan, one line per "
                  "step: `<n>. [TYPE] <operation> | out <name>` for a LOCAL step (no API call), and "
                  "`<n>. [TYPE] <operation> | out <name> | api <Tool name> / <API name> | args {json}` for an API step "
                  "(READ retrieves data, WRITE changes an external system, PROCESS computes or transforms input via an "
                  "API). Use $stepN.<output> to pass an earlier step's output. Output only the plan lines.")
DISC_SYSTEM = SYSTEM + (" Write the whole plan, one line per step: `<n>. [TYPE] <intent> | params <json> | deps <json "
                        "list>`. Output only the plan lines.")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def final_lines(final_plan):
    out = []
    for s in final_plan:
        line = f"{s['step_num']}. [{s['primitive_function']}] {s['operation']} | out {s['output']}"
        if s["primitive_function"] != "LOCAL":
            line += (f" | api {s['tool_name']} / {s['api_name']} | args "
                     f"{json.dumps(s.get('arguments') or {}, ensure_ascii=False)}")
        out.append(line)
    return "\n".join(out)


def plan_lines(plan):
    return "\n".join(f"{s['step_num']}. [{s['primitive_function']}] {step_line(s)}" for s in plan)


@torch.no_grad()
def generate(model, tok, convs, bs, max_new):
    out = []
    for b in range(0, len(convs), bs):
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in convs[b:b + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  generated {min(b + bs, len(convs))}/{len(convs)}", flush=True)
    return out


def shots(system, examples):
    msgs = [{"role": "system", "content": system}]
    for user, assistant in examples:
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return msgs


def parse_plan(text):
    plan = []
    for line in text.strip().splitlines():
        m = LINE.match(line)
        if m and len(plan) < MAX_STEPS:
            plan.append(parse_step(m[3], len(plan) + 1, m[2]))
    return plan


def run_onepass(model, tok, rows, shot_rows, ret, bs):
    ex = [(f"Query: {r['query']}\n\nWrite the final plan.", final_lines(r["final_plan"])) for r in shot_rows[:1]]
    base = shots(ONEPASS_SYSTEM, ex)
    t0 = time.time()
    raw = generate(model, tok, [base + [{"role": "user", "content": f"Query: {r['query']}\n\nWrite the final plan."}]
                                for r in rows], bs, 1100)
    secs = time.time() - t0
    reg_pairs = {(norm(r["tool_name"]), norm(r["api_name"])) for r in ret.reg}
    reg_tools = {norm(r["tool_name"]) for r in ret.reg}
    api_re = re.compile(r"\s\|\s*api\s+(?P<name>.+?)\s*\|\s*args\s*(?P<args>\{.*\})\s*$")
    out_re = re.compile(r"\s\|\s*out\s+(?P<out>\S*)\s*$")
    plans, stats = [], dict(gold=0, exact=0, tool=0, gen_api=0, valid_pair=0, valid_tool=0, parsed=0, lines=0)
    for r, text in zip(rows, raw):
        plan, gen_pairs = [], []
        for line in text.strip().splitlines():
            stats["lines"] += 1
            m = LINE.match(line)
            if not m:
                continue
            stats["parsed"] += 1
            n, label, body = len(plan) + 1, m[2], m[3]
            am = api_re.search(body)
            if am:
                body = body[:am.start()]
                tool, _, api = am["name"].partition(" / ")
                gen_pairs.append((norm(tool), norm(api)))
                stats["gen_api"] += 1
                stats["valid_pair"] += (norm(tool), norm(api)) in reg_pairs
                stats["valid_tool"] += norm(tool) in reg_tools
            om = out_re.search(body)
            body = body[:om.start()] if om else body
            plan.append({"step_num": n, "canonical_intent": body.strip(), "primitive_function": label,
                         "parameters": {}, "depends_on": [], "requires_api": label != "LOCAL"})
        plans.append(plan)
        gp = {p for p in gen_pairs}
        gt = {p[0] for p in gen_pairs}
        for step in r["canonical_plan"]:
            if not step.get("requires_api"):
                continue
            stats["gold"] += 1
            acc = [ret.record(t) for t in step["acceptable_tools"] if t in ret.idx_of]
            stats["exact"] += any((norm(a["tool_name"]), norm(a["api_name"])) in gp for a in acc)
            stats["tool"] += any(norm(a["tool_name"]) in gt for a in acc)
    res = {"seconds": round(secs, 1), "n": len(rows),
           "gold_api_steps": stats["gold"],
           "gold_api_exact_name_recall_%": round(100 * stats["exact"] / stats["gold"], 2),
           "gold_tool_name_recall_%": round(100 * stats["tool"] / stats["gold"], 2),
           "generated_api_steps": stats["gen_api"],
           "invented_api_pair_%": round(100 * (1 - stats["valid_pair"] / max(stats["gen_api"], 1)), 2),
           "invented_tool_%": round(100 * (1 - stats["valid_tool"] / max(stats["gen_api"], 1)), 2),
           "lines_parsed_%": round(100 * stats["parsed"] / max(stats["lines"], 1), 2)}
    return plans, raw, res


def run_twopass(model, tok, rows, shot_rows, ret, bs):
    d_ex = [(f"Query: {r['query']}\n\nWrite the plan.", plan_lines(r["canonical_plan"])) for r in shot_rows[:2]]
    d_base = shots(DISC_SYSTEM, d_ex)
    t0 = time.time()
    raw_d = generate(model, tok, [d_base + [{"role": "user", "content": f"Query: {r['query']}\n\nWrite the plan."}]
                                  for r in rows], bs, 800)
    t1 = time.time()
    plans = [parse_plan(t) for t in raw_d]
    cands = [retrieve_for_plan(p, ret) for p in plans]
    t2 = time.time()
    s = shot_rows[0]
    sc = gold_cands(s, ret)
    ex_msgs = compose_messages(s["query"], s["canonical_plan"], sc, ret)
    c_base = [ex_msgs[0], ex_msgs[1], {"role": "assistant", "content": target_text(s["final_plan"], sc)}]
    convs = [c_base + [compose_messages(r["query"], p, c, ret)[1]] for r, p, c in zip(rows, plans, cands)]
    raw_c = generate(model, tok, convs, max(1, bs // 2), 1100)
    t3 = time.time()
    composed = [parse_composition(t, c, ret) for t, c in zip(raw_c, cands)]
    res = {"seconds": {"discovery": round(t1 - t0, 1), "retrieval": round(t2 - t1, 1), "composition": round(t3 - t2, 1),
                       "total": round(t3 - t0, 1)}, "n": len(rows),
           "discovery_plans_empty": sum(not p for p in plans)}
    return plans, composed, raw_d, raw_c, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["onepass", "twopass"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_split(args.split)[:args.limit or None]
    shot_rows = load_split("train")[:3]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    model = load_base_model(four_bit=False).eval()
    ret = Retriever()
    torch.cuda.reset_peak_memory_stats()

    if args.mode == "onepass":
        plans, raw, res = run_onepass(model, tok, rows, shot_rows, ret, args.bs)
        with open(f"{args.out}_raw.jsonl", "w") as f:
            for r, t in zip(rows, raw):
                f.write(json.dumps({"sample_id": r["sample_id"], "raw": t}) + "\n")
    else:
        plans, composed, raw_d, raw_c, res = run_twopass(model, tok, rows, shot_rows, ret, args.bs)
        with open(f"{args.out}_composed.jsonl", "w") as f:
            for r, (steps, ok), t in zip(rows, composed, raw_c):
                f.write(json.dumps({"sample_id": r["sample_id"], "cond": "base_twopass", "ok": ok, "steps": steps,
                                    "raw": t}) + "\n")
    with open(f"{args.out}_plans.jsonl", "w") as f:
        for r, p in zip(rows, plans):
            f.write(json.dumps({"sample_id": r["sample_id"], "query": r["query"], "pred_plan": p}) + "\n")
    res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    json.dump(res, open(f"{args.out}_base.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
