"""Repair queued samples and add gap-filling samples.

generate_plan_dataset.py parks every sample whose skeleton or grounding still fails
after its retries in plan_dataset_v3.repair.jsonl, together with the candidate pools
and the judge's last output. This tool turns them into complete samples without
spending Gemini requests, and adds new samples for coverage gaps. Both go through
the same checks as generated samples: the generator's validator, the same
candidate retriever (pools are recomputed, so chosen ids must come from them), and
ground_plan.validate_composition / attach.

    python3 src/dataset/repair_plan.py queue                 # list the repair queue
    python3 src/dataset/repair_plan.py show ID [ID ...]      # errors, plan, last output, candidates
    python3 src/dataset/repair_plan.py draft ID [ID ...] -o draft.json  # judge output -> draft
    python3 src/dataset/repair_plan.py candidates draft.json # candidates for draft plans
    python3 src/dataset/repair_plan.py commit draft.json     # validate + append to the dataset
    python3 src/dataset/repair_plan.py check draft.json      # commit dry run; shows failing pools
    python3 src/dataset/repair_plan.py find "text" [...]      # search the registry (same retriever)
    python3 src/dataset/repair_plan.py discard ID "reason" [--final]  # queue item -> rejects
    python3 src/dataset/repair_plan.py reground ID [ID ...]  # re-run the Gemini judge (1-3 requests)

draft.json is a list of
    {"sample_id": ..., "query": ..., "canonical_plan": [...],   # both default to the queued item
     "meta": {"domain", "task_type", "persona", "style", "delivery"},  # new samples only
     "api_choices": [{"step_num", "chosen_id", "acceptable_ids", "rationale"}],
     "final_plan": [{"step_num", "registry_id", "operation", "arguments": {...}, "output"}]}
canonical_plan steps use the generator's shape (parameters as a dict).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_plan_dataset import TAXONOMY, locked, normalize, validate  # noqa: E402
from ground_plan import (CandidateIndex, _params, attach, fix_refs, ground,  # noqa: E402
                         validate_composition)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = f"{REPO}/data/processed"
DATASET = f"{ROOT}/plan_dataset_v3.jsonl"
REPAIR = DATASET.replace(".jsonl", ".repair.jsonl")
REJECTS = DATASET.replace(".jsonl", ".rejects.jsonl")
REVIEWER = "llm-reviewer"
FILL_VERSION = "llm_fill_v1"
TOOLBENCH_VERSION = "toolbench_convert_v1"
# ToolBench-converted drafts take their ground truth from the API the original trace called,
# so it need not be in our retriever's pool (the pool is still stored for reference).
POOL_ERROR_RE = re.compile(r"(is|are) not in that step's candidates")


def _load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def _write(path, recs):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    os.replace(tmp, path)


def _raw(plan):
    """Stored/draft plan -> the model-output shape normalize() expects."""
    return [{"step_num": s["step_num"], "canonical_intent": s["canonical_intent"],
             "primitive_function": s["primitive_function"], "depends_on": s.get("depends_on", []),
             "parameters": [{"name": k, "value": str(v)}
                            for k, v in (s.get("parameters") or {}).items()]}
            for s in plan]


def _composition(d):
    def args(a):
        return [{"name": k, "value": str(v)} for k, v in a.items()] if isinstance(a, dict) else a
    return {"api_choices": d["api_choices"], "plan_defects": [],
            "final_plan": [{**e, "arguments": args(e.get("arguments", {}))}
                           for e in d["final_plan"]]}


def show_plan(rec, index, pools=None):
    print(f"Q: {rec['query']}")
    for s in rec["canonical_plan"]:
        print(f"  {s['step_num']:2}. {s['primitive_function']:7} {s['canonical_intent']:<48} "
              f"d={s['depends_on']} {json.dumps(s['parameters']) if s['parameters'] else ''}")
    for s in rec["canonical_plan"]:
        if s["primitive_function"] == "LOCAL":
            continue
        pool = (pools or {}).get(str(s["step_num"])) or index.candidates(s["canonical_intent"],
                                                                         rec["query"])
        print(f"\n  candidates for step {s['step_num']} ({s['primitive_function']}) "
              f"'{s['canonical_intent']}':")
        for rid in pool:
            print("  " + index.doc(rid))


def cmd_queue(_):
    for r in _load(REPAIR):
        print(f"{r['sample_id']}  [{r.get('repair_stage')}]  {r.get('errors', [])[:2]}")
        print(f"    {r.get('query', '')[:150]}")


def cmd_show(a):
    index = CandidateIndex()
    queue = {r["sample_id"]: r for r in _load(REPAIR)}
    for sid in a.ids:
        r = queue[sid]
        print(f"\n######## {sid} [{r.get('repair_stage')}] {r.get('domain')} | {r.get('delivery')}")
        print("errors:", json.dumps(r.get("errors"), indent=1))
        g = r.get("grounding") or {}
        if g.get("last_output"):
            print("judge last output:", json.dumps(g["last_output"])[:3000])
        show_plan(r, index, g.get("candidate_pools"))


def cmd_draft(a):
    """Seed drafts from the judge's last output (refs auto-fixed) and print a compact view."""
    index = CandidateIndex()
    queue = {r["sample_id"]: r for r in _load(REPAIR)}
    drafts = []
    for sid in a.ids:
        r = queue[sid]
        last = dict((r.get("grounding") or {}).get("last_output") or {})
        last.setdefault("api_choices", [])
        last.setdefault("final_plan", [])
        fix_refs(last)
        d = {"sample_id": sid, "api_choices": last["api_choices"],
             "final_plan": [{**e, "arguments": {x["name"]: x["value"] for x in e["arguments"]}}
                            for e in last["final_plan"]]}
        drafts.append(d)
        print(f"\n######## {sid}  errors: {r.get('errors', [])[:3]}")
        print(f"Q: {r['query']}")
        choice = {c["step_num"]: c for c in d["api_choices"]}
        final = {e["step_num"]: e for e in d["final_plan"]}
        for s in r["canonical_plan"]:
            e, c = final.get(s["step_num"], {}), choice.get(s["step_num"])
            print(f"  {s['step_num']:2}. {s['primitive_function']:7} {s['canonical_intent']:<46} "
                  f"d={s['depends_on']} {json.dumps(s['parameters']) if s['parameters'] else ''}")
            if c and c["chosen_id"] != -1:
                print(f"      -> {index.doc(c['chosen_id']).strip()[:230]}")
                print(f"         acceptable={c['acceptable_ids']} args={json.dumps(e.get('arguments'))}"
                      f" -> {e.get('output')}")
            elif c:
                print("      -> chosen_id -1")
            else:
                print(f"      op: {e.get('operation', '')[:160]} -> {e.get('output')}")
    json.dump(drafts, open(a.o, "w"), indent=1)
    print(f"\nwrote {len(drafts)} draft(s) to {a.o}")


def cmd_candidates(a):
    index = CandidateIndex()
    queue = {r["sample_id"]: r for r in _load(REPAIR)}
    for d in json.load(open(a.draft)):
        base = queue.get(d["sample_id"], {})
        rec = normalize({"query": d.get("query", base.get("query")),
                         "canonical_plan": _raw(d.get("canonical_plan") or base["canonical_plan"])})
        print(f"\n######## {d['sample_id']}")
        show_plan(rec, index)


def _short(index, rid):
    r = index.entry(rid)
    names = lambda key: ",".join(p.get("name", "") for p in _params(r.get(key))[:6])
    return (f"    {rid:>5} {r['tool_name'][:28]} / {r['api_name'][:34]} "
            f"req[{names('required_parameters')}] opt[{names('optional_parameters')}] "
            f"- {str(r.get('api_description') or '')[:50].strip()}").replace("\n", " ")


def cmd_find(a):
    index = CandidateIndex()
    for text in a.text:
        print(f"## {text}")
        for rid in index.candidates(text, text)[:a.k]:
            print(_short(index, rid))


def cmd_check(a):
    a.dry_run = True
    cmd_commit(a)


def cmd_commit(a):
    dry = getattr(a, "dry_run", False)
    index = CandidateIndex()
    with locked(DATASET):
        dataset, queue = _load(DATASET), {r["sample_id"]: r for r in _load(REPAIR)}
        seen_ids = {r["sample_id"] for r in dataset}
        seen_q = {r["query"].lower() for r in dataset}
        committed = []
        for d in json.load(open(a.draft)):
            sid, base = d["sample_id"], queue.get(d["sample_id"])
            if sid in seen_ids:
                print(f"{sid}: already in the dataset, skipped")
                continue
            rec = normalize({"query": d.get("query", (base or {}).get("query")),
                             "canonical_plan": _raw(d.get("canonical_plan")
                                                    or base["canonical_plan"])})
            errors, warnings = validate(rec, set())
            if rec["query"].lower() in seen_q:
                errors.append("duplicate query")
            pools = {s["step_num"]: index.candidates(s["canonical_intent"], rec["query"])
                     for s in rec["canonical_plan"] if s["requires_api"]}
            if not errors:
                g_errors, ungrounded = validate_composition(rec, pools, _composition(d), index)
                if d.get("source") == "toolbench":
                    g_errors = [e for e in g_errors if not POOL_ERROR_RE.search(e)]
                errors += g_errors + [f"step {n} has chosen_id -1" for n in ungrounded]
            if errors:
                print(f"{sid}: NOT committed\n  - " + "\n  - ".join(errors))
                if dry:
                    bad = {int(n) for n in re.findall(r"step (\d+)", " ".join(errors))}
                    for n in sorted(bad & set(pools)):
                        print(f"  pool for step {n}:")
                        print("\n".join(_short(index, rid) for rid in pools[n]))
                continue
            if dry:
                print(f"{sid}: ok")
                continue
            if base:
                meta = {k: base[k] for k in ("sample_id", "domain", "task_type", "persona", "style",
                                             "delivery", "generator_model", "prompt_version",
                                             "attempts") if k in base}
                meta["repair"] = {"by": REVIEWER, "stage": base.get("repair_stage"),
                                  "original_errors": base.get("errors", [])}
            elif d.get("source") == "toolbench":
                meta = {"sample_id": sid, **d["meta"], "generator_model": REVIEWER,
                        "prompt_version": TOOLBENCH_VERSION, "attempts": 1,
                        "source": {"dataset": "toolbench", "file": d["source_file"],
                                   "query": d["source_query"], "ground_truth": "trace api calls"}}
            else:
                meta = {"sample_id": sid, **d["meta"], "generator_model": REVIEWER,
                        "prompt_version": FILL_VERSION, "attempts": 1}
            out = {**meta, "taxonomy": TAXONOMY, **rec,
                   "n_steps": len(rec["canonical_plan"]),
                   "n_api_steps": sum(s["requires_api"] for s in rec["canonical_plan"]),
                   "qc": {"errors": [], "warnings": warnings}}
            out = attach(out, _composition(d), pools, index, REVIEWER, attempts=1)
            dataset.append(out)
            seen_ids.add(sid)
            seen_q.add(rec["query"].lower())
            committed.append(sid)
            print(f"{sid}: committed")
        if dry:
            return
        _write(DATASET, dataset)
        _write(REPAIR, [r for s, r in queue.items() if s not in committed])
    print(f"{len(committed)} committed")


def cmd_reground(a):
    """Re-run the grounding judge on queued grounding-stage items, e.g. after a prompt fix.
    The Gemini call happens outside the lock; the queue is re-read under it, so items the
    running generator appends meanwhile are kept."""
    from generate_plan_dataset import QuotaExhausted, call_model, genai
    index = CandidateIndex()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    ungrounded_path = DATASET.replace(".jsonl", ".ungrounded.jsonl")
    queue = {r["sample_id"]: r for r in _load(REPAIR)}
    for sid in a.ids:
        r = queue[sid]
        if r.get("repair_stage") != "grounding":
            print(f"{sid}: skeleton failed validation, fix the plan first")
            continue
        rec = {k: v for k, v in r.items() if k not in ("grounding", "errors", "repair_stage")}
        try:
            out, status, errs = ground(rec, index, lambda c, s, sch: call_model(
                client, a.model, c, s, sch, temperature=0.2), judge_model=a.model)
        except QuotaExhausted:
            print("daily quota exhausted; stopping")
            break
        with locked(DATASET):
            if status == "complete":
                out["repair"] = {"by": "reground", "original_errors": r.get("errors", [])}
                with open(DATASET, "a") as f:
                    f.write(json.dumps(out) + "\n")
            elif status == "ungrounded":
                with open(ungrounded_path, "a") as f:
                    f.write(json.dumps({**out, "errors": errs}) + "\n")
            fresh = _load(REPAIR)
            if status in ("complete", "ungrounded"):
                fresh = [x for x in fresh if x["sample_id"] != sid]
            else:
                fresh = [{**out, "repair_stage": "grounding", "errors": errs}
                         if x["sample_id"] == sid else x for x in fresh]
            _write(REPAIR, fresh)
        print(f"{sid}: {status} {errs[:2]}")


def cmd_discard(a):
    with locked(DATASET):
        queue = _load(REPAIR)
        hit = [r for r in queue if r["sample_id"] == a.id]
        if not hit:
            sys.exit(f"{a.id} not in the repair queue")
        with open(REJECTS, "a") as f:
            f.write(json.dumps({**hit[0], "rejected_by": REVIEWER, "reject_reason": a.reason,
                                "no_retry": a.final}) + "\n")
        _write(REPAIR, [r for r in queue if r["sample_id"] != a.id])
    print(f"{a.id}: discarded")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("queue")
    p = sub.add_parser("show")
    p.add_argument("ids", nargs="+")
    p = sub.add_parser("draft")
    p.add_argument("ids", nargs="+")
    p.add_argument("-o", required=True)
    for name in ("candidates", "commit", "check"):
        sub.add_parser(name).add_argument("draft")
    p = sub.add_parser("find")
    p.add_argument("text", nargs="+")
    p.add_argument("-k", type=int, default=10)
    p = sub.add_parser("reground")
    p.add_argument("ids", nargs="+")
    p.add_argument("--model", default="gemini-3.1-flash-lite")
    p = sub.add_parser("discard")
    p.add_argument("id")
    p.add_argument("reason")
    p.add_argument("--final", action="store_true",
                   help="never regenerate this id (e.g. the registry has no API for the task)")
    a = ap.parse_args()
    {"queue": cmd_queue, "show": cmd_show, "draft": cmd_draft, "candidates": cmd_candidates,
     "commit": cmd_commit, "check": cmd_check, "find": cmd_find, "discard": cmd_discard,
     "reground": cmd_reground}[a.cmd](a)


if __name__ == "__main__":
    main()
