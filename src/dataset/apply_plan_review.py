"""Apply reviewer corrections to plan_dataset_v2.jsonl, with an audit trail.

Corrections file: JSON list of ops, each with "sample_id", "op", "reason":
    {"op": "set_step",  "step_num": 3, "fields": {"canonical_intent": ..., "primitive_function": ..., "parameters": {...}}}
    {"op": "drop_step", "step_num": 6}
    {"op": "set_query", "query": "..."}
    {"op": "set_final", "step_num": 3, "fields": {"operation": ..., "arguments": {...}}}
    {"op": "set_acceptable", "step_num": 3, "acceptable_tools": [registry_id, ...]}
    {"op": "reject"}                      # moves the sample to the rejects file
    {"op": "approve"}                     # reviewed, kept unchanged (recorded in rec["review"])
On grounded samples (with final_plan), drop_step and label changes are refused:
they would desync the final plan, so such samples must be rejected instead.
step_num always refers to the sample's numbering BEFORE this review. Drops are
applied last; later steps are renumbered and depends_on is remapped (a dropped
step's own dependencies are inherited by the steps that depended on it).

Every change is appended to rec["review"]["changes"], and the sample is
re-validated with the generator's validator; samples that fail are reported and
left unchanged.

Usage:
    python3 src/dataset/apply_plan_review.py corrections.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_plan_dataset import API_PRIMITIVES, PRIMITIVE_IDS, locked, validate  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REVIEWER = "llm-reviewer"


def drop_steps(plan, to_drop):
    by_num = {s["step_num"]: s for s in plan}

    def resolve(deps):
        out = []
        for d in deps:
            out.extend(resolve(by_num[d]["depends_on"]) if d in to_drop else [d])
        return out

    kept = [s for s in plan if s["step_num"] not in to_drop]
    renum = {s["step_num"]: i for i, s in enumerate(kept, 1)}
    for s in kept:
        s["depends_on"] = sorted({renum[d] for d in resolve(s["depends_on"])})
    for s in kept:
        s["step_num"] = renum[s["step_num"]]
    return kept


def apply_ops(rec, ops):
    plan = rec["canonical_plan"]
    by_num = {s["step_num"]: s for s in plan}
    to_drop, log = set(), []
    for op in ops:
        kind = op["op"]
        if kind == "set_step":
            step = by_num[op["step_num"]]
            before = {k: step[k] for k in op["fields"]}
            step.update(op["fields"])
            prim = step["primitive_function"]
            step["primitive_id"] = PRIMITIVE_IDS[prim]
            step["requires_api"] = prim in API_PRIMITIVES
            log.append({"op": kind, "step_num": op["step_num"], "before": before,
                        "after": op["fields"], "reason": op["reason"]})
        elif kind == "drop_step":
            to_drop.add(op["step_num"])
            log.append({"op": kind, "step_num": op["step_num"],
                        "before": by_num[op["step_num"]]["canonical_intent"],
                        "reason": op["reason"]})
        elif kind == "set_final":
            entry = next(e for e in rec["final_plan"] if e["step_num"] == op["step_num"])
            before = {k: entry.get(k) for k in op["fields"]}
            entry.update(op["fields"])
            log.append({"op": kind, "step_num": op["step_num"], "before": before,
                        "after": op["fields"], "reason": op["reason"]})
        elif kind == "set_acceptable":
            step = by_num[op["step_num"]]
            log.append({"op": kind, "step_num": op["step_num"], "before": step["acceptable_tools"],
                        "after": op["acceptable_tools"], "reason": op["reason"]})
            step["acceptable_tools"] = op["acceptable_tools"]
        elif kind == "approve":
            log.append({"op": kind, "reason": op["reason"]})
        elif kind == "set_query":
            log.append({"op": kind, "before": rec["query"], "after": op["query"],
                        "reason": op["reason"]})
            rec["query"] = op["query"]
    if to_drop:
        rec["canonical_plan"] = drop_steps(plan, to_drop)
    rec["n_steps"] = len(rec["canonical_plan"])
    rec["n_api_steps"] = sum(s["requires_api"] for s in rec["canonical_plan"])
    review = rec.setdefault("review", {"reviewer": REVIEWER, "changes": []})
    review["changes"].extend(log)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corrections")
    ap.add_argument("--dataset",
                    default=f"{REPO}/data/processed/plan_dataset_v3.jsonl")
    args = ap.parse_args()
    rejects_path = args.dataset.replace(".jsonl", ".rejects.jsonl")

    ops_by_id = {}
    for op in json.load(open(args.corrections)):
        ops_by_id.setdefault(op["sample_id"], []).append(op)

    # Hold the lock for the whole read-modify-write so a running generator can't append mid-rewrite.
    with locked(args.dataset):
        review(args.dataset, rejects_path, ops_by_id)


def review(dataset, rejects_path, ops_by_id):
    out, n_fixed = [], 0
    for line in open(dataset):
        rec = json.loads(line)
        ops = ops_by_id.pop(rec["sample_id"], None)
        if ops is None:
            out.append(rec)
            continue
        if any(op["op"] == "reject" for op in ops):
            reason = "; ".join(op["reason"] for op in ops if op["op"] == "reject")
            with open(rejects_path, "a") as f:
                f.write(json.dumps({**rec, "errors": [f"review reject: {reason}"],
                                    "rejected_by": REVIEWER}) + "\n")
            print(f"{rec['sample_id']}: rejected ({reason})")
            continue
        structural = [op for op in ops if op["op"] == "drop_step" or (
            op["op"] == "set_step" and "primitive_function" in op.get("fields", {}))]
        if "final_plan" in rec and structural:
            print(f"{rec['sample_id']}: corrections NOT applied, drop_step / label changes would "
                  f"desync the grounded final_plan; reject the sample instead")
            out.append(rec)
            continue
        fixed = apply_ops(json.loads(line), ops)
        new = fixed.get("prompt_version", "") >= "plan_v3.6"
        errors, warnings = validate(fixed, set(), min_api=4 if new else 3, check_invented=new)
        for s in fixed["canonical_plan"]:
            gt = s.get("ground_truth_tool")
            if gt and gt["registry_id"] not in s.get("acceptable_tools", []):
                errors.append(f"step {s['step_num']}: acceptable_tools must include the chosen API")
        if errors:
            print(f"{rec['sample_id']}: corrections NOT applied, validator errors: {errors}")
            out.append(rec)
            continue
        fixed["qc"]["warnings"] = warnings
        out.append(fixed)
        n_fixed += 1
        print(f"{rec['sample_id']}: {len(ops)} change(s) applied")
    for sid in ops_by_id:
        print(f"{sid}: not found in dataset")

    with open(dataset, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print(f"{n_fixed} sample(s) corrected, {len(out)} kept")


if __name__ == "__main__":
    main()
