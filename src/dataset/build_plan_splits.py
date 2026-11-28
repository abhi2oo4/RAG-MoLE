"""Build the released RWPL plan dataset: splits, per-pass training views, stats and a data card.

Reads data/processed/plan_dataset_v3.jsonl (complete samples only) and writes data/processed/plan_v3/:
    {train,val,test}.jsonl     full records of our own samples (Gemini skeletons + LLM-written fills)
    discovery_<split>.jsonl    query -> ordered canonical steps (Discovery / MoLE training target)
    steps_<split>.jsonl        one row per step: intent, RWPL label, ground-truth tool, acceptable tools
                               (expert/router training and per-step retrieval evaluation)
    composition_<split>.jsonl  query + canonical plan -> gold final plan (Composition reference)
    stats.json, README.md

Splits are assigned by a hash of the sample's group id, so a sample keeps its split when the
dataset grows. Queries that are near-duplicates (TF-IDF cosine >= DUP_COSINE) share a group,
so paraphrases never straddle train and test.

    python3 src/dataset/build_plan_splits.py
"""
import argparse
import collections
import hashlib
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = f"{REPO}/data/processed"
DUP_COSINE = 0.8
SPLITS = ("train", "val", "test")


def source(rec):
    v = rec.get("prompt_version", "")
    if v.startswith("toolbench"):
        return "toolbench"
    return "gemini_skeleton" if v.startswith("plan_v") else "llm_fill"


def groups_by_similarity(recs):
    """Union-find over near-duplicate queries; returns sample_id -> group id (smallest member id)."""
    ids = [r["sample_id"] for r in recs]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    m = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform([r["query"] for r in recs])
    sim = (m @ m.T).toarray()
    for a, b in zip(*np.where(np.triu(sim, 1) >= DUP_COSINE)):
        ra, rb = find(ids[a]), find(ids[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in ids}


def split_of(group_id):
    bucket = int(hashlib.sha1(group_id.encode()).hexdigest(), 16) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def discovery_view(r):
    return {"sample_id": r["sample_id"], "query": r["query"],
            "steps": [{"step_num": s["step_num"], "intent": s["canonical_intent"],
                       "primitive": s["primitive_function"], "depends_on": s["depends_on"],
                       "parameters": s["parameters"]} for s in r["canonical_plan"]]}


def step_rows(r):
    for s in r["canonical_plan"]:
        gt = s.get("ground_truth_tool")
        yield {"sample_id": r["sample_id"], "step_num": s["step_num"], "query": r["query"],
               "intent": s["canonical_intent"], "primitive": s["primitive_function"],
               "requires_api": s["requires_api"],
               "ground_truth_id": gt["registry_id"] if gt else None,
               "ground_truth_tool": f"{gt['tool_name']} / {gt['api_name']}" if gt else None,
               "acceptable_ids": s.get("acceptable_tools", [])}


def composition_view(r):
    return {"sample_id": r["sample_id"], "query": r["query"],
            "canonical_plan": [{k: s[k] for k in ("step_num", "canonical_intent", "primitive_function",
                                                 "depends_on", "parameters")} for s in r["canonical_plan"]],
            "final_plan": r["final_plan"]}


def write(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)


def stats(recs):
    prims = collections.Counter(s["primitive_function"] for r in recs for s in r["canonical_plan"])
    return {"samples": len(recs),
            "steps": sum(r["n_steps"] for r in recs),
            "steps_per_sample": collections.Counter(r["n_steps"] for r in recs),
            "api_steps_per_sample": collections.Counter(r["n_api_steps"] for r in recs),
            "primitives": prims,
            "sources": collections.Counter(source(r) for r in recs),
            "deliveries": collections.Counter(r.get("delivery") or "unspecified" for r in recs),
            "distinct_ground_truth_endpoints": len({s["ground_truth_tool"]["registry_id"] for r in recs
                                                    for s in r["canonical_plan"] if s.get("ground_truth_tool")}),
            "reviewed": sum(bool(r.get("review") or r.get("repair")) or source(r) != "gemini_skeleton"
                            for r in recs)}


def data_card(report):
    a = report["all"]
    rows = "\n".join(f"| {n} | {report[n]['samples']} | {report[n]['steps']} | "
                     f"{', '.join(f'{k} {v}' for k, v in sorted(report[n]['sources'].items()))} |"
                     for n in SPLITS)
    prims = ", ".join(f"{k} {v}" for k, v in sorted(a["primitives"].items()))
    lengths = ", ".join(f"{k} steps: {v}" for k, v in sorted(a["steps_per_sample"].items()))
    return f"""# RWPL plan dataset (v3)

Natural-language automation requests paired with (1) a canonical step plan labelled with the RWPL
taxonomy, (2) the ground-truth API endpoint for every API step, drawn from `master_registry.json`,
and (3) a gold executable plan. It trains and evaluates the RAG-MoLE Discovery pass (query -> steps
-> retrieval) and serves as the reference for the Composition pass.

Generated by `src/dataset/build_plan_splits.py` from `data/processed/plan_dataset_v3.jsonl`.

## Contents

{a['samples']} samples, {a['steps']} steps, {a['distinct_ground_truth_endpoints']} distinct ground-truth endpoints.

| split | samples | steps | sources |
|---|---|---|---|
{rows}

- Step labels: {prims}
- Plan lengths: {lengths}

## Taxonomy (RWPL)

| label | meaning |
|---|---|
| READ | an API call that retrieves stored data without changing it |
| WRITE | an API call that changes state in an external system (send, post, create, book, upload) |
| PROCESS | an API call that computes or transforms its input (translate, summarize, convert, OCR) |
| LOCAL | no API call: parsing, filtering, ranking, arithmetic, formatting |

## Sources

- **gemini_skeleton**: Gemini 3.1 Flash-Lite generated the query and plan, a Gemini judge grounded
  the plan, and the samples then went through an LLM-assisted review. Corrections are recorded in
  `review` (or `repair`, when a failed sample was fixed).
- **llm_fill**: written by a large language model (prompt_version `llm_fill_v1`) to fill coverage gaps: more
  WRITE steps, deliveries other than spreadsheets, and longer plans.

All three sources pass the same checks. The validator in `generate_plan_dataset.py` checks:
- 10-15 steps, with 4-6 API steps (3-6 for the six samples made before plan_v3.6);
- intents of 4-8 words that name no vendor or user value;
- no filler steps;
- parameter values taken from the query.

The composition check in `ground_plan.py` then checks:
- every chosen endpoint is in the step's candidate pool (BGE-M3 + TF-IDF, which is not the
  retriever under test);
- required arguments are present;
- every dependency passes data through a `$stepN.<output>` reference;
- READ/WRITE labels match the endpoint names.

## Files

| file | one row per | use |
|---|---|---|
| `<split>.jsonl` | sample | full records |
| `discovery_<split>.jsonl` | sample | Discovery target: query -> ordered steps (intent, primitive, depends_on, parameters) |
| `steps_<split>.jsonl` | step | expert/router training on (intent, primitive); retrieval eval with `ground_truth_id` / `acceptable_ids` |
| `composition_<split>.jsonl` | sample | Composition reference: query + canonical plan -> `final_plan` |
| `stats.json` | - | the numbers above, per split |

Retrieval metrics should count a hit when the retrieved id is in `acceptable_ids`. The registry has
many interchangeable endpoints (dozens of email senders), and `ground_truth_id` is one choice among
them.

## Gold plan conventions

- Arguments are literals from the query, `$stepN.<output>[.<field>]` references, or `$user.<name>`
  for values the user or account must supply (API keys, recipients, ids).
- `undocumented_arguments` lists arguments that a registry entry needs but does not document. For
  example, "Calendar / Create event" (12401) documents only `calendarId`.
- LOCAL steps carry `inputs` instead of an endpoint.

## Splits

A sample's split comes from a hash of its near-duplicate group, so it never changes as the dataset
grows. Queries with TF-IDF cosine >= {report['dup_cosine']} share a group, which keeps paraphrases on
one side of the train/test line ({report['near_duplicate_groups']} such groups). The ratio is about
80/10/10.

## Known limitations

- **Registry gaps:**
  - no endpoint creates a blog post: blog deliveries were moved to social posts;
  - calendar-event and spreadsheet endpoints lack data parameters (flagged in `undocumented_arguments`);
  - no spreadsheet-read endpoints;
  - no traffic or trending-by-city APIs.
- **Endpoint concentration:** a few generic endpoints dominate the ground truth (one email sender, one
  file upload, one calendar endpoint). Use `acceptable_ids` rather than exact-id matching.
- **Size:** the dataset is small for training. Paraphrase augmentation of the queries (plans
  unchanged) is recommended before LoRA training.
- **Example values:** user-specific values in the gold plans are examples. The plans have never been
  executed against the live APIs.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=f"{ROOT}/plan_dataset_v3.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/plan_v3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    recs = [json.loads(l) for l in open(a.dataset)]
    recs = [r for r in recs if (r.get("grounding") or {}).get("status") == "complete"]
    ours = recs
    group = groups_by_similarity(ours)
    by_split = {s: [] for s in SPLITS}
    for r in ours:
        r["split"] = split_of(group[r["sample_id"]])
        r["dup_group"] = group[r["sample_id"]]
        by_split[r["split"]].append(r)

    report = {"dup_cosine": DUP_COSINE,
              "near_duplicate_groups": sum(1 for g, n in collections.Counter(group.values()).items() if n > 1)}
    for name, rows in by_split.items():
        write(f"{a.out}/{name}.jsonl", rows)
        write(f"{a.out}/discovery_{name}.jsonl", map(discovery_view, rows))
        write(f"{a.out}/steps_{name}.jsonl", (x for r in rows for x in step_rows(r)))
        write(f"{a.out}/composition_{name}.jsonl", map(composition_view, rows))
        report[name] = stats(rows)
    report["all"] = stats(recs)
    json.dump(report, open(f"{a.out}/stats.json", "w"), indent=1)
    open(f"{a.out}/README.md", "w").write(data_card(report))
    for name in (*by_split, "all"):
        s = report[name]
        print(f"{name:15} {s['samples']:4} samples {s['steps']:5} steps  {dict(s['primitives'])}  "
              f"sources={dict(s['sources'])}")


if __name__ == "__main__":
    main()
