"""Where the pipelines lose gold API steps: by step type, per plan and by plan size.

  python3 src/error_breakdown.py
"""
import collections
import json

from mole_common import ROOT, load_split
from mole_compose import Retriever, retrieve_for_plan
ret = Retriever(device="cpu")
gold = {r["sample_id"]: r for r in load_split("test")}
R = f"{ROOT}/data/results/"
def plans(f): return {j["sample_id"]: j["pred_plan"] for j in map(json.loads, open(R+f))}
def comp(f): return {j["sample_id"]: j for j in map(json.loads, open(R+f))}
out = {}
for name, pf, cf in (("mole","pipeline_mole_plans.jsonl","pipeline_mole_composed.jsonl"),
                     ("single","pipeline_single_plans.jsonl","pipeline_single_composed.jsonl"),
                     ("base2","base_twopass_plans.jsonl","base_twopass_composed.jsonl")):
    P, C = plans(pf), comp(cf)
    st = collections.defaultdict(lambda: collections.Counter())
    per_plan = []
    for sid, g in gold.items():
        cands = retrieve_for_plan(P[sid], ret)
        allc = set(i for ids in cands.values() for i in ids)
        chosen = {s.get("registry_id") for s in C[sid]["steps"] if s.get("registry_id") is not None}
        gapi = [(s, c) for s, c in zip(g["final_plan"], g["canonical_plan"]) if s["primitive_function"] != "LOCAL"]
        n_rec = 0
        for s, c in gapi:
            a = set(c["acceptable_tools"]); t = s["primitive_function"]
            inc = bool(a & allc); sel = bool(a & chosen)
            for k in (t, "ALL"):
                st[k]["n"] += 1; st[k]["in_cands"] += inc; st[k]["chosen"] += sel
                st[k]["lost_by_comp"] += (inc and not sel)
            n_rec += sel
        per_plan.append((len(gapi), n_rec))
    out[name] = {k: dict(v) for k, v in st.items()}
    # per plan recovered fraction distribution
    fr = [r/n for n, r in per_plan if n]
    out[name]["plan_frac"] = {"zero": sum(f==0 for f in fr), "le_half": sum(0<f<=.5 for f in fr), "gt_half_not_all": sum(.5<f<1 for f in fr), "all": sum(f==1 for f in fr), "n": len(fr)}
    # by number of API steps
    b = collections.defaultdict(lambda:[0,0])
    for n, r in per_plan:
        k = "4" if n<=4 else "5" if n==5 else "6+"
        b[k][0]+=n; b[k][1]+=r
    out[name]["by_api_count"] = {k: [v[0], round(100*v[1]/v[0],1)] for k,v in b.items()}
print(json.dumps(out, indent=1))
json.dump(out, open(R + "pipeline_error_breakdown.json", "w"), indent=1)
