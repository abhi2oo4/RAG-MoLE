"""Stage 2 of plan-dataset generation: ground API steps in master_registry.json and
write the reference (gold) final implementation plan.

For each API step (READ/WRITE/PROCESS) a candidate pool is built from the registry
with a retriever that is deliberately DIFFERENT from the one under evaluation
(benchmark_retrieval.py uses all-MiniLM-L6-v2 on canonical intents):
    BGE-M3 dense search on "intent + query context"  (top K_DENSE)
  ∪ BGE-M3 dense search on the intent alone          (top K_INTENT)
  ∪ TF-IDF lexical search on the intent               (top K_LEX)
  The intent-only search keeps generic steps ("send summary email") from being swamped by
  the query's topic, which otherwise fills the pool with topic endpoints and no sender.
Endpoints with an empty api_description are excluded: the judge cannot verify them.

Gemini then, from the candidates' docs, picks the best endpoint per step plus
every other candidate that is also correct (acceptable set, since the registry has
many interchangeable APIs), and writes the final plan: one entry per canonical
step, API steps with registry_id + arguments, LOCAL steps with an operation.
Arguments are literals from the query or references "$stepN.<field>" /
"$user.<field>", never invented values.

A sample is "complete" only if every API step is grounded and the plan passes
validation; otherwise the caller stores it separately as ungrounded.
"""
import ast
import json
import os
import re
import threading

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGISTRY_PATH = f"{REPO}/data/processed/master_registry.json"
DENSE_MODEL = "BAAI/bge-m3"
EMB_CACHE = f"{REPO}/data/processed/registry_emb_bge-m3.npy"
K_DENSE, K_INTENT, K_LEX = 12, 6, 8
CANDIDATE_METHOD = (f"bge-m3 dense(intent+query) top{K_DENSE} ∪ dense(intent) top{K_INTENT} "
                    f"∪ tfidf(intent) top{K_LEX}")

SYSTEM_PROMPT = """You produce the REFERENCE (gold) implementation plan for a task-planning benchmark.

Input: a user query, its canonical plan (steps labeled READ / WRITE / PROCESS / LOCAL with parameters and dependencies), and for every API step (READ/WRITE/PROCESS) a list of candidate API endpoints from a registry, each with an id, tool, endpoint name, HTTP method, description and parameters.

1. API selection, for every API step:
   - chosen_id: the single best candidate id from THAT step's list that can actually perform the step. Judge by the description and parameters (what the endpoint really does), not by name similarity.
   - The endpoint must match the step's label: READ retrieves stored data, WRITE changes state in an external system, PROCESS computes/transforms input and returns a result.
   - acceptable_ids: every candidate in that step's list that could ALSO correctly perform the step (same capability, usable inputs), including chosen_id.
   - If no candidate can perform the step: chosen_id = -1, acceptable_ids = []. Never force a poor match.

2. final_plan: exactly one entry per canonical step, same step_num and order.
   - API step: registry_id = chosen_id. arguments = every REQUIRED parameter of that endpoint plus the optional ones this task needs, using the endpoint's exact parameter names. Each value is one of:
       * a literal value stated in the query or the step's parameters,
       * "$step<N>.<field>" - output of earlier step N (N < current step_num),
       * "$user.<field>" - a value the user must supply at runtime (an id, address, API account, etc.).
     Never invent concrete values the user did not give - including "harmless" defaults like a sheet number, a source currency, a source language or a result count; use "$user.<field>" or an earlier step's output instead.
     The data the step works on (the text to send, the rows to add, the item to book) must be passed through one of the endpoint's parameters as a $step reference. If the endpoint has no parameter that can carry it, that endpoint cannot perform the step.
     An id must come from the same service: never pass text as a post/draft/file id, and never pass an id returned by one vendor's API to another vendor's endpoint (a file id from one cloud drive is not valid on another). An endpoint that only acts on an existing object ("publish a draft", "get file by id") needs an earlier step that creates or lists that object on the same service; otherwise choose an endpoint that takes the content directly.
   - LOCAL step: registry_id = -1, operation = one precise sentence of what the code does, referencing inputs as $stepN.<field>. arguments = the inputs it consumes (same value rules as above), or [] if none.
   - API step operation: one short sentence of what the call does in this plan.
   - output: snake_case name of the step's result. Later steps reference it EXACTLY as $stepN.<output> or $stepN.<output>.<subfield> (e.g. step 4 output "shipping_rate" -> "$step4.shipping_rate.amount").
   - Every reference you use must be something step N really produces.

3. plan_defects: list real problems in the CANONICAL PLAN itself that make it non-executable, e.g. an API needs coordinates but no step produces them (missing geocoding), a step uses data that is only fetched in a later step, a required value can never be obtained. Empty list if the plan is sound. Do not paper over a defect with a wrong API or a made-up reference.
   NOT defects: an API step with no fitting candidate (use chosen_id -1 for it instead), and applying a step to each item of a list (iteration is handled by the executor).

Output ONLY the JSON object."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "api_choices": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "step_num": {"type": "INTEGER"},
                    "chosen_id": {"type": "INTEGER"},
                    "acceptable_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "rationale": {"type": "STRING"},
                },
                "required": ["step_num", "chosen_id", "acceptable_ids", "rationale"],
            },
        },
        "final_plan": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "step_num": {"type": "INTEGER"},
                    "registry_id": {"type": "INTEGER"},
                    "operation": {"type": "STRING"},
                    "arguments": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {"name": {"type": "STRING"}, "value": {"type": "STRING"}},
                            "required": ["name", "value"],
                        },
                    },
                    "output": {"type": "STRING"},
                },
                "required": ["step_num", "registry_id", "operation", "arguments", "output"],
            },
        },
        "plan_defects": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["api_choices", "final_plan", "plan_defects"],
}

# "No API available to ..." is a missing candidate, not a defect of the plan itself.
NO_API_RE = re.compile(r"\bno (available |suitable |valid )?(api|endpoint|candidate)s?\b|"
                       r"\bno [\w ]{0,30}(api|endpoint) (is )?(available|accepts|supports|provides|offers|can)\b",
                       re.I)
REF_RE = re.compile(r"\$step(\d+)(?:\.(\w+))?")
# Leading verb of an endpoint name ("Add rows", "createOrder", "get_stock") that
# contradicts the step label: a READ step can't be served by a writer, and vice versa.
WRITE_NAME_RE = re.compile(r"^(add|create|delete|remove|update|send|post|insert|upload|put|"
                           r"edit|cancel|book|clear|publish|set)\b"
                           # 'Post search' / 'post details' search posts; they do not write one.
                           r"(?!\s+(search|list|get|lookup|find|details?|info|by)\b)")
READ_NAME_RE = re.compile(r"^(get|list|search|fetch|read|find|retrieve|lookup|query)\b")


def _name_words(name):
    """'createOrder' / 'get_stock-price' -> 'create order' / 'get stock price'."""
    return re.sub(r"[_\-/]+", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", name)).lower().strip()


def _params(raw):
    try:
        val = ast.literal_eval(raw) if isinstance(raw, str) else (raw or [])
        return val if isinstance(val, list) else []
    except (ValueError, SyntaxError):
        return []


class CandidateIndex:
    def __init__(self, registry_path=REGISTRY_PATH, device="cuda"):
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._lock = threading.Lock()  # one GPU model shared by the generator's worker threads
        self.reg = json.load(open(registry_path))
        self.by_id = {int(r["registry_id"]): i for i, r in enumerate(self.reg)}
        texts = [f"{r['tool_name']} | {r['api_name']}: {r.get('api_description') or ''} "
                 f"{(r.get('tool_description') or '')[:200]}" for r in self.reg]
        self.eligible = np.array([bool(str(r.get("api_description") or "").strip())
                                  for r in self.reg])
        self.model = SentenceTransformer(DENSE_MODEL, device=device)
        self.model.half()
        self.model.max_seq_length = 256  # bge-m3 defaults to 8192; endpoint texts don't need it
        if os.path.exists(EMB_CACHE):
            self.emb = np.load(EMB_CACHE)
        else:
            self.emb = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                         show_progress_bar=True).astype(np.float16)
            np.save(EMB_CACHE, self.emb)
        self.tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)
        self.lex = self.tfidf.fit_transform(texts)

    def candidates(self, intent, query):
        with self._lock:
            q, qi = self.model.encode([f"{intent}. Context: {query}", intent], normalize_embeddings=True)
        dense = (self.emb @ q.astype(np.float16)).astype(np.float32)
        dense_i = (self.emb @ qi.astype(np.float16)).astype(np.float32)
        lex = (self.lex @ self.tfidf.transform([intent]).T).toarray().ravel()
        dense[~self.eligible] = dense_i[~self.eligible] = lex[~self.eligible] = -1
        picked = list(np.argsort(-dense)[:K_DENSE])
        picked += [i for i in np.argsort(-dense_i)[:K_INTENT * 2] if i not in picked][:K_INTENT]
        picked += [i for i in np.argsort(-lex)[:K_LEX * 2] if i not in picked][:K_LEX]
        return [int(self.reg[i]["registry_id"]) for i in picked]

    def entry(self, rid):
        return self.reg[self.by_id[rid]]

    def doc(self, rid):
        r = self.entry(rid)
        fmt = lambda ps: ", ".join(f"{p.get('name')}({p.get('type', '').lower()})" for p in ps[:8])
        return (f"  id={rid} | {r['tool_name']} / {r['api_name']} [{r.get('method')}] - "
                f"{str(r.get('api_description') or '')[:220].strip()}"
                f"\n      required: {fmt(_params(r.get('required_parameters'))) or 'none'}"
                f" | optional: {fmt(_params(r.get('optional_parameters'))) or 'none'}")


def build_prompt(rec, pools, index):
    lines = [f"QUERY: {rec['query']}", "", "CANONICAL PLAN:"]
    for s in rec["canonical_plan"]:
        lines.append(f"  {s['step_num']}. [{s['primitive_function']}] {s['canonical_intent']}"
                     f" | params={json.dumps(s['parameters'])} | depends_on={s['depends_on']}")
    lines += ["", "CANDIDATES PER API STEP:"]
    for num, pool in pools.items():
        lines.append(f"Step {num}:")
        lines += [index.doc(rid) for rid in pool]
    return "\n".join(lines)


def fix_refs(out):
    """'$step1.topic' where step 1's output is 'search_params' -> '$step1.search_params.topic'.
    The model names the field it wants but skips the output name; the fix is mechanical, so
    it is applied in place instead of spending a retry on it. Returns the number of fixes."""
    outputs = {e["step_num"]: e["output"] for e in out.get("final_plan", [])}
    n = 0

    def sub(m):
        nonlocal n
        ref, field = int(m.group(1)), m.group(2)
        if field and ref in outputs and field != outputs[ref]:
            n += 1
            return f"$step{ref}.{outputs[ref]}.{field}"
        return m.group(0)
    for e in out.get("final_plan", []):
        e["operation"] = REF_RE.sub(sub, e["operation"])
        for a in e["arguments"]:
            a["value"] = REF_RE.sub(sub, a["value"])
    return n


def validate_composition(rec, pools, out, index):
    """Return (errors, ungrounded_steps)."""
    errors, ungrounded = [], []
    plan = {s["step_num"]: s for s in rec["canonical_plan"]}
    choices = {c["step_num"]: c for c in out["api_choices"]}
    if set(choices) != set(pools):
        errors.append(f"api_choices must cover exactly API steps {sorted(pools)}, got {sorted(choices)}")
        return errors, ungrounded
    for num, c in choices.items():
        if c["chosen_id"] == -1:
            ungrounded.append(num)
            continue
        if c["chosen_id"] not in pools[num]:
            errors.append(f"step {num}: chosen_id {c['chosen_id']} is not in that step's candidates")
        bad = [a for a in c["acceptable_ids"] if a not in pools[num]]
        if bad:
            errors.append(f"step {num}: acceptable_ids {bad} are not in that step's candidates")
        if c["chosen_id"] not in c["acceptable_ids"]:
            errors.append(f"step {num}: acceptable_ids must include chosen_id")
        label = plan[num]["primitive_function"]
        for rid in c["acceptable_ids"]:
            if rid not in pools[num]:
                continue
            words = _name_words(index.entry(rid)["api_name"])
            if (label == "READ" and WRITE_NAME_RE.match(words)) or (
                    label == "WRITE" and READ_NAME_RE.match(words)):
                errors.append(f"step {num} is {label} but endpoint {rid} '{index.entry(rid)['api_name']}' "
                              f"does the opposite; pick a matching endpoint or chosen_id -1")

    fp = out["final_plan"]
    if [e["step_num"] for e in fp] != sorted(plan):
        errors.append(f"final_plan must have one entry per step 1..{len(plan)} in order")
        return errors, ungrounded
    outputs = {e["step_num"]: e["output"] for e in fp}
    for e in fp:
        num, step = e["step_num"], plan[e["step_num"]]
        for ref, field in REF_RE.findall(" ".join([e["operation"]] + [a["value"] for a in e["arguments"]])):
            if not 1 <= int(ref) < num:
                errors.append(f"final_plan step {num}: reference $step{ref} must point to an earlier step")
            elif field and field != outputs[int(ref)]:
                errors.append(f"final_plan step {num}: $step{ref}.{field} does not exist; step {ref} "
                              f"outputs '{outputs[int(ref)]}' (use $step{ref}.{outputs[int(ref)]}[.<subfield>])")
        if step["primitive_function"] == "LOCAL":
            if e["registry_id"] != -1:
                errors.append(f"final_plan step {num} is LOCAL: registry_id must be -1")
            continue
        chosen = choices[num]["chosen_id"]
        if e["registry_id"] != chosen:
            errors.append(f"final_plan step {num}: registry_id {e['registry_id']} != chosen_id {chosen}")
            continue
        if chosen == -1:
            continue
        r = index.entry(chosen)
        req = {p.get("name") for p in _params(r.get("required_parameters"))}
        allowed = req | {p.get("name") for p in _params(r.get("optional_parameters"))}
        names = {a["name"] for a in e["arguments"]}
        if req - names:
            errors.append(f"final_plan step {num}: missing required argument(s) {sorted(req - names)}")
        if step["depends_on"] and not any("$step" in a["value"] for a in e["arguments"]):
            errors.append(f"final_plan step {num}: the step uses output of step(s) {step['depends_on']} "
                          f"but no argument passes it; if no parameter of endpoint {chosen} can carry "
                          f"that data, the endpoint cannot do this step (chosen_id -1)")
        # 28% of endpoints document no parameters; then any names are allowed (flagged in attach()).
        # undocumented_arguments: declared by a human repair when the registry entry lacks
        # the parameter that must carry the data (e.g. 'Add rows' has no rows parameter).
        unknown = names - allowed - set(e.get("undocumented_arguments", []))
        if allowed and unknown:
            errors.append(f"final_plan step {num}: unknown argument(s) {sorted(unknown)} "
                          f"for endpoint {chosen}")
    return errors, ungrounded


def attach(rec, out, pools, index, judge_model, attempts):
    """Merge grounding + final plan into the record."""
    choices = {c["step_num"]: c for c in out["api_choices"]}
    for s in rec["canonical_plan"]:
        if s["step_num"] in choices:
            c = choices[s["step_num"]]
            r = index.entry(c["chosen_id"])
            s["ground_truth_tool"] = {"registry_id": c["chosen_id"], "tool_name": r["tool_name"],
                                      "api_name": r["api_name"]}
            s["acceptable_tools"] = c["acceptable_ids"]
            s["candidate_pool"] = pools[s["step_num"]]
            s["selection_rationale"] = c["rationale"]
    final = []
    for e, s in zip(out["final_plan"], rec["canonical_plan"]):
        item = {"step_num": e["step_num"], "primitive_function": s["primitive_function"],
                "operation": e["operation"], "output": e["output"]}
        if e["registry_id"] != -1:
            r = index.entry(e["registry_id"])
            documented = _params(r.get("required_parameters")) + _params(r.get("optional_parameters"))
            item.update({"registry_id": e["registry_id"], "tool_name": r["tool_name"],
                         "api_name": r["api_name"], "method": r.get("method"), "url": r.get("url"),
                         "arguments": {a["name"]: a["value"] for a in e["arguments"]},
                         "arguments_documented": bool(documented)})
            if e.get("undocumented_arguments"):
                item["undocumented_arguments"] = e["undocumented_arguments"]
        elif e["arguments"]:
            item["inputs"] = {a["name"]: a["value"] for a in e["arguments"]}
        final.append(item)
    rec["final_plan"] = final
    rec["grounding"] = {"status": "complete", "judge_model": judge_model,
                        "candidate_method": CANDIDATE_METHOD, "attempts": attempts}
    return rec


def ground(rec, index, call, judge_model, max_attempts=3):
    """call(contents, system, schema) -> response. Returns (rec, status, errors)."""
    pools = {s["step_num"]: index.candidates(s["canonical_intent"], rec["query"])
             for s in rec["canonical_plan"] if s["requires_api"]}
    prompt, feedback, errors, out = build_prompt(rec, pools, index), "", [], None
    for attempt in range(1, max_attempts + 1):
        try:
            out = json.loads(call(prompt + feedback, SYSTEM_PROMPT, SCHEMA).text)
            fix_refs(out)
            for c in out.get("api_choices", []):  # the chosen endpoint is acceptable by definition
                if c["chosen_id"] != -1 and c["chosen_id"] not in c["acceptable_ids"]:
                    c["acceptable_ids"].insert(0, c["chosen_id"])
            errors, ungrounded = validate_composition(rec, pools, out, index)
        except Exception as e:  # malformed JSON / schema mismatch
            errors, ungrounded = [f"call/parse error: {str(e)[:200]}"], []
        if not errors and (missing := [d for d in out["plan_defects"] if NO_API_RE.search(d)]):
            errors = [f"plan_defects must not report missing APIs ({missing[:2]}): for that step either "
                      f"pick a candidate that can do it (a general search API that takes a free-text "
                      f"query can cover location or topic filtering) or set chosen_id -1"]
        if not errors:
            if out["plan_defects"]:
                rec["grounding"] = {"status": "plan_defect", "plan_defects": out["plan_defects"],
                                    "judge_model": judge_model, "candidate_pools": pools,
                                    "last_output": out}
                return rec, "plan_defect", [f"plan defect: {d}" for d in out["plan_defects"]]
            if ungrounded:
                rec["grounding"] = {"status": "ungrounded", "ungrounded_steps": ungrounded,
                                    "judge_model": judge_model, "candidate_method": CANDIDATE_METHOD,
                                    "candidate_pools": pools}
                return rec, "ungrounded", [f"no suitable API for step(s) {ungrounded}"]
            return attach(rec, out, pools, index, judge_model, attempt), "complete", []
        feedback = ("\n\nYour previous output was rejected for these reasons; fix all of them:\n- "
                    + "\n- ".join(errors))
    rec["grounding"] = {"status": "failed", "judge_model": judge_model, "candidate_pools": pools,
                        "last_output": out if isinstance(out, dict) else None}
    return rec, "failed", errors
