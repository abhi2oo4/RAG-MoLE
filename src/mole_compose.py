"""
mole_compose.py
---------------
RAG-MoLE Composition pass: query + Discovery plan + retrieved candidate APIs -> final executable plan.

For every API step of the plan, the dense retriever (MiniLM over the master registry) supplies the
top-K candidate endpoints, listed A, B, C...  The Composition adapter writes one line per step:

    <n>. [TYPE] <operation> | out <name>                           (LOCAL step)
    <n>. [TYPE] <operation> | out <name> | api <letter> | args {json}   (API step)

The chosen letter is resolved back to a registry endpoint (tool, api, method, url).

Training uses gold plans with the retrieval's own top-K, forcing the gold endpoint into the list when
retrieval misses it (at a random slot, so position does not leak the answer). At inference the list is
exactly what retrieval returned, so a miss cannot be repaired - that is the retrieval ceiling.
"""

import json
import random
import re

import numpy as np

from mole_common import ROOT, SYSTEM, step_line

K = 5
LETTERS = "ABCDEFGHIJ"
COMPOSE_SYSTEM = ("You are the Composition expert of an API task-planning system. Given a query, a plan "
                  "and candidate APIs for each API step, write the final executable plan: one line per "
                  "step, choosing the best candidate letter for each API step and filling its arguments "
                  "(use $stepN.<output> to pass an earlier step's output).")
REG_PATH = f"{ROOT}/data/processed/master_registry.json"
EMB_PATH = f"{ROOT}/data/processed/registry_emb_minilm.npy"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"

_RETRIEVER = None


class Retriever:
    """Dense retrieval over the registry (same model and cache as benchmark_plan_retrieval's MiniLM)."""

    def __init__(self, device=None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reg = json.load(open(REG_PATH))
        self.idx_of = {int(r["registry_id"]): i for i, r in enumerate(self.reg)}
        self.enc = SentenceTransformer(MINILM, device=self.device)
        self.emb = torch.from_numpy(np.load(EMB_PATH).astype(np.float32)).to(self.device)

    def search(self, texts, k=K):
        """Top-k registry ids per text."""
        if not texts:
            return []
        q = self.enc.encode(texts, normalize_embeddings=True, convert_to_tensor=True,
                            show_progress_bar=False).to(self.device).float()
        top = self.torch.topk(q @ self.emb.T, k, dim=1).indices.cpu().tolist()
        return [[int(self.reg[i]["registry_id"]) for i in row] for row in top]

    def record(self, rid):
        return self.reg[self.idx_of[rid]]


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


def cand_text(rec):
    desc = " ".join((rec.get("api_description") or rec.get("tool_description") or "").split())[:140]
    req = [p["name"] + "*" for p in rec.get("required_parameters") or []][:8]
    opt = [p["name"] for p in rec.get("optional_parameters") or []][:6]
    return (f"{rec['tool_name']} / {rec['api_name']} ({rec.get('method', 'GET')}): {desc} "
            f"| params: {', '.join(req + opt) or 'none'}")


def compose_messages(query, plan, cands, retriever):
    """plan: Discovery steps; cands: {step_num: [registry_id, ...]} for the plan's API steps."""
    lines = [f"Query: {query}", "", "Plan:"]
    for s in plan:
        lines.append(f"{s['step_num']}. [{s['primitive_function']}] {step_line(s)}")
        for letter, rid in zip(LETTERS, cands.get(s["step_num"], [])):
            lines.append(f"   {letter}. {cand_text(retriever.record(rid))}")
    lines += ["", "Write the final executable plan."]
    return [{"role": "system", "content": COMPOSE_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def retrieve_for_plan(plan, retriever, k=K):
    api = [s for s in plan if s["primitive_function"] != "LOCAL" and s["canonical_intent"]]
    return {s["step_num"]: ids for s, ids in zip(api, retriever.search([s["canonical_intent"] for s in api], k))}


def force_gold(ids, gold, rng):
    """Make sure the gold endpoint is among the candidates, at a random position when it was missed."""
    ids = list(ids)
    if gold in ids:
        return ids
    ids = ids[:-1]
    ids.insert(rng.randrange(len(ids) + 1), gold)
    return ids


def target_text(final_plan, cands):
    out = []
    for s in final_plan:
        line = f"{s['step_num']}. [{s['primitive_function']}] {s['operation']} | out {s['output']}"
        if s["primitive_function"] != "LOCAL":
            letter = LETTERS[cands[s["step_num"]].index(s["registry_id"])]
            line += f" | api {letter} | args {json.dumps(s.get('arguments') or {}, ensure_ascii=False)}"
        out.append(line)
    return "\n".join(out)


def gold_cands(row, retriever, k=K, seed=0):
    """Candidate lists for a gold plan: retrieval top-k with the gold endpoint forced in."""
    rng = random.Random(f"{seed}-{row['sample_id']}")
    cands = retrieve_for_plan(row["canonical_plan"], retriever, k)
    for s in row["final_plan"]:
        if s["primitive_function"] != "LOCAL":
            cands[s["step_num"]] = force_gold(cands[s["step_num"]], s["registry_id"], rng)
    return cands


def compose_examples(rows):
    """(messages, target) pairs for training the Composition adapter."""
    ret = get_retriever()
    out = []
    for r in rows:
        cands = gold_cands(r, ret)
        out.append((compose_messages(r["query"], r["canonical_plan"], cands, ret), target_text(r["final_plan"], cands)))
    return out


LINE_RE = re.compile(r"^\s*(?P<n>\d+)\.\s*\[(?P<t>[A-Z]+)\]\s*(?P<body>.*)$")
API_RE = re.compile(r"\s\|\s*api\s+(?P<letter>[A-J])\s*\|\s*args\s*(?P<args>\{.*\})\s*$")
OUT_RE = re.compile(r"\s\|\s*out\s+(?P<out>\S*)\s*$")


def parse_composition(text, cands, retriever):
    """Parse generated lines into steps; resolves candidate letters to registry endpoints.
    Returns (steps, ok) where ok means every line parsed and every API step resolved with JSON args."""
    steps, ok = [], True
    for line in text.strip().splitlines():
        m = LINE_RE.match(line)
        if not m:
            ok = False
            continue
        n, label, body = int(m["n"]), m["t"], m["body"]
        step = {"step_num": n, "primitive_function": label}
        am = API_RE.search(body)
        if am:
            body = body[:am.start()]
            ids = cands.get(n, [])
            li = LETTERS.index(am["letter"])
            try:
                step["arguments"] = json.loads(am["args"])
                if not isinstance(step["arguments"], dict):
                    raise ValueError
            except ValueError:
                step["arguments"], ok = {}, False
            if li < len(ids):
                rec = retriever.record(ids[li])
                step.update(registry_id=int(rec["registry_id"]), tool_name=rec["tool_name"], api_name=rec["api_name"],
                            method=rec.get("method"), url=rec.get("url"))
            else:
                ok = False
        elif label != "LOCAL":
            ok = False
        om = OUT_RE.search(body)
        if om:
            body = body[:om.start()]
            step["output"] = om["out"]
        step["operation"] = body.strip()
        steps.append(step)
    return steps, ok and bool(steps)


class Composer:
    """Runs the 'compose' adapter of an already-loaded PeftModel."""

    def __init__(self, model, tok, retriever, adapter="compose"):
        self.model, self.tok, self.ret, self.adapter = model, tok, retriever, adapter

    def compose(self, queries, plans, cands, bs=8, max_new_tokens=1100):
        import torch
        self.model.set_adapter(self.adapter)
        out = []
        for b in range(0, len(queries), bs):
            convs = [compose_messages(q, p, c, self.ret)
                     for q, p, c in zip(queries[b:b + bs], plans[b:b + bs], cands[b:b + bs])]
            texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in convs]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return out
