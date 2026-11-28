"""
mole_common.py
--------------
Prompt format shared by train_mole.py and mole_discovery.py, so training and inference agree.

The Discovery pass writes a plan one step at a time:
  router  - reads the query and the plan so far, names the next step's type (or END)
  expert  - the LoRA for that type (read/write/process/local) writes the step itself

A step is written as one line:  <intent> | params <json> | deps <json list>
"""

import json
import os
import re

# Override on another machine, e.g. RAG_MOLE_ROOT=~/RAG_MOLE MOLE_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
ROOT = os.environ.get("RAG_MOLE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_MODEL = os.environ.get("MOLE_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_DIR = f"{ROOT}/models/mole"
SPLITS_DIR = f"{ROOT}/data/processed/plan_v3"

EXPERTS = ["read", "write", "process", "local"]
LABELS = ["READ", "WRITE", "PROCESS", "LOCAL", "END"]
MAX_STEPS = 15

SYSTEM = ("You are the Discovery planner of an API task-planning system. Plans use four step "
          "types: READ (API call that retrieves data), WRITE (API call that changes an external "
          "system), PROCESS (API call that computes or transforms input), LOCAL (no API call). "
          "Intents are 4-8 words and never name a vendor.")

STEP_RE = re.compile(r"^(?P<intent>.*?)\s*\|\s*params\s*(?P<params>\{.*\})\s*\|\s*deps\s*(?P<deps>\[.*\])\s*$")


def step_line(step):
    return (f"{step['canonical_intent']} | params {json.dumps(step.get('parameters') or {}, ensure_ascii=False)}"
            f" | deps {json.dumps(step.get('depends_on') or [])}")


def context(query, prefix):
    lines = [f"Query: {query}", "", "Plan so far:"]
    lines += [f"{s['step_num']}. [{s['primitive_function']}] {step_line(s)}" for s in prefix] or ["(empty)"]
    return "\n".join(lines)


def router_messages(query, prefix):
    user = context(query, prefix) + f"\n\nWhat type is step {len(prefix) + 1}? Answer READ, WRITE, PROCESS, LOCAL or END."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def single_messages(query, prefix):
    """Prompt for the single-LoRA Discovery baseline: one adapter names the step type and writes the step."""
    user = (context(query, prefix) + f"\n\nWrite step {len(prefix) + 1}: its type in brackets, then the step, "
            "or END if the plan is complete.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def expert_messages(query, prefix, label):
    user = context(query, prefix) + f"\n\nWrite step {len(prefix) + 1} ({label})."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_step(text, step_num, label):
    """Inverse of step_line; tolerates a malformed tail by keeping the intent only."""
    text = text.strip().splitlines()[0] if text.strip() else ""
    m = STEP_RE.match(text)
    params, deps = {}, []
    if m:
        intent = m["intent"].strip()
        try:
            params = json.loads(m["params"])
            deps = [int(d) for d in json.loads(m["deps"]) if int(d) < step_num]
        except (ValueError, TypeError):
            pass
    else:
        intent = text.split("|")[0].strip()
    return {"step_num": step_num, "canonical_intent": intent, "primitive_function": label,
            "parameters": params if isinstance(params, dict) else {}, "depends_on": deps,
            "requires_api": label != "LOCAL"}


def load_base_model(four_bit=True):
    """The shared base model: 4-bit NF4 (fits 8 GB) or plain bf16 (no bitsandbytes, 24 GB GPUs)."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    quant = None
    if four_bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0},
                                                quantization_config=quant)


def load_split(name):
    return [json.loads(l) for l in open(f"{SPLITS_DIR}/{name}.jsonl")]


def examples(rows, adapter):
    """(messages, target) pairs for one adapter, built from gold plans with teacher forcing."""
    if adapter == "compose":
        from mole_compose import compose_examples
        return compose_examples(rows)
    out = []
    for r in rows:
        plan = r["canonical_plan"]
        for i, s in enumerate(plan):
            if adapter == "single":
                out.append((single_messages(r["query"], plan[:i]), f"[{s['primitive_function']}] {step_line(s)}"))
            elif adapter == "router":
                out.append((router_messages(r["query"], plan[:i]), s["primitive_function"]))
            elif adapter in ("all", s["primitive_function"].lower()):
                out.append((expert_messages(r["query"], plan[:i], s["primitive_function"]), step_line(s)))
        if adapter == "router":
            out.append((router_messages(r["query"], plan), "END"))
        elif adapter == "single":
            out.append((single_messages(r["query"], plan), "END"))
    return out
