#!/usr/bin/env python3
"""
mole_discovery.py
-----------------
RAG-MoLE Discovery pass: query -> RWPL plan, written one step at a time.

At every step the router adapter names the next step's type (READ/WRITE/PROCESS/LOCAL/END) and
the expert adapter for that type writes the step. All adapters share one base model;
samples are processed in batches, grouped by expert at each step.

  python3 src/mole_discovery.py --split test --output data/results/mole_plans_test.jsonl
  python3 src/mole_discovery.py --expert_adapter all ...   # generalist baseline
"""

import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer

import re

from mole_common import (ADAPTER_DIR, BASE_MODEL, EXPERTS, LABELS, MAX_STEPS, expert_messages,
                         load_base_model, load_split, parse_step, router_messages, single_messages)


class Discovery:
    def __init__(self, adapter_dir=ADAPTER_DIR, expert_adapter=None, four_bit=True):
        """expert_adapter=None uses the per-type experts; a name (e.g. "all") uses that adapter for every step."""
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
        base = load_base_model(four_bit)
        self.model = PeftModel.from_pretrained(base, os.path.join(adapter_dir, "router"), adapter_name="router")
        names = [expert_adapter] if expert_adapter else EXPERTS
        for name in names:
            self.model.load_adapter(os.path.join(adapter_dir, name), adapter_name=name)
        self.expert_for = {l: expert_adapter or l.lower() for l in LABELS[:-1]}
        self.model.eval()
        self.label_ids = [self.tok(l, add_special_tokens=False).input_ids[0] for l in LABELS]

    def _encode(self, convs):
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in convs]
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

    @torch.no_grad()
    def route(self, items, bs):
        self.model.set_adapter("router")
        out = []
        for b in range(0, len(items), bs):
            enc = self._encode([router_messages(q, p) for q, p in items[b:b + bs]])
            pos = (enc.attention_mask.cumsum(dim=1) - 1).clamp(min=0)
            logits = self.model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, self.label_ids]
            out += [LABELS[i] for i in logits.argmax(dim=1).tolist()]
        return out

    @torch.no_grad()
    def write(self, items, label, bs):
        self.model.set_adapter(self.expert_for[label])
        out = []
        for b in range(0, len(items), bs):
            enc = self._encode([expert_messages(q, p, label) for q, p in items[b:b + bs]])
            gen = self.model.generate(**enc, max_new_tokens=96, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return out

    def plan(self, queries, bs=16, min_steps=1):
        plans = [[] for _ in queries]
        active = list(range(len(queries)))
        while active:
            labels = self.route([(queries[i], plans[i]) for i in active], bs)
            nxt = []
            for i, lab in zip(active, labels):
                if lab == "END" and len(plans[i]) >= min_steps:
                    continue
                nxt.append((i, lab if lab != "END" else "LOCAL"))
            for lab in LABELS[:-1]:
                group = [i for i, l in nxt if l == lab]
                if not group:
                    continue
                texts = self.write([(queries[i], plans[i]) for i in group], lab, bs)
                for i, t in zip(group, texts):
                    plans[i].append(parse_step(t, len(plans[i]) + 1, lab))
            active = [i for i, _ in nxt if len(plans[i]) < MAX_STEPS]
        return plans


class SingleDiscovery:
    """Ablation: one LoRA writes "[TYPE] step" (or END) for every step - no router, no per-type experts."""

    TYPE_RE = re.compile(r"^\s*\[(READ|WRITE|PROCESS|LOCAL)\]\s*(.*)$", re.S)

    def __init__(self, adapter_dir=ADAPTER_DIR, four_bit=True, adapter="single"):
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
        base = load_base_model(four_bit)
        self.model = PeftModel.from_pretrained(base, os.path.join(adapter_dir, adapter), adapter_name=adapter)
        self.adapter = adapter
        self.model.eval()

    @torch.no_grad()
    def _write(self, items, bs):
        self.model.set_adapter(self.adapter)
        out = []
        for b in range(0, len(items), bs):
            texts = [self.tok.apply_chat_template(single_messages(q, p), tokenize=False, add_generation_prompt=True)
                     for q, p in items[b:b + bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = self.model.generate(**enc, max_new_tokens=110, do_sample=False, pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return out

    def plan(self, queries, bs=16, min_steps=1):
        plans = [[] for _ in queries]
        active = list(range(len(queries)))
        while active:
            texts = self._write([(queries[i], plans[i]) for i in active], bs)
            nxt = []
            for i, t in zip(active, texts):
                t = t.strip()
                if t.startswith("END") and len(plans[i]) >= min_steps:
                    continue
                m = self.TYPE_RE.match(t.splitlines()[0] if t else "")
                label, body = (m[1], m[2]) if m else ("LOCAL", t)
                plans[i].append(parse_step(body, len(plans[i]) + 1, label))
                nxt.append(i)
            active = [i for i in nxt if len(plans[i]) < MAX_STEPS]
        return plans


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True)
    ap.add_argument("--adapter_dir", default=ADAPTER_DIR)
    ap.add_argument("--expert_adapter", default=None)
    ap.add_argument("--single", action="store_true", help="single-LoRA Discovery ablation (adapter 'single')")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--query", default=None, help="plan a single query and print it")
    ap.add_argument("--no_4bit", action="store_true", help="bf16 base model instead of 4-bit")
    args = ap.parse_args()

    disc = (SingleDiscovery(args.adapter_dir, four_bit=not args.no_4bit) if args.single
            else Discovery(args.adapter_dir, args.expert_adapter, four_bit=not args.no_4bit))
    if args.query:
        for s in disc.plan([args.query])[0]:
            print(f"{s['step_num']:>2}. [{s['primitive_function']}] {s['canonical_intent']}  "
                  f"params={s['parameters']} deps={s['depends_on']}")
        return
    rows = load_split(args.split)[:args.limit or None]
    t0 = time.time()
    plans = disc.plan([r["query"] for r in rows], bs=args.batch_size)
    with open(args.output, "w") as f:
        for r, p in zip(rows, plans):
            f.write(json.dumps({"sample_id": r["sample_id"], "query": r["query"], "pred_plan": p}) + "\n")
    print(f"{len(rows)} plans in {(time.time() - t0) / 60:.1f} min -> {args.output}")


if __name__ == "__main__":
    main()
