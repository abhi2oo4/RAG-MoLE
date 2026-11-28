#!/usr/bin/env python3
"""
train_mole.py
-------------
QLoRA training of one RAG-MoLE Discovery adapter on the RWPL plan dataset (v3).

  --adapter read|write|process|local   expert: writes a step of its own type
  --adapter router                     names the next step's type, or END
  --adapter all                        one generalist adapter over every step (baseline)
  --adapter single                     ablation: ONE adapter does the router's and the experts' job
                                       (writes "[TYPE] step" or END) - the single-LoRA Discovery
  --adapter compose                    Composition adapter: query + plan + retrieved candidate APIs
                                       -> final executable plan (mole_compose.py)

Examples are built by teacher forcing on gold plans (mole_common.examples); the loss covers only
the target line. Adapters are saved to models/mole/<adapter>/.
"""

import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from mole_common import ADAPTER_DIR, BASE_MODEL, LABELS, examples, load_base_model, load_split


def encode(tok, pairs, max_len):
    out = []
    for messages, target in pairs:
        prompt = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                     add_special_tokens=False).input_ids
        tgt = tok(target + "<|im_end|>", add_special_tokens=False).input_ids
        prompt = prompt[-(max_len - len(tgt)):]
        out.append((prompt + tgt, [-100] * len(prompt) + tgt))
    return out


def batches(data, bs, seed=None):
    """Length-bucketed batches: sort within chunks of 50 batches to cut padding.
    A seed shuffles reproducibly, so a resumed run sees the same batch order."""
    rng = random.Random(seed)
    idx = list(range(len(data)))
    if seed is not None:
        rng.shuffle(idx)
    chunk = bs * 50
    out = []
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: len(data[i][0]))
        out += [part[b:b + bs] for b in range(0, len(part), bs)]
    if seed is not None:
        rng.shuffle(out)
    return out


def collate(data, ids, pad_id):
    """Left-padded batch, so every target sits at the end and only the tail needs logits."""
    width = max(len(data[i][0]) for i in ids)
    inp = torch.full((len(ids), width), pad_id)
    lab = torch.full((len(ids), width), -100)
    att = torch.zeros((len(ids), width), dtype=torch.long)
    for row, i in enumerate(ids):
        x, y = data[i]
        inp[row, width - len(x):], lab[row, width - len(y):], att[row, width - len(x):] = \
            torch.tensor(x), torch.tensor(y), 1
    keep = int((lab != -100).sum(dim=1).max()) + 1
    return inp.cuda(), att.cuda(), lab[:, -keep:].cuda(), keep


def tail_logits(model, inp, att, keep):
    pos = (att.cumsum(dim=1) - 1).clamp(min=0)
    return model(input_ids=inp, attention_mask=att, position_ids=pos, logits_to_keep=keep).logits


def target_loss(logits, lab, reduction="mean"):
    shift, gold = logits[:, :-1].float(), lab[:, 1:]
    mask = gold != -100
    return torch.nn.functional.cross_entropy(shift[mask], gold[mask], reduction=reduction), shift, gold, mask


@torch.no_grad()
def evaluate(model, data, bs, pad_id, label_ids=None):
    """Mean target loss; for the router also first-token label accuracy."""
    model.eval()
    loss_sum, tok_sum, correct = 0.0, 0, 0
    for ids in batches(data, bs):
        inp, att, lab, keep = collate(data, ids, pad_id)
        loss, shift, gold, mask = target_loss(tail_logits(model, inp, att, keep), lab, "sum")
        loss_sum += loss.item()
        tok_sum += int(mask.sum())
        if label_ids is not None:
            first = mask.float().argmax(dim=1)
            rows = torch.arange(len(ids), device=inp.device)
            pred = shift[rows, first][:, label_ids].argmax(dim=1)
            correct += int((torch.tensor(label_ids, device=inp.device)[pred] == gold[rows, first]).sum())
    model.train()
    res = {"val_loss": round(loss_sum / tok_sum, 4)}
    if label_ids is not None:
        res["val_label_acc"] = round(100 * correct / len(data), 2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, choices=["read", "write", "process", "local", "router", "all", "single", "compose"])
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_val", type=int, default=600, help="val examples scored per evaluation")
    ap.add_argument("--max_train", type=int, default=0, help="cap on train examples (0 = all), for smoke tests")
    ap.add_argument("--out_dir", default=ADAPTER_DIR)
    ap.add_argument("--ckpt_every", type=int, default=100, help="optimizer steps between checkpoints")
    ap.add_argument("--resume", action="store_true", help="continue from <out_dir>/<adapter>/checkpoint")
    ap.add_argument("--no_4bit", action="store_true", help="bf16 base model instead of 4-bit (needs ~16 GB+)")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="faster, but needs far more memory")
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    train = encode(tok, examples(load_split("train"), args.adapter), args.max_len)
    val = encode(tok, examples(load_split("val"), args.adapter), args.max_len)
    random.shuffle(val)
    val = val[:args.max_val]
    if args.max_train:
        random.shuffle(train)
        train = train[:args.max_train]
    label_ids = None
    if args.adapter == "router":
        label_ids = [tok(l, add_special_tokens=False).input_ids[0] for l in LABELS]
        assert len(set(label_ids)) == len(LABELS), "router labels must differ in their first token"
    print(f"[{args.adapter}] train {len(train)} examples, val {len(val)}; "
          f"mean length {sum(len(x) for x, _ in train) / len(train):.0f} tokens")

    model = load_base_model(four_bit=not args.no_4bit)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not args.no_grad_ckpt)
    elif not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights train in fp32 whatever the base precision
    model.print_trainable_parameters()

    steps_per_epoch = len(batches(train, args.batch_size)) // args.grad_accum
    total = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)
    log = {"args": vars(args), "n_train": len(train), "n_val": len(val), "history": []}
    out = os.path.join(args.out_dir, args.adapter)
    ckpt = os.path.join(out, "checkpoint")

    def save_checkpoint(epoch, batch):
        model.save_pretrained(ckpt)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "epoch": epoch, "batch": batch, "log": log}, os.path.join(ckpt, "state.pt"))

    step, epoch, skip = 0, 0, 0
    if args.resume and os.path.exists(os.path.join(ckpt, "state.pt")):
        set_peft_model_state_dict(model, load_file(os.path.join(ckpt, "adapter_model.safetensors")))
        state = torch.load(os.path.join(ckpt, "state.pt"), weights_only=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step, epoch, skip, log = state["step"], state["epoch"], state["batch"], state["log"]
        print(f"resumed at step {step}/{total} (epoch {epoch}, batch {skip})")
    else:
        print("before training:", evaluate(model, val, args.batch_size, tok.pad_token_id, label_ids))

    model.train()
    t0, start, run_loss = time.time(), step, []
    while step < total:
        for b, ids in enumerate(batches(train, args.batch_size, seed=epoch)):
            if b < skip:
                continue
            inp, att, lab, keep = collate(train, ids, tok.pad_token_id)
            loss = target_loss(tail_logits(model, inp, att, keep), lab)[0] / args.grad_accum
            loss.backward()
            run_loss.append(loss.item() * args.grad_accum)
            if (b + 1) % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                print(f"step {step}/{total}  loss {sum(run_loss) / len(run_loss):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {el / 60:.1f} min elapsed, "
                      f"~{el / (step - start) * (total - step) / 60:.0f} min left", flush=True)
                log["history"].append({"step": step, "train_loss": round(sum(run_loss) / len(run_loss), 4)})
                run_loss = []
            if step % args.ckpt_every == 0:
                save_checkpoint(epoch, b + 1)
            if step >= total:
                break
        skip = 0
        epoch += 1
        res = evaluate(model, val, args.batch_size, tok.pad_token_id, label_ids)
        print(f"epoch {epoch}: {res}", flush=True)
        log["history"].append({"epoch": epoch, "step": step, **res})

    model.save_pretrained(out)
    log["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=2)
    print(f"saved {out} ({log['minutes']} min)")


if __name__ == "__main__":
    main()
