"""Regenerates the figures in docs/figures from the files in data/results (fig_train also needs models/mole/*/train_log.json)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = os.environ.get("RAG_MOLE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
OUT = f"{P}/docs/figures"
R = f"{P}/data/results"
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight"})
NAVY, BLUE, GREY, LIGHT, ORANGE = "#1f3a5f", "#4f81bd", "#7f7f7f", "#dbe5f1", "#d9822b"
J = lambda f: json.load(open(f"{R}/{f}"))


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=10.5, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.3, color=NAVY))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 3.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.8); ax.axis("off")
    box(ax, 0.05, 1.5, 1.5, 1.0, "User\nquery", fc="white")
    box(ax, 2.1, 1.3, 2.2, 1.4, "Discovery\n(typed step plan)", fc=LIGHT, bold=True)
    box(ax, 4.95, 1.3, 2.0, 1.4, "Per-step\nretrieval", fc="#eeeeee", bold=True)
    box(ax, 7.5, 1.3, 2.4, 1.4, "Composition\n(final plan)", fc=LIGHT, bold=True)
    for a, b in ((1.55, 2.1), (4.3, 4.95), (6.95, 7.5)):
        arrow(ax, a, 2.0, b, 2.0)
    ax.text(3.2, 0.95, "[READ] fetch air cargo rates\n[PROCESS] convert currency\n[LOCAL] compare costs\n[WRITE] send report email",
            ha="center", va="top", fontsize=8.6, family="DejaVu Sans Mono")
    ax.text(5.95, 0.95, "top-5 endpoints for\neach API step from a\n47,075-endpoint registry", ha="center", va="top", fontsize=9)
    ax.text(8.7, 0.95, "tool + API + arguments\nfor every step, with\n$step references", ha="center", va="top", fontsize=9)
    ax.text(5, 3.45, "Plan  →  Retrieve  →  Compose", ha="center", fontsize=12, fontweight="bold", color=NAVY)
    fig.savefig(f"{OUT}/fig_pipeline.png"); plt.close(fig)


def fig_arch():
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})
    for ax in axs:
        ax.set_ylim(0, 6.2); ax.axis("off")
    ax = axs[0]; ax.set_xlim(0, 8)
    ax.set_title("(a) MoLE Discovery", fontsize=11.5, fontweight="bold", color=NAVY)
    box(ax, 1.9, 5.0, 4.2, 0.8, "Query + plan so far", fc="white")
    arrow(ax, 4.0, 5.0, 4.0, 4.55)
    box(ax, 2.5, 3.75, 3.0, 0.8, "Router (LoRA)  \u2192  step type", fc=ORANGE + "55", fs=10)
    arrow(ax, 4.0, 3.75, 4.0, 3.3)
    for i, n in enumerate(["READ", "WRITE", "PROCESS", "LOCAL"]):
        box(ax, 0.1 + i * 1.95, 2.25, 1.8, 1.0, f"{n}\nexpert", fs=9.5)
    ax.text(4.25, 3.52, "selects one expert", ha="left", fontsize=8.5, color=GREY, va="center")
    arrow(ax, 4.0, 2.25, 4.0, 1.85)
    box(ax, 2.5, 1.05, 3.0, 0.8, "next step", fc="white")
    box(ax, 0.1, 0.05, 7.8, 0.6, "Shared base model: Qwen2.5-3B-Instruct (five adapters of 114 MB each)", fc="#eeeeee", fs=9)
    ax = axs[1]; ax.set_xlim(0, 5)
    ax.set_title("(b) Single-LoRA Discovery", fontsize=11.5, fontweight="bold", color=NAVY)
    box(ax, 0.4, 5.0, 4.2, 0.8, "Query + plan so far", fc="white")
    arrow(ax, 2.5, 5.0, 2.5, 4.55)
    box(ax, 0.4, 3.2, 4.2, 1.35, "One LoRA adapter\nwrites \"[TYPE] step\" or END", fc=LIGHT, bold=True, fs=10)
    box(ax, 1.0, 1.05, 3.0, 0.8, "next step", fc="white")
    arrow(ax, 2.5, 3.2, 2.5, 1.85)
    box(ax, 0.1, 0.05, 4.8, 0.6, "Shared base model (one adapter of 114 MB)", fc="#eeeeee", fs=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_arch.png"); plt.close(fig)


def fig_data():
    fig, ax = plt.subplots(figsize=(9.4, 3.3)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
    steps = [("1. Generate", "LLMs write a\nplain-language query and a\n10\u201315 step typed plan"),
             ("2. Validate", "structure, intent form,\nno vendor names, no\nleaked values; retry"),
             ("3. Ground", "candidate pool (BGE-M3 +\nTF-IDF) per API step;\nLLM picks best + acceptable"),
             ("4. Check", "endpoint in pool, required\narguments, $step links;\nhand review of skeletons"),
             ("5. Split", "near-duplicate groups\nkept together:\n1600 / 195 / 193 plans")]
    w, g = 1.7, 0.25
    for i, (t, d) in enumerate(steps):
        x = 0.25 + i * (w + g)
        box(ax, x, 1.55, w, 0.75, t, fc=LIGHT, bold=True, fs=11)
        ax.text(x + w / 2, 1.35, d, ha="center", va="top", fontsize=8.8)
        if i < len(steps) - 1:
            arrow(ax, x + w, 1.92, x + w + g, 1.92)
    box(ax, 3.2, 2.85, 3.6, 0.6, "Master registry: 47,075 ToolBench endpoints", fc="#eeeeee", fs=10)
    arrow(ax, 5.0, 2.85, 5.0, 2.32)
    fig.savefig(f"{OUT}/fig_data.png"); plt.close(fig)


def fig_train():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for a, c, lab in (("single", NAVY, "Single Discovery LoRA"), ("router", ORANGE, "Router"), ("compose", "#2e8b57", "Composition LoRA")):
        h = [x for x in json.load(open(f"{P}/models/mole/{a}/train_log.json"))["history"] if "train_loss" in x]
        tot = h[-1]["step"]
        ax.plot([x["step"] / tot for x in h], [x["train_loss"] for x in h], color=c, lw=2, label=lab)
    ax.set_xlabel("Fraction of training completed"); ax.set_ylabel("Training loss"); ax.set_yscale("log")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(f"{OUT}/fig_train.png"); plt.close(fig)


def fig_latency():
    m, s = J("pipeline_mole_bench.json")["online_bs1"], J("pipeline_single_bench.json")["online_bs1"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (lab, d) in enumerate((("MoLE + Composition", m), ("Single LoRA + Composition", s))):
        d0, c0 = d["discovery"]["mean_s"], d["composition"]["mean_s"]
        ax.barh(i, d0, color=NAVY, label="Discovery" if i == 0 else None)
        ax.barh(i, c0, left=d0, color=BLUE, label="Composition" if i == 0 else None)
        ax.text(d0 / 2, i, f"{d0:.1f}", ha="center", va="center", color="white", fontsize=10)
        ax.text(d0 + c0 / 2, i, f"{c0:.1f}", ha="center", va="center", color="white", fontsize=10)
        ax.text(d0 + c0 + .6, i, f"total {d['total']['mean_s']:.1f} s", va="center", fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["MoLE + Composition", "Single LoRA + Composition"])
    ax.set_xlabel("Mean latency per query at batch size 1 (seconds)"); ax.set_xlim(0, 50); ax.invert_yaxis()
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2); fig.savefig(f"{OUT}/fig_latency.png"); plt.close(fig)


def fig_retrieval():
    rows = [("Raw query", 2.18, GREY)]
    if os.path.exists(f"{R}/base_onepass_discovery_minilm.json"):
        rows.append(("Base model,\none pass", J("base_onepass_discovery_minilm.json")["retrieval"]["mole"]["Hit@5"], GREY))
    if os.path.exists(f"{R}/base_twopass_discovery_minilm.json"):
        rows.append(("Base model,\ntwo pass", J("base_twopass_discovery_minilm.json")["retrieval"]["mole"]["Hit@5"], GREY))
    rows += [("MoLE\nDiscovery", J("pipeline_mole_discovery_minilm.json")["retrieval"]["mole"]["Hit@5"], NAVY),
             ("Single-LoRA\nDiscovery", J("pipeline_single_discovery_minilm.json")["retrieval"]["mole"]["Hit@5"], BLUE),
             ("Gold steps\n(oracle)", J("pipeline_mole_discovery_minilm.json")["retrieval"]["gold_steps"]["Hit@5"], ORANGE)]
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    ax.bar(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows], width=.62)
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 1, f"{r[1]:.1f}", ha="center", fontsize=10.5)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylabel("Hit@5 (%)"); ax.set_ylim(0, 65); ax.grid(axis="y", alpha=.25)
    fig.savefig(f"{OUT}/fig_retrieval.png"); plt.close(fig)


def fig_compose():
    sets = [("Oracle inputs", J("pipeline_oracle_compose_eval.json"), ORANGE), ("MoLE pipeline", J("pipeline_mole_compose_eval.json"), NAVY),
            ("Single pipeline", J("pipeline_single_compose_eval.json"), BLUE)]
    if os.path.exists(f"{R}/base_twopass_compose_eval.json"):
        sets.append(("Base two-pass", J("base_twopass_compose_eval.json"), GREY))
    keys = [("api_recall", "Gold API steps\nrecovered"), ("arg_name_f1", "Argument-name\nF1"), ("arg_value_exact", "Argument values\nexact")]
    fig, ax = plt.subplots(figsize=(7.8, 3.7)); w = .8 / len(sets)
    for j, (lab, d, c) in enumerate(sets):
        xs = [i + (j - (len(sets) - 1) / 2) * w for i in range(len(keys))]
        ax.bar(xs, [d[k] for k, _ in keys], w * .92, color=c, label=lab)
        for x, (k, _) in zip(xs, keys):
            ax.text(x, d[k] + 1, f"{d[k]:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys]); ax.set_ylabel("%"); ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, ncol=len(sets), loc="upper center", fontsize=9)
    fig.savefig(f"{OUT}/fig_compose.png"); plt.close(fig)


def fig_errors():
    B = J("pipeline_error_breakdown.json")["mole"]
    types = ["READ", "WRITE", "PROCESS", "ALL"]; labs = ["READ", "WRITE", "PROCESS", "All API steps"]
    fig, ax = plt.subplots(figsize=(7.4, 3.5)); w = .36
    for j, (key, lab, c) in enumerate((("in_cands", "Gold endpoint in a candidate list", LIGHT), ("chosen", "Recovered by Composition", NAVY))):
        xs = [i + (j - .5) * w for i in range(4)]
        vals = [100 * B[t][key] / B[t]["n"] for t in types]
        ax.bar(xs, vals, w * .92, color=c, ec=NAVY, lw=.8, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylabel("% of gold API steps"); ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=9)
    fig.savefig(f"{OUT}/fig_errors.png"); plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["pipeline", "data", "arch", "train", "latency", "retrieval", "compose", "errors"]
    for w in which:
        globals()["fig_" + w](); print("made", w)
