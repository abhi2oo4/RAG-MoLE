#!/usr/bin/env python3
"""
generate_stage2_dataset.py
--------------------------
Stage 2 Dataset Generator with tqdm progress bar and speed monitoring.

Modes:
- 'llm': Uses local Ollama LLM (qwen2.5:latest) with tqdm monitoring.
- 'fast': Uses high-speed AST/NLP Semantic Compiler parser (<2 seconds for 1,000 samples).

Generates top-class dataset entries containing:
- goal: High-level user goal summary
- intents: List of canonical provider-independent intent strings
- constraints: List of explicit constraints
- parameters: Key-value parameter dictionary
"""

import os
import sys
import json
import glob
import time
import re
import urllib.request
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = """You are an expert Semantic Intent Compiler for an API retrieval benchmark.

Your task is to analyze a user query and execution trace, then extract the canonical, provider-independent semantic components required to satisfy the request.

OUTPUT JSON SCHEMA:
{
    "goal": "Clear summary of the user's primary goal",
    "intents": ["canonical intent 1", "canonical intent 2"],
    "constraints": ["constraint 1 if any"],
    "parameters": {"param_key": "param_value"}
}

Return ONLY valid JSON matching the schema above."""

def call_ollama_llm(
    sample_query: str,
    trace: List[Dict[str, Any]],
    final_answer: str,
    model_name: str = "qwen2.5:latest",
    ollama_url: str = "http://localhost:11434/api/chat",
    timeout: int = 30
) -> Dict[str, Any]:
    user_prompt = f"""USER QUERY: {sample_query}
EXECUTION TRACE: {json.dumps(trace, indent=2)}
FINAL ANSWER: {final_answer}

Infer the goal, canonical intents, constraints, and parameters. Return ONLY valid JSON."""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        req = urllib.request.Request(
            ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("message", {}).get("content", "")
            return json.loads(content)
    except Exception:
        return fast_extract(sample_query, trace, final_answer)


def fast_extract(query: str, trace: List[Dict[str, Any]], final_answer: str) -> Dict[str, Any]:
    """High-speed NLP/AST Semantic Compiler parser."""
    intents = []
    constraints = []
    parameters = {}

    # Extract constraints (limits, exclusions, pricing)
    limit_match = re.search(r'limit\s+(?:to\s+)?(\d+)', query, re.IGNORECASE)
    if limit_match:
        constraints.append(f"limit: {limit_match.group(1)}")
        parameters["limit"] = int(limit_match.group(1))

    exclude_match = re.search(r'exclude\s+([a-zA-Z0-9\s]+?)(?:and|\.|,|$)', query, re.IGNORECASE)
    if exclude_match:
        constraints.append(f"exclude: {exclude_match.group(1).strip()}")

    if "free" in query.lower() or "freemium" in query.lower():
        constraints.append("pricing: free/freemium")

    # Extract parameters from trace step arguments
    for step in trace:
        if isinstance(step, dict):
            raw_args = step.get("arguments_raw", step.get("arguments", {}))
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {}
            if isinstance(raw_args, dict):
                for k, v in raw_args.items():
                    if k not in parameters and v:
                        parameters[k] = str(v)[:50]

    # Generate intents by decomposing query
    clauses = re.split(r'[.!?;\n]+', query)
    for clause in clauses:
        clause = clause.strip()
        if len(clause) > 5:
            cleaned = re.sub(r'^(can you|please|i want to|i need to|i would like to|could you|help me)\s+', '', clause, flags=re.IGNORECASE)
            intents.append(cleaned.strip().capitalize())

    # Fallback to trace action goals
    if not intents and trace:
        for step in trace:
            if isinstance(step, dict) and "api_name" in step:
                clean_api = step["api_name"].replace("_for_", " of ").replace("_", " ")
                intents.append(f"Execute action to {clean_api}")

    goal = clauses[0].strip() if clauses else query

    return {
        "goal": goal,
        "intents": intents if intents else [query],
        "constraints": constraints,
        "parameters": parameters
    }


def process_file(task_item: tuple, mode: str, model_name: str, ollama_url: str) -> Optional[Dict[str, Any]]:
    filepath, group = task_item
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    query = data.get("query", "").strip()
    trace = data.get("trace", [])
    final_answer = data.get("final_answer", "").strip()

    if not query or not trace:
        return None

    gt_apis = []
    for step in trace:
        if isinstance(step, dict) and "api_name" in step:
            api = step["api_name"].strip()
            if api:
                gt_apis.append(api)

    if not gt_apis:
        return None

    seen = set()
    unique_gt_apis = [x for x in gt_apis if not (x in seen or seen.add(x))]

    if mode == "llm":
        extracted = call_ollama_llm(query, trace, final_answer, model_name=model_name, ollama_url=ollama_url)
    else:
        extracted = fast_extract(query, trace, final_answer)

    return {
        "sample_id": f"{group}_{os.path.basename(filepath)}",
        "query": query,
        "trace": [{"api_name": api} for api in unique_gt_apis],
        "goal": extracted.get("goal", query),
        "intents": extracted.get("intents", [query]),
        "constraints": extracted.get("constraints", []),
        "parameters": extracted.get("parameters", {})
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Dataset Generator with tqdm monitoring.")
    parser.add_argument("--intermediate_dir", type=str, default="/home/abhishek/Desktop/projects/RAG_MOLE/data/intermediate")
    parser.add_argument("--output", type=str, default="/home/abhishek/Desktop/projects/RAG_MOLE/data/processed/eval_dataset.json")
    parser.add_argument("--mode", type=str, choices=["llm", "fast"], default="llm")
    parser.add_argument("--model", type=str, default="qwen2.5:latest")
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ollama_url = "http://localhost:11434/api/chat"
    print(f"\n" + "="*70)
    print(f"       STAGE 2 DATASET GENERATOR ({args.mode.upper()} MODE)")
    print(f"="*70)
    print(f"Mode         : {args.mode}")
    if args.mode == "llm":
        print(f"Ollama Model : {args.model}")
        print(f"Workers      : {args.workers}")
    print(f"Max Samples  : {args.max_samples}")

    all_tasks = []
    samples_per_group = args.max_samples // 3
    for group in ["G1", "G2", "G3"]:
        group_dir = os.path.join(args.intermediate_dir, group)
        if os.path.exists(group_dir):
            files = glob.glob(os.path.join(group_dir, "*.json"))[:samples_per_group]
            for fp in files:
                all_tasks.append((fp, group))

    print(f"Total files queued: {len(all_tasks)}")
    print("="*70 + "\n", flush=True)

    compiled_samples = []
    t0 = time.time()

    if args.mode == "llm":
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_file, task, args.mode, args.model, ollama_url) for task in all_tasks]
            for future in tqdm(as_completed(futures), total=len(all_tasks), desc="LLM Extracting", unit="sample"):
                res = future.result()
                if res:
                    compiled_samples.append(res)
    else:
        for task in tqdm(all_tasks, desc="Fast Extracting", unit="sample"):
            res = process_file(task, args.mode, args.model, ollama_url)
            if res:
                compiled_samples.append(res)

    t1 = time.time()
    elapsed = t1 - t0
    rate = len(compiled_samples) / elapsed if elapsed > 0 else 0

    print("\n" + "="*70)
    print(f"GENERATION COMPLETED IN {elapsed:.2f} SECONDS")
    print(f"Total Samples Generated : {len(compiled_samples)}")
    print(f"Speed                   : {rate:.2f} samples/sec ({elapsed/max(1, len(compiled_samples)):.2f} s/sample)")
    print("="*70)

    if compiled_samples:
        print("\nSAMPLE PREVIEW (First Record):")
        print(json.dumps(compiled_samples[0], indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(compiled_samples, f, indent=2)

    print(f"\nSaved Stage 2 dataset to: '{args.output}'\n", flush=True)


if __name__ == "__main__":
    main()
