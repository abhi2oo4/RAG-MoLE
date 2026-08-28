#!/usr/bin/env python3
"""
build_eval_dataset.py
---------------------
Compiles eval_dataset.json from raw intermediate trace files (G1, G2, G3).
Extracts ground truth APIs from traces and generates semantic intents for each sample query.
"""

import os
import json
import glob
import re
import argparse
from typing import List, Dict, Any

def extract_intents_from_query(query: str, trace: List[Dict[str, Any]]) -> List[str]:
    """
    Generate semantic intents for a given user query and execution trace.
    Splits multi-clause queries and converts trace actions into semantic goals.
    """
    intents = []
    
    # 1. Break query into sentences / main sub-goals
    raw_clauses = re.split(r'[.!?;\n]+', query)
    for clause in raw_clauses:
        clause = clause.strip()
        if len(clause) > 5:
            # Clean leading filler phrases like 'Can you', 'I want to', 'Please'
            cleaned_clause = re.sub(r'^(can you|please|i want to|i need to|i would like to|could you)\s+', '', clause, flags=re.IGNORECASE)
            intents.append(cleaned_clause.strip().capitalize())
            
    # 2. Derive domain intents from ground truth trace steps if query breakdown was minimal
    if len(intents) <= 1 and trace:
        for step in trace:
            if isinstance(step, dict) and 'api_name' in step:
                api_raw = step['api_name']
                # Convert snake_case api_name like 'search_zip_codes_for_mexico_zip_codes' to natural language
                clean_api = api_raw.replace('_for_', ' of ').replace('_', ' ')
                intent_desc = f"Execute action to {clean_api}"
                if intent_desc not in intents:
                    intents.append(intent_desc)

    # Fallback to query itself if empty
    if not intents:
        intents = [query.strip()]
        
    return intents

def build_dataset(
    intermediate_dir: str,
    master_registry_path: str,
    output_path: str,
    max_samples_per_group: int = 200
) -> None:
    print(f"Loading master registry from {master_registry_path}...")
    with open(master_registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        
    print(f"Loaded {len(registry)} APIs from registry.")

    eval_samples = []
    total_found = 0

    groups = ["G1", "G2", "G3"]
    for group in groups:
        group_dir = os.path.join(intermediate_dir, group)
        if not os.path.exists(group_dir):
            print(f"Warning: Group directory {group_dir} not found. Skipping.")
            continue

        json_files = glob.glob(os.path.join(group_dir, "*.json"))
        print(f"Found {len(json_files)} files in {group}. Processing up to {max_samples_per_group} samples...")

        group_count = 0
        for filepath in json_files:
            if group_count >= max_samples_per_group:
                break
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                continue

            query = data.get("query", "").strip()
            trace = data.get("trace", [])

            if not query or not trace:
                continue

            # Extract ground truth API names from trace
            gt_apis = []
            for step in trace:
                if isinstance(step, dict) and "api_name" in step:
                    api_name = step["api_name"].strip()
                    if api_name:
                        gt_apis.append(api_name)

            if not gt_apis:
                continue

            # Deduplicate ground truth APIs preserving order
            seen = set()
            unique_gt_apis = [x for x in gt_apis if not (x in seen or seen.add(x))]

            # Extract semantic intents
            intents = extract_intents_from_query(query, trace)

            eval_sample = {
                "sample_id": f"{group}_{os.path.basename(filepath)}",
                "query": query,
                "trace": [{"api_name": api} for api in unique_gt_apis],
                "intents": intents
            }

            eval_samples.append(eval_sample)
            group_count += 1
            total_found += 1

    print(f"\nSuccessfully compiled {len(eval_samples)} evaluation samples.")
    print(f"Sample preview (first record):")
    if eval_samples:
        print(json.dumps(eval_samples[0], indent=2))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_samples, f, indent=2)

    print(f"\nSaved eval dataset to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build eval_dataset.json from ToolBench intermediate files.")
    parser.add_argument("--intermediate_dir", type=str, default="/home/abhishek/Desktop/projects/RAG_MOLE/data/intermediate")
    parser.add_argument("--master_registry", type=str, default="/home/abhishek/Desktop/projects/RAG_MOLE/data/processed/master_registry.json")
    parser.add_argument("--output", type=str, default="/home/abhishek/Desktop/projects/RAG_MOLE/data/processed/eval_dataset.json")
    parser.add_argument("--max_samples_per_group", type=int, default=200, help="Max samples per G1/G2/G3 group (default: 200 -> ~600 samples)")
    args = parser.parse_args()

    build_dataset(
        intermediate_dir=args.intermediate_dir,
        master_registry_path=args.master_registry,
        output_path=args.output,
        max_samples_per_group=args.max_samples_per_group
    )
