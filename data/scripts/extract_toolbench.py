#!/usr/bin/env python3
"""
Extract ToolBench G1/G2/G3 answer data into final_dataset.jsonl format.

Processes raw answer files from ToolBench, extracts successful traces,
and produces JSONL records with:
  - task_query
  - decomposed_steps
  - candidate_tool_schemas  (tools actually used)
  - distractor_tool_schemas (tools available but not used)
"""

import json
import os
import sys
import logging
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: simple progress indicator
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                self.n += 1
                if self.n % 5000 == 0:
                    print(f"  {self.desc}: {self.n}/{self.total or '?'}...", flush=True)
                yield item
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, n=1): self.n += n

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # data/
RAW_ANSWER_DIR = BASE_DIR / "raw" / "ToolBench" / "data" / "data" / "answer"
OUTPUT_FILE = BASE_DIR / "toolbench_extracted.jsonl"
FINAL_DATASET = BASE_DIR / "final_dataset.jsonl"

ANSWER_DIRS = {
    "G1": RAW_ANSWER_DIR / "G1_answer",
    "G2": RAW_ANSWER_DIR / "G2_answer",
    "G3": RAW_ANSWER_DIR / "G3_answer",
}

LOG_FILE = BASE_DIR / "extract_toolbench.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def extract_query_id(filename: str) -> str:
    """Extract the numeric query ID from a filename like '43185_ChatGPT_DFS_woFilter_w2.json'."""
    return filename.split("_")[0]


def format_step(step_num: int, api_name: str, arguments_str: str) -> str:
    """Format a decomposed step string from a function call."""
    # Try to parse arguments for a cleaner representation
    try:
        args = json.loads(arguments_str)
        if isinstance(args, dict) and args:
            args_desc = ", ".join(f"{k}: {json.dumps(v)}" for k, v in args.items())
            return f"Step {step_num}: Call {api_name} with {args_desc}"
        else:
            return f"Step {step_num}: Call {api_name}"
    except (json.JSONDecodeError, TypeError):
        if arguments_str and arguments_str.strip():
            return f"Step {step_num}: Call {api_name} with {arguments_str.strip()}"
        return f"Step {step_num}: Call {api_name}"


def process_single_file(filepath: str) -> dict | None:
    """
    Process a single ToolBench answer JSON file and return a record
    in the final_dataset.jsonl format, or None if it should be skipped.
    """
    try:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return {"_error": f"Failed to read {filepath}: {e}"}

        # ---- Quality filters ----
        if not data.get("win", False):
            return None

        ag = data.get("answer_generation", {})

        if not ag.get("valid_data", False):
            return None

        if ag.get("finish_type", "") != "give_answer":
            return None

        # ---- Extract query ----
        query = ag.get("query", "")
        if isinstance(query, list):
            query = " ".join(str(q) for q in query)
        elif not isinstance(query, str):
            query = str(query)
        query = query.strip()
        if not query:
            return None

        # ---- Extract tool schemas ----
        functions = ag.get("function", [])
        if not functions:
            return None

        # ---- Extract function calls from the winning conversation ----
        train_messages = ag.get("train_messages", [])
        if not train_messages:
            return None

        # Use the last conversation (the winning one)
        winning_conv = train_messages[-1]

        # Walk through messages and collect function calls
        used_tool_names = set()
        decomposed_steps = []
        step_num = 0

        for msg in winning_conv:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue

            fn_call = msg.get("function_call")
            if not fn_call:
                continue

            fn_name = fn_call.get("name", "")
            fn_args = fn_call.get("arguments", "")

            # Skip the Finish meta-call
            if fn_name.lower() == "finish":
                continue

            step_num += 1
            used_tool_names.add(fn_name)
            decomposed_steps.append(format_step(step_num, fn_name, fn_args))

        # Must have at least one actual tool call
        if not decomposed_steps:
            return None

        # ---- Split tools into candidate (used) vs distractor (unused) ----
        candidate_tool_schemas = []
        distractor_tool_schemas = []

        for func_schema in functions:
            tool_name = func_schema.get("name", "")

            # Skip the Finish meta-tool if present
            if tool_name.lower() == "finish":
                continue

            schema_record = {
                "tool_name": tool_name,
                "raw_definition": json.dumps(func_schema, ensure_ascii=False),
            }

            if tool_name in used_tool_names:
                candidate_tool_schemas.append(schema_record)
            else:
                distractor_tool_schemas.append(schema_record)

        # Must have at least one candidate tool
        if not candidate_tool_schemas:
            return None

        return {
            "task_query": query,
            "decomposed_steps": decomposed_steps,
            "candidate_tool_schemas": candidate_tool_schemas,
            "distractor_tool_schemas": distractor_tool_schemas,
        }
    except Exception as e:
        return {"_error": f"Unexpected error processing {filepath}: {e}"}


def collect_files(answer_dirs: dict) -> list[tuple[str, str]]:
    """Collect all JSON file paths with their group label."""
    files = []
    for group, dirpath in answer_dirs.items():
        if not dirpath.is_dir():
            logger.warning(f"Directory not found: {dirpath}")
            continue
        for fname in os.listdir(dirpath):
            if fname.endswith(".json"):
                files.append((group, str(dirpath / fname)))
    logger.info(f"Collected {len(files)} total files across {len(answer_dirs)} groups")
    return files


def run_extraction(
    answer_dirs: dict,
    output_path: Path,
    num_workers: int = None,
):
    """Run the full extraction pipeline."""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Collect all files
    all_files = collect_files(answer_dirs)
    if not all_files:
        logger.error("No files found. Exiting.")
        return

    groups = [g for g, _ in all_files]
    filepaths = [fp for _, fp in all_files]

    # Process files in parallel
    logger.info(f"Processing {len(filepaths)} files with {num_workers} workers...")

    results = []
    errors = 0
    skipped = 0
    group_counts = Counter()
    group_totals = Counter(groups)

    with Pool(num_workers) as pool:
        for i, result in enumerate(
            tqdm(
                pool.imap(process_single_file, filepaths, chunksize=500),
                total=len(filepaths),
                desc="Extracting",
            )
        ):
            group = groups[i]
            if result is None:
                skipped += 1
            elif "_error" in result:
                errors += 1
                logger.debug(result["_error"])
            else:
                results.append(result)
                group_counts[group] += 1

    # Deduplicate by query text
    seen_queries = set()
    unique_results = []
    duplicates = 0
    for record in results:
        q = record["task_query"]
        if q in seen_queries:
            duplicates += 1
            continue
        seen_queries.add(q)
        unique_results.append(record)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in unique_results:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {len(filepaths)}")
    logger.info(f"  Skipped (filtered out): {skipped}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Extracted (pre-dedup): {len(results)}")
    logger.info(f"  Duplicates removed: {duplicates}")
    logger.info(f"  Final records: {len(unique_results)}")
    logger.info(f"")
    logger.info(f"Per-group breakdown:")
    for g in sorted(group_totals.keys()):
        logger.info(f"  {g}: {group_counts[g]}/{group_totals[g]} files → records")
    logger.info(f"")
    logger.info(f"Output written to: {output_path}")

    # Validation stats
    if unique_results:
        cand_counts = [len(r["candidate_tool_schemas"]) for r in unique_results]
        dist_counts = [len(r["distractor_tool_schemas"]) for r in unique_results]
        step_counts = [len(r["decomposed_steps"]) for r in unique_results]
        logger.info(f"")
        logger.info(f"Validation stats:")
        logger.info(f"  candidate_tools: min={min(cand_counts)} max={max(cand_counts)} avg={sum(cand_counts)/len(cand_counts):.1f}")
        logger.info(f"  distractor_tools: min={min(dist_counts)} max={max(dist_counts)} avg={sum(dist_counts)/len(dist_counts):.1f}")
        logger.info(f"  decomposed_steps: min={min(step_counts)} max={max(step_counts)} avg={sum(step_counts)/len(step_counts):.1f}")

    return unique_results


def merge_into_final(toolbench_path: Path, final_path: Path):
    """Merge toolbench extracted data into the final dataset, preserving existing records."""
    # Read existing final dataset
    existing = []
    if final_path.exists():
        with open(final_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
        logger.info(f"Loaded {len(existing)} existing records from {final_path}")

    # Read toolbench extracted
    toolbench = []
    with open(toolbench_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                toolbench.append(json.loads(line))
    logger.info(f"Loaded {len(toolbench)} ToolBench records from {toolbench_path}")

    # Deduplicate against existing by query
    existing_queries = {r["task_query"] for r in existing}
    new_records = [r for r in toolbench if r["task_query"] not in existing_queries]
    logger.info(f"After cross-dedup: {len(new_records)} new ToolBench records to add")

    # Combine and write
    combined = existing + new_records
    with open(final_path, "w", encoding="utf-8") as f:
        for record in combined:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(combined)} total records to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract ToolBench data into final dataset format")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--merge", action="store_true", help="Merge extracted data into final_dataset.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    output = Path(args.output) if args.output else OUTPUT_FILE

    if args.merge:
        if not output.exists():
            logger.error(f"Extracted file not found: {output}. Run extraction first.")
            sys.exit(1)
        merge_into_final(output, FINAL_DATASET)
    else:
        run_extraction(ANSWER_DIRS, output, num_workers=args.workers)


if __name__ == "__main__":
    main()
