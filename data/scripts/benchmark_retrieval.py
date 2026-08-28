#!/usr/bin/env python3
"""
benchmark_retrieval.py
----------------------
Lightweight Zero-Shot Vector Retrieval Benchmark for RAG-MoLE.

Compares:
1. Baseline Retrieval: Raw user query embedded using all-MiniLM-L6-v2.
2. Semantic Retrieval (Ours): Combined semantic intents embedded using all-MiniLM-L6-v2.

Evaluates:
- Hit@1
- Hit@5
- Mean Reciprocal Rank (MRR)

Generates:
- Console stdout metrics table
- Grouped bar chart saved as `benchmark_results.png`
"""

import os
import json
import re
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer


def normalize_string(s: str) -> str:
    """Normalize text for consistent string matching."""
    if not s:
        return ""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


class APIRegistryIndexer:
    """Handles loading and indexing of master_registry.json."""
    
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.apis: List[Dict[str, Any]] = []
        self.candidate_texts: List[str] = []
        self.api_name_to_index: Dict[str, List[int]] = {}
        
    def load_and_index(self) -> Tuple[List[str], Dict[str, List[int]]]:
        print(f"Loading master registry from '{self.registry_path}'...")
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            self.apis = json.load(f)
            
        print(f"Indexing {len(self.apis)} APIs...")
        for idx, item in enumerate(self.apis):
            tool_name = str(item.get("tool_name", "") or "").strip()
            api_name = str(item.get("api_name", "") or "").strip()
            api_description = str(item.get("api_description", "") or "").strip()
            
            # Form required combined text field
            text = f"{tool_name} {api_name}: {api_description}"
            self.candidate_texts.append(text)
            
            # Build flexible lookup indices for ground truth matching
            norm_api = normalize_string(api_name)
            norm_tool = normalize_string(tool_name)
            
            # Key variants: exact api_name, tool+api, api_for_tool snake_case
            keys = [
                norm_api,
                f"{norm_api}for{norm_tool}",
                f"{norm_tool}{norm_api}",
                normalize_string(f"{api_name}_for_{tool_name}".replace(" ", "_"))
            ]
            
            for key in keys:
                if key:
                    if key not in self.api_name_to_index:
                        self.api_name_to_index[key] = []
                    self.api_name_to_index[key].append(idx)
                    
        print(f"Successfully created combined text descriptions for {len(self.candidate_texts)} APIs.")
        return self.candidate_texts, self.api_name_to_index

    def resolve_gt_indices(self, gt_api_names: List[str]) -> List[int]:
        """Resolves ground truth API names to their indices in candidate_texts."""
        matched_indices = set()
        for name in gt_api_names:
            norm_name = normalize_string(name)
            if norm_name in self.api_name_to_index:
                matched_indices.update(self.api_name_to_index[norm_name])
            else:
                # Try splitting by '_for_' if present (ToolBench format)
                if '_for_' in name.lower():
                    parts = name.lower().split('_for_')
                    api_part = normalize_string(parts[0])
                    tool_part = normalize_string(parts[1]) if len(parts) > 1 else ""
                    combo_key = f"{api_part}for{tool_part}"
                    if combo_key in self.api_name_to_index:
                        matched_indices.update(self.api_name_to_index[combo_key])
                    elif api_part in self.api_name_to_index:
                        matched_indices.update(self.api_name_to_index[api_part])
        return list(matched_indices)


class VectorRetriever:
    """Manages sentence embeddings and fast vectorized similarity calculation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading SentenceTransformer model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.api_embeddings: Optional[np.ndarray] = None
        
    def index_apis(self, candidate_texts: List[str], batch_size: int = 512) -> np.ndarray:
        print(f"Computing L2-normalized embeddings for {len(candidate_texts)} candidate APIs...")
        start_time = time.time()
        # normalize_embeddings=True ensures dot product equals cosine similarity
        self.api_embeddings = self.model.encode(
            candidate_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        elapsed = time.time() - start_time
        print(f"API embedding completed in {elapsed:.2f}s (Shape: {self.api_embeddings.shape}).")
        return self.api_embeddings

    def encode_queries(self, queries: List[str], batch_size: int = 256) -> np.ndarray:
        return self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def compute_similarity_matrix(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Fully vectorized cosine similarity computation: Q @ M.T."""
        if self.api_embeddings is None:
            raise ValueError("APIs must be indexed before computing similarity.")
        # Vectorized dot product of normalized vectors
        return np.dot(query_embeddings, self.api_embeddings.T)


class BenchmarkEvaluator:
    """Runs zero-shot vector retrieval evaluation and computes metrics."""
    
    @staticmethod
    def evaluate(
        similarity_matrix: np.ndarray,
        ground_truth_indices_list: List[List[int]]
    ) -> Dict[str, float]:
        """
        Calculates Hit@1, Hit@5, and MRR.
        
        Args:
            similarity_matrix: Shape [N_queries, N_apis] containing cosine similarities.
            ground_truth_indices_list: List of ground truth API indices for each query.
        """
        n_queries = len(ground_truth_indices_list)
        hit1_count = 0
        hit5_count = 0
        reciprocal_ranks = []
        valid_queries = 0

        for i in range(n_queries):
            gt_indices = set(ground_truth_indices_list[i])
            if not gt_indices:
                # Edge case: No ground truth APIs mapped in registry for this query
                reciprocal_ranks.append(0.0)
                continue

            valid_queries += 1
            scores = similarity_matrix[i]
            
            # Rank all candidates descending
            # Using argsort on negative scores for fast top ranking
            top_ranked_indices = np.argsort(-scores)
            
            # Find best (min) 1-based rank among all ground truth APIs for this query
            ranks = []
            for gt_idx in gt_indices:
                # np.where returns tuple of arrays; top_ranked_indices is 1D
                matching_pos = np.where(top_ranked_indices == gt_idx)[0]
                if len(matching_pos) > 0:
                    rank = matching_pos[0] + 1  # 1-based rank
                    ranks.append(rank)

            if ranks:
                best_rank = min(ranks)
                rr = 1.0 / best_rank
                reciprocal_ranks.append(rr)
                
                if best_rank == 1:
                    hit1_count += 1
                if best_rank <= 5:
                    hit5_count += 1
            else:
                reciprocal_ranks.append(0.0)

        total_eval = n_queries if n_queries > 0 else 1
        hit1 = (hit1_count / total_eval) * 100.0
        hit5 = (hit5_count / total_eval) * 100.0
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

        return {
            "Hit@1": hit1,
            "Hit@5": hit5,
            "MRR": mrr,
            "Total_Queries": n_queries,
            "Valid_Queries": valid_queries
        }


class BenchmarkVisualizer:
    """Generates console table and grouped bar chart visualization."""
    
    @staticmethod
    def print_metrics_table(baseline_metrics: Dict[str, float], semantic_metrics: Dict[str, float]) -> None:
        print("\n" + "="*65)
        print("         ZERO-SHOT VECTOR RETRIEVAL BENCHMARK RESULTS")
        print("="*65)
        df = pd.DataFrame([
            {
                "Approach": "Raw Query (Baseline)",
                "Hit@1 (%)": f"{baseline_metrics['Hit@1']:.2f}%",
                "Hit@5 (%)": f"{baseline_metrics['Hit@5']:.2f}%",
                "MRR": f"{baseline_metrics['MRR']:.4f}"
            },
            {
                "Approach": "Semantic Intents (Ours)",
                "Hit@1 (%)": f"{semantic_metrics['Hit@1']:.2f}%",
                "Hit@5 (%)": f"{semantic_metrics['Hit@5']:.2f}%",
                "MRR": f"{semantic_metrics['MRR']:.4f}"
            }
        ])
        print(df.to_string(index=False))
        print("="*65 + "\n")

    @staticmethod
    def plot_benchmark_chart(
        baseline_metrics: Dict[str, float],
        semantic_metrics: Dict[str, float],
        output_image_path: str = "benchmark_results.png"
    ) -> None:
        metrics_names = ['Hit@1 (%)', 'Hit@5 (%)', 'MRR']
        
        baseline_values = [
            baseline_metrics['Hit@1'],
            baseline_metrics['Hit@5'],
            baseline_metrics['MRR'] * 100.0  # Scaled to % for visual consistency
        ]
        
        semantic_values = [
            semantic_metrics['Hit@1'],
            semantic_metrics['Hit@5'],
            semantic_metrics['MRR'] * 100.0  # Scaled to % for visual consistency
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        rects1 = ax.bar(x - width/2, baseline_values, width, label='Raw Query (Baseline)', color='#4C72B0', edgecolor='none', alpha=0.9)
        rects2 = ax.bar(x + width/2, semantic_values, width, label='Semantic Intents (Ours)', color='#55A868', edgecolor='none', alpha=0.9)

        ax.set_ylabel('Score / Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Zero-Shot Vector Retrieval Benchmark: Baseline vs. Semantic Intents', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
        ax.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='lightgray')
        ax.set_ylim(0, max(max(baseline_values), max(semantic_values)) * 1.25 + 5)

        # Add bar value annotations
        def autolabel(rects, is_mrr=False):
            for i, rect in enumerate(rects):
                height = rect.get_height()
                if i == 2: # MRR
                    actual_mrr = height / 100.0
                    text = f"{actual_mrr:.4f}"
                else:
                    text = f"{height:.1f}%"
                ax.annotate(text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),  # 4 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300)
        plt.close()
        print(f"Saved benchmark grouped bar chart to '{output_image_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Run Zero-Shot Vector Retrieval Benchmark.")
    parser.add_argument("--registry", type=str, default="/home/abhishek/Desktop/projects/RAG_MOLE/data/processed/master_registry.json")
    parser.add_argument("--eval_dataset", type=str, default="/home/abhishek/Desktop/projects/RAG_MOLE/data/processed/eval_dataset.json")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output_chart", type=str, default="benchmark_results.png")
    args = parser.parse_args()

    # Step 1: Setup and Indexing
    indexer = APIRegistryIndexer(args.registry)
    candidate_texts, name_to_index = indexer.load_and_index()

    retriever = VectorRetriever(args.model)
    retriever.index_apis(candidate_texts)

    # Step 2: Evaluation Loop Setup
    print(f"\nLoading evaluation dataset from '{args.eval_dataset}'...")
    with open(args.eval_dataset, 'r', encoding='utf-8') as f:
        eval_dataset = json.load(f)
    print(f"Loaded {len(eval_dataset)} evaluation samples.")

    baseline_queries = []
    semantic_intent_queries = []
    ground_truth_indices_list = []

    for sample in eval_dataset:
        raw_query = sample.get("query", "").strip()
        baseline_queries.append(raw_query)

        # Combine intents list into a single string (separated by periods)
        intents = sample.get("intents", [])
        if isinstance(intents, list) and intents:
            intent_str = ". ".join([str(i).strip() for i in intents if i])
        else:
            intent_str = raw_query
        semantic_intent_queries.append(intent_str)

        # Extract ground truth APIs from trace
        trace = sample.get("trace", [])
        gt_names = []
        for step in trace:
            if isinstance(step, dict) and "api_name" in step:
                gt_names.append(step["api_name"])
            elif isinstance(step, str):
                gt_names.append(step)
                
        gt_indices = indexer.resolve_gt_indices(gt_names)
        ground_truth_indices_list.append(gt_indices)

    # Compute Embeddings for Baseline & Semantic
    print("\nEmbedding Raw Queries (Baseline)...")
    baseline_query_embeddings = retriever.encode_queries(baseline_queries)
    
    print("Embedding Semantic Intents (Ours)...")
    semantic_query_embeddings = retriever.encode_queries(semantic_intent_queries)

    # Vectorized Cosine Similarity Matrix Computation
    print("\nComputing vectorized similarity matrix for Baseline...")
    baseline_sim_matrix = retriever.compute_similarity_matrix(baseline_query_embeddings)

    print("Computing vectorized similarity matrix for Semantic Intents...")
    semantic_sim_matrix = retriever.compute_similarity_matrix(semantic_query_embeddings)

    # Step 3: Metrics Calculation
    print("\nCalculating retrieval metrics...")
    baseline_metrics = BenchmarkEvaluator.evaluate(baseline_sim_matrix, ground_truth_indices_list)
    semantic_metrics = BenchmarkEvaluator.evaluate(semantic_sim_matrix, ground_truth_indices_list)

    # Step 4: Visualization
    BenchmarkVisualizer.print_metrics_table(baseline_metrics, semantic_metrics)
    BenchmarkVisualizer.plot_benchmark_chart(baseline_metrics, semantic_metrics, args.output_chart)


if __name__ == "__main__":
    main()
