#!/usr/bin/env python3
"""Held-Out Benchmark for Nursing Clinical Language and Antonym Collapse.

Evaluates:
1. Hard-Negative Triplet Ranking: Accuracy, Mean Margin, and Antonym Collapse Rate.
2. Jargon Top-1 / Top-5 Retrieval over Taiwanese Nursing Abbreviations.
3. Breakdown across 5 core nursing subjects.

Source: Held-out 15% split of MOEX Nursing Exam Mining Dataset (seed=42).
Author: WeeMed AI / IlhaEmbed Evaluation Suite
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

SEED = 42
MAX_LEN = 32


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ModelWrapper:
    """Loads IlhaEmbed (or any ModernBERT/BERT model) and produces unit-normalized vectors."""

    def __init__(self, model_id_or_path: str, device: str = "cpu"):
        self.device = device
        print(f"Loading model '{model_id_or_path}' on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.model = AutoModel.from_pretrained(model_id_or_path).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            # ModernBERT / BERT mean pooling with attention mask
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                inputs.attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # L2 normalize
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_vecs.append(normalized.cpu().numpy())

        return np.concatenate(all_vecs, axis=0)


def load_held_out_triplets(path: Path, holdout: float = 0.15) -> list[dict]:
    """Load triplets and deterministically extract held-out test split."""
    all_triplets = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_triplets.append(json.loads(line))

    # Shuffle with fixed seed and split
    rng = random.Random(SEED)
    rng.shuffle(all_triplets)

    split_idx = int(len(all_triplets) * (1 - holdout))
    test_triplets = all_triplets[split_idx:]
    print(f"Total Triplets: {len(all_triplets)} | Test Split (15%): {len(test_triplets)}")
    return test_triplets


def load_held_out_jargon(path: Path, holdout: float = 0.15) -> list[dict]:
    """Load jargon pairs and deterministically extract held-out test split."""
    all_jargon = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("surface") and row.get("canonical"):
                all_jargon.append(row)

    rng = random.Random(SEED)
    rng.shuffle(all_jargon)

    split_idx = int(len(all_jargon) * (1 - holdout))
    test_jargon = all_jargon[split_idx:]
    print(f"Total Jargon Pairs: {len(all_jargon)} | Test Split (15%): {len(test_jargon)}")
    return test_jargon


def evaluate_triplets(model: ModelWrapper, triplets: list[dict]) -> dict:
    """Evaluate margin and accuracy on held-out triplets."""
    print(f"Embedding {len(triplets)} triplet anchors, positives, and negatives...")
    anchors = [t["anchor"] for t in triplets]
    positives = [t["positive"] for t in triplets]
    negatives = [t["hard_negative"] for t in triplets]

    vec_a = model.embed(anchors)
    vec_p = model.embed(positives)
    vec_n = model.embed(negatives)

    # Cosine similarities
    sim_pos = np.sum(vec_a * vec_p, axis=1)
    sim_neg = np.sum(vec_a * vec_n, axis=1)
    margins = sim_pos - sim_neg
    wins = margins > 0

    results = {
        "overall": {
            "count": len(triplets),
            "accuracy": float(np.mean(wins)),
            "mean_margin": float(np.mean(margins)),
            "collapse_rate": float(np.mean(margins <= 0)),
            "mean_pos_sim": float(np.mean(sim_pos)),
            "mean_neg_sim": float(np.mean(sim_neg)),
        },
        "by_polarity": {},
        "by_subject": {},
    }

    # Group by polarity
    pol_groups = defaultdict(list)
    for idx, t in enumerate(triplets):
        pol_groups[t.get("polarity", "unknown")].append(idx)

    for pol, idxs in pol_groups.items():
        sub_wins = wins[idxs]
        sub_m = margins[idxs]
        results["by_polarity"][pol] = {
            "count": len(idxs),
            "accuracy": float(np.mean(sub_wins)),
            "mean_margin": float(np.mean(sub_m)),
            "collapse_rate": float(np.mean(sub_m <= 0)),
        }

    # Group by subject
    subj_groups = defaultdict(list)
    for idx, t in enumerate(triplets):
        subj_groups[t.get("subject", "unknown")].append(idx)

    for subj, idxs in subj_groups.items():
        sub_wins = wins[idxs]
        sub_m = margins[idxs]
        results["by_subject"][subj] = {
            "count": len(idxs),
            "accuracy": float(np.mean(sub_wins)),
            "mean_margin": float(np.mean(sub_m)),
            "collapse_rate": float(np.mean(sub_m <= 0)),
        }

    return results


def evaluate_jargon(model: ModelWrapper, jargon: list[dict]) -> dict:
    """Evaluate Top-1 and Top-5 retrieval accuracy over unique canonical concepts."""
    surfaces = [j["surface"] for j in jargon]
    targets = [j["canonical"] for j in jargon]
    unique_canonicals = sorted(list(set(targets)))

    print(f"Embedding {len(surfaces)} test jargon terms and {len(unique_canonicals)} canonical pool...")
    vec_s = model.embed(surfaces)
    vec_c = model.embed(unique_canonicals)

    sim_matrix = vec_s @ vec_c.T  # (N_surfaces, N_canonicals)

    top1_correct = 0
    top5_correct = 0
    c_index = {c: i for i, c in enumerate(unique_canonicals)}

    for i, target in enumerate(targets):
        true_idx = c_index[target]
        ranked = np.argsort(-sim_matrix[i])
        if ranked[0] == true_idx:
            top1_correct += 1
        if true_idx in ranked[:5]:
            top5_correct += 1

    return {
        "count": len(jargon),
        "candidate_pool_size": len(unique_canonicals),
        "top1_accuracy": top1_correct / len(jargon),
        "top5_accuracy": top5_correct / len(jargon),
    }


def print_report(model_name: str, triplet_res: dict, jargon_res: dict):
    print("\n" + "=" * 70)
    print(f"📊 IlhaEmbed Baseline Benchmark — Nursing Domain")
    print(f"Model: {model_name}")
    print("=" * 70)

    ov = triplet_res["overall"]
    print("\n[1] Expert Hard-Negative Triplet Ranking (Held-Out 15%)")
    print(f"  • Test Count:          {ov['count']}")
    print(f"  • Accuracy (Pos > Neg): {ov['accuracy'] * 100:.2f}%")
    print(f"  • Mean Cosine Margin:  {ov['mean_margin']:+.4f} (Pos: {ov['mean_pos_sim']:.4f}, Neg: {ov['mean_neg_sim']:.4f})")
    print(f"  • Antonym Collapse:    {ov['collapse_rate'] * 100:.2f}% (Distractor rated higher than correct action)")

    print("\n  Polarity Breakdown:")
    for pol, stat in triplet_res["by_polarity"].items():
        print(f"    - {pol:<15}: Acc={stat['accuracy']*100:5.2f}% | Margin={stat['mean_margin']:+6.4f} | Collapse={stat['collapse_rate']*100:5.2f}% (N={stat['count']})")

    print("\n  Subject Breakdown:")
    for subj, stat in triplet_res["by_subject"].items():
        short_s = subj.split("(")[0].split("（")[0]
        print(f"    - {short_s:<15}: Acc={stat['accuracy']*100:5.2f}% | Margin={stat['mean_margin']:+6.4f} | Collapse={stat['collapse_rate']*100:5.2f}% (N={stat['count']})")

    print("\n[2] Nursing Jargon / Abbreviation Retrieval")
    print(f"  • Test Terms:          {jargon_res['count']}")
    print(f"  • Candidate Pool:      {jargon_res['candidate_pool_size']} concepts")
    print(f"  • Top-1 Accuracy:      {jargon_res['top1_accuracy'] * 100:.2f}%")
    print(f"  • Top-5 Accuracy:      {jargon_res['top5_accuracy'] * 100:.2f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate IlhaEmbed on Nursing Benchmark")
    parser.add_argument("--model", default="weemed/IlhaEmbed", help="Hugging Face ID or local path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    parser.add_argument("--triplets", type=Path, default=Path("training/mining/nursing_hard_negatives.jsonl"))
    parser.add_argument("--jargon", type=Path, default=Path("training/mining/nursing_jargon_pairs.tsv"))
    parser.add_argument("--out-json", type=Path, default=Path("results/nursing_baseline.json"))
    args = parser.parse_args()

    set_seed(SEED)

    if not args.triplets.exists() or not args.jargon.exists():
        raise FileNotFoundError("Mining outputs not found. Run nursing_exam_mine.py first.")

    model = ModelWrapper(args.model, device=args.device)

    triplets = load_held_out_triplets(args.triplets)
    jargon = load_held_out_jargon(args.jargon)

    triplet_res = evaluate_triplets(model, triplets)
    jargon_res = evaluate_jargon(model, jargon)

    print_report(args.model, triplet_res, jargon_res)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "model": args.model,
        "device": args.device,
        "triplets": triplet_res,
        "jargon": jargon_res,
    }
    args.out_json.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {args.out_json}")


if __name__ == "__main__":
    main()
