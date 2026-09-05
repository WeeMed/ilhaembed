#!/usr/bin/env python3
"""Contrastive Fine-Tuning of IlhaEmbed on Taiwan MOEX Nursing Hard Negatives and Jargon.

Objective:
- Joint Triplet Margin Loss + InfoNCE to push hard distractors and contraindicated actions apart.
- Zero test contamination: uses only the 85% train split (seed=42).
- Output: fine-tuned checkpoint saved to /mnt/model-cache/ for subsequent ONNX export & benchmark.

Author: WeeMed AI / IlhaEmbed Training Pipeline
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

SEED = 42
MAX_LEN = 32


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TripletDataset(Dataset):
    def __init__(self, triplets: list[dict], jargon_pairs: list[dict]):
        self.data = []
        # Add clinical triplets
        for t in triplets:
            self.data.append({
                "anchor": t["anchor"],
                "positive": t["positive"],
                "negative": t["hard_negative"],
                "weight": 1.0,
            })

        # Add jargon pairs as pseudo-triplets with random negative from canonical pool
        canonicals = [j["canonical"] for j in jargon_pairs]
        for j in jargon_pairs:
            neg = random.choice(canonicals)
            while neg == j["canonical"] and len(canonicals) > 1:
                neg = random.choice(canonicals)
            self.data.append({
                "anchor": j["surface"],
                "positive": j["canonical"],
                "negative": neg,
                "weight": 1.5,  # Slightly upweight jargon alignment
            })

        random.Random(SEED).shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ModernBertEmbedder(nn.Module):
    """Wraps ModernBERT / BERT with mean-pooling and L2 normalization."""

    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return F.normalize(mean_pooled, p=2, dim=1)


def load_train_splits(triplets_path: Path, jargon_path: Path, holdout: float = 0.15):
    """Load only the 85% train split (identical split logic as nursing_eval.py)."""
    # 1. Triplets
    all_triplets = []
    with triplets_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_triplets.append(json.loads(line))
    rng_t = random.Random(SEED)
    rng_t.shuffle(all_triplets)
    split_t = int(len(all_triplets) * (1 - holdout))
    train_triplets = all_triplets[:split_t]

    # 2. Jargon
    all_jargon = []
    with jargon_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("surface") and row.get("canonical"):
                all_jargon.append(row)
    rng_j = random.Random(SEED)
    rng_j.shuffle(all_jargon)
    split_j = int(len(all_jargon) * (1 - holdout))
    train_jargon = all_jargon[:split_j]

    print(f"Loaded Train Triplets: {len(train_triplets)} (Held-out test: {len(all_triplets) - len(train_triplets)})")
    print(f"Loaded Train Jargon:   {len(train_jargon)} (Held-out test: {len(all_jargon) - len(train_jargon)})")
    return train_triplets, train_jargon


def collate_fn(batch, tokenizer, device):
    anchors = [b["anchor"] for b in batch]
    positives = [b["positive"] for b in batch]
    negatives = [b["negative"] for b in batch]
    weights = torch.tensor([b["weight"] for b in batch], dtype=torch.float32, device=device)

    tok_a = tokenizer(anchors, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    tok_p = tokenizer(positives, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    tok_n = tokenizer(negatives, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)

    return tok_a, tok_p, tok_n, weights


def train(args):
    set_seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Load data
    train_triplets, train_jargon = load_train_splits(args.triplets, args.jargon, holdout=0.15)
    dataset = TripletDataset(train_triplets, train_jargon)
    print(f"Total Combined Training Items: {len(dataset)}")

    # 2. Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = ModernBertEmbedder(args.base_model).to(device)

    # 3. DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, device),
        drop_last=True,
    )

    # 4. Optimizer & Scheduler
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    margin = args.margin
    tau = args.temperature

    print(f"\nStarting fine-tuning: {args.epochs} epochs, {total_steps} steps, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Margin: {margin}, Temp: {tau}\n")

    model.train()
    step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_triplet_loss = 0.0
        epoch_infonce_loss = 0.0
        correct_count = 0
        total_items = 0

        for tok_a, tok_p, tok_n, weights in loader:
            optimizer.zero_grad()

            vec_a = model(tok_a["input_ids"], tok_a["attention_mask"])
            vec_p = model(tok_p["input_ids"], tok_p["attention_mask"])
            vec_n = model(tok_n["input_ids"], tok_n["attention_mask"])

            # Cosine similarities
            sim_ap = torch.sum(vec_a * vec_p, dim=-1)
            sim_an = torch.sum(vec_a * vec_n, dim=-1)

            # 1. Triplet Margin Loss: max(0, sim_an - sim_ap + margin)
            triplet_losses = F.relu(sim_an - sim_ap + margin) * weights
            loss_triplet = triplet_losses.mean()

            # 2. In-batch InfoNCE Loss over (A, P)
            sim_matrix = (vec_a @ vec_p.T) / tau
            labels = torch.arange(len(vec_a), device=device)
            loss_infonce = F.cross_entropy(sim_matrix, labels)

            # Combined loss
            loss = loss_triplet + 0.5 * loss_infonce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            epoch_triplet_loss += loss_triplet.item()
            epoch_infonce_loss += loss_infonce.item()
            correct_count += (sim_ap > sim_an).sum().item()
            total_items += len(vec_a)

            if step % 50 == 0 or step == total_steps:
                cur_margin = (sim_ap - sim_an).mean().item()
                cur_acc = (sim_ap > sim_an).float().mean().item() * 100
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step}/{total_steps}] | "
                    f"Loss: {loss.item():.4f} (Trip: {loss_triplet.item():.4f}, Info: {loss_infonce.item():.4f}) | "
                    f"Margin: {cur_margin:+.4f} | Batch Acc: {cur_acc:.1f}%"
                )

        train_acc = (correct_count / total_items) * 100
        print(f"--- Epoch {epoch} Complete | Avg Triplet Loss: {epoch_triplet_loss/len(loader):.4f} | Train Acc: {train_acc:.2f}% ---\n")

    # Save model and tokenizer
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving fine-tuned model and tokenizer to {out_dir}...")
    model.encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Model saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IlhaEmbed on Nursing Data")
    parser.add_argument("--base-model", default="weemed/IlhaEmbed", help="Base model HuggingFace ID")
    parser.add_argument("--triplets", type=Path, default=Path("training/mining/nursing_hard_negatives.jsonl"))
    parser.add_argument("--jargon", type=Path, default=Path("training/mining/nursing_jargon_pairs.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/ilhaembed-nursing-v1"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
