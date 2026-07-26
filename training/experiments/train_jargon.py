#!/usr/bin/env python3
"""Train a SMALL model to read the trade's jargon, on the full surface-to-canonical corpus.

The goal is a model that is small enough to run inside the on-prem budget and accurate enough on
jargon to be trusted -- not a strong model that cannot be deployed. Everything here follows from two
measurements:

1. On jargon, every general model tested reads roughly one term in three correctly (0.16-0.32 top1).
   The purpose-built model reaches 0.433 on ITS OWN held-out split. Jargon is a trainable skill, and
   nothing else we tried has been trained for it.
2. Our earlier attempts fed 1,652 surface pairs into a loss dominated by 37k category rows. The
   purpose-built model saw 119k pairs on a single objective. Upsampling 1,652 pairs forty times does
   not add information -- the same pairs recycled -- so the likeliest cause of the gap is the 108k
   bulk pairs we had on disk and never loaded, not a cleverer recipe.

So: one objective (InfoNCE over surface -> canonical), the full corpus, and a small body.

The held-out split is 15% by the same seed as the reference model, so the number this prints is
directly comparable to its 0.433 rather than to a contaminated full-set score. Everything measured
before this on the full set was an upper bound; this is not.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
BASE = os.environ.get("BASE_MODEL", "ckiplab/bert-tiny-chinese")
OUT = Path(os.environ.get("OUT_DIR", str(HERE / "jargon-small")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
MAX_LEN = 32
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BATCH = int(os.environ.get("BATCH", "256"))
LR = float(os.environ.get("LR", "3e-5"))
TEMP = 0.05
UPSAMPLE = int(os.environ.get("UPSAMPLE", "8"))
HOLDOUT = 0.15


def load(name: str, left: str = "a", right: str = "b") -> list[tuple[str, str]]:
    path = DATA / name
    if not path.exists():
        return []
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            a, b = (row.get(left) or "").strip(), (row.get(right) or "").strip()
            if a and b and a != b:
                pairs.append((a, b))
    return pairs


def encode(model, tok, texts: list[str], batch: int = 256, grad: bool = False) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), batch):
        ids = tok(
            texts[start : start + batch], max_length=MAX_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        ).to(DEV)
        with torch.set_grad_enabled(grad):
            hidden = model(**ids).last_hidden_state[:, 0]
        chunks.append(F.normalize(hidden, dim=1))
    return torch.cat(chunks)


@torch.no_grad()
def report(model, tok, test: list[tuple[str, str]], pool: list[str], tag: str) -> float:
    """Held-out retrieval accuracy against the full canonical pool.

    The pool is every canonical in the set, not just the held-out ones, so a held-out surface form
    competes against everything the model knows -- the runtime's actual question."""
    model.eval()
    index = {canonical: position for position, canonical in enumerate(pool)}
    scores = (encode(model, tok, [a for a, _ in test]) @ encode(model, tok, pool).T).cpu().numpy()
    top1 = sum(int(np.argmax(row) == index[gold]) for row, (_, gold) in zip(scores, test))
    top5 = sum(int(index[gold] in np.argsort(-row)[:5]) for row, (_, gold) in zip(scores, test))
    print(f"[{tag}] HELD-OUT jargon top1={top1 / len(test):.3f} top5={top5 / len(test):.3f} "
          f"(n={len(test)} pool={len(pool)})", flush=True)
    return top1 / len(test)


def main() -> int:
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    specialized = load("specialized_pairs.tsv")
    # Taiwan FDA licences: every marketed product name is another real way its active ingredient is
    # written. This is the many-surfaces-per-concept structure the corpus otherwise lacks (2.16
    # surfaces per concept), and it is drawn from approved licences rather than generated, so no
    # model has to vouch for any of it.
    bulk = load("bulk_all.tsv") + load("fda_pairs.tsv")
    rng.shuffle(specialized)
    cut = max(30, int(HOLDOUT * len(specialized)))
    test, train_specialized = specialized[:cut], specialized[cut:]
    pool = sorted({b for _, b in specialized})

    train = train_specialized * UPSAMPLE + bulk
    rng.shuffle(train)
    print(f"device={DEV} base={BASE}")
    print(f"train={len(train)} ({len(train_specialized)}x{UPSAMPLE} specialized + {len(bulk)} bulk) "
          f"| held-out={len(test)} pool={len(pool)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModel.from_pretrained(BASE, trust_remote_code=True).to(DEV)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={params:.1f}M", flush=True)

    report(model, tok, test, pool, "base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(train)
        total = 0.0
        steps = 0
        for start in range(0, len(train) - BATCH, BATCH):
            batch = train[start : start + BATCH]
            left = encode(model, tok, [a for a, _ in batch], grad=True)
            right = encode(model, tok, [b for _, b in batch], grad=True)
            logits = (left @ right.T) / TEMP
            target = torch.arange(len(batch), device=DEV)
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
            steps += 1
        print(f"epoch {epoch} loss={total / max(steps, 1):.4f}", flush=True)
        report(model, tok, test, pool, f"ep{epoch}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    print(f"saved -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
