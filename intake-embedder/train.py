#!/usr/bin/env python3
"""Train the intake embedder for the decision production actually makes.

Base is `BAAI/bge-small-zh-v1.5`, the model the importer already runs. That is a deliberate choice,
not inertia: it is 512-dimensional and small enough to quantize to ~23 MB, which keeps it inside the
free on-prem tier where a 179 MB BERT-base would not fit, and it keeps the deployment shape
identical so adoption is a model-file swap plus a threshold re-fit rather than a re-architecture.

Four objectives train one encoder, because the runtime failure has four shapes:

1. **Category supervision (SupCon).** The production classifier scores a fragment against per-category
   centroids. Nothing ever trained that geometry. This pulls a category's members together and pushes
   categories apart, which is the objective the 0.70 gate is actually thresholding.
2. **Synonym / cross-lingual InfoNCE.** Same concept written two ways -- a Chinese display and its
   English display, an abbreviation and its expansion -- must land together. This is what keeps the
   model robust to how staff actually write, and it is the one objective prior work did train.
3. **Hard negatives.** Code-adjudicated near-identical pairs that denote DIFFERENT concepts get an
   explicit margin. This attacks antonym collapse directly: character overlap must stop dominating.
4. **Rejection.** Most real cells are not clinical terms. Measurements, dates, identifiers and marks
   are pushed away from every category centroid, so the classifier can decline instead of over-routing.

Evaluation uses the hand-labelled production sets, which are NEVER trained on. They are the only real
in-domain measurement; training on them would make every number here meaningless.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EVAL = HERE / "eval"
RESEARCH_DATA = HERE.parent / "data"
OUT = HERE / os.environ.get("RUN_NAME", "intake-embedder-v2")
BASE = os.environ.get("BASE_MODEL", "BAAI/bge-small-zh-v1.5")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_LEN = 32

EPOCHS = int(os.environ.get("EPOCHS", "3"))
BATCH = int(os.environ.get("BATCH", "128"))
LR = float(os.environ.get("LR", "2e-5"))
TEMP = 0.05
HARDNEG_MARGIN = 0.30
# The category objective is what is missing from every prior model, so it carries the most weight.
# Rejection is weighted below it: refusing everything is a cheap way to lower the loss and a useless
# classifier, so it must not be allowed to dominate.
W_CATEGORY, W_PAIR, W_HARDNEG, W_REJECT = 1.0, 1.0, 2.0, 2.0
# Cap examples per category when sampling. Run 1 showed the cost of leaving this uncapped: pulling
# 12k condition terms together inflated that category's basin until held-out negatives scored INSIDE
# it (neg_p95 rose 0.833 -> 0.916 over two epochs). Category cohesion and rejection are in direct
# tension, and an unbalanced category wins that fight by sheer count.
PER_CATEGORY_CAP = 2000
# Junk must sit clearly below any plausible accept threshold, not merely below 1.0.
REJECT_CEILING = 0.25


def read_tsv(name: str) -> list[dict[str, str]]:
    path = DATA / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def encode(model, tok, texts: list[str], batch: int = 256, grad: bool = False) -> torch.Tensor:
    """CLS-pooled, L2-normalized embeddings -- the same representation the runtime consumes."""
    chunks = []
    for start in range(0, len(texts), batch):
        ids = tok(
            texts[start : start + batch],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(DEV)
        with torch.set_grad_enabled(grad):
            hidden = model(**ids).last_hidden_state[:, 0]
        chunks.append(F.normalize(hidden, dim=1))
    return torch.cat(chunks)


# ---------------------------------------------------------------------------
# Held-out evaluation -- mirrors the production decision exactly
# ---------------------------------------------------------------------------
def readable_by_model(text: str) -> bool:
    """The gate the runtime applies before scoring anything.

    Production refuses text with no Han character, because a Chinese model collapses such tokens onto
    one vector. Measuring on fragments the runtime never scores would report a failure the product
    does not have -- and run 1's negative set was mostly bare numbers, exactly that case."""
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def load_eval() -> tuple[list[tuple[str, str]], list[str], dict[str, list[str]]]:
    positives: list[tuple[str, str]] = []
    negatives: list[str] = []
    for name in ("semantic_labels.jsonl", "gold_labels.jsonl"):
        path = EVAL / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            fragment, category = row.get("fragment"), row.get("category")
            if not fragment:
                continue
            if not readable_by_model(str(fragment)):
                continue  # the runtime refuses these outright; scoring them measures nothing
            if category and category != "residue":
                positives.append((fragment, category))
            else:
                negatives.append(fragment)
    seeds = json.loads((EVAL / "taxonomy_seeds.json").read_text(encoding="utf-8"))
    return positives, negatives, seeds


def build_centroids(model, tok, seeds: dict[str, list[str]]) -> tuple[list[str], torch.Tensor]:
    """Per-category centroid from seed exemplars only -- exactly how the runtime builds prototypes.

    Evaluating against centroids built from the TRAINING data instead would flatter the model by
    measuring something production never does."""
    names, vectors = [], []
    for category, exemplars in sorted(seeds.items()):
        terms = [t for t in exemplars if t and t.strip()]
        if not terms:
            continue
        centroid = encode(model, tok, terms).mean(dim=0)
        names.append(category)
        vectors.append(centroid / centroid.norm().clamp(min=1e-9))
    return names, torch.stack(vectors)


@torch.no_grad()
def evaluate(model, tok, tag: str) -> dict[str, float]:
    model.eval()
    positives, negatives, seeds = load_eval()
    names, centroids = build_centroids(model, tok, seeds)
    index = {name: position for position, name in enumerate(names)}

    scored = [(f, c) for f, c in positives if c in index]
    top1 = margin_sum = 0.0
    pos_scores: list[float] = []
    if scored:
        vectors = encode(model, tok, [f for f, _ in scored])
        similarity = (vectors @ centroids.T).cpu().numpy()
        for row, (_, category) in zip(similarity, scored):
            order = np.argsort(-row)
            top1 += order[0] == index[category]
            margin_sum += float(row[order[0]] - row[order[1]])
            pos_scores.append(float(row[order[0]]))
        top1 /= len(scored)
        margin_sum /= len(scored)

    neg_top: list[float] = []
    if negatives:
        vectors = encode(model, tok, negatives)
        neg_top = (vectors @ centroids.T).max(dim=1).values.cpu().tolist()

    # Separation is the number that matters and the only one comparable across models: absolute
    # cosine shifts wholesale when the objective changes, so a raw threshold says nothing on its own.
    separation = 0.0
    if pos_scores and neg_top:
        separation = float(np.percentile(pos_scores, 5) - np.percentile(neg_top, 95))

    metrics = {
        "top1": top1,
        "margin": margin_sum,
        "pos_p5": float(np.percentile(pos_scores, 5)) if pos_scores else 0.0,
        "neg_p95": float(np.percentile(neg_top, 95)) if neg_top else 0.0,
        "separation": separation,
    }
    print(
        f"[{tag}] category top1={metrics['top1']:.3f} margin={metrics['margin']:.3f} | "
        f"pos_p5={metrics['pos_p5']:.3f} neg_p95={metrics['neg_p95']:.3f} "
        f"separation={metrics['separation']:+.3f} (n_pos={len(scored)} n_neg={len(negatives)})",
        flush=True,
    )
    return metrics


@torch.no_grad()
def report_jargon(model, tok, tag: str) -> float:
    """Surface-form retrieval accuracy: the trade's own shorthand against its canonical terms.

    Reported every epoch because it is the judgment the product is graded on. A run that improves
    taxonomy while this falls is not progress, and without measuring it here that trade is invisible."""
    path = RESEARCH_DATA / "specialized_pairs.tsv"
    if not path.exists():
        return 0.0
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            a, b = (row.get("a") or "").strip(), (row.get("b") or "").strip()
            if a and b and a != b:
                pairs.append((a, b))
    if not pairs:
        return 0.0
    model.eval()
    pool = sorted({b for _, b in pairs})
    index = {c: i for i, c in enumerate(pool)}
    surfaces = encode(model, tok, [a for a, _ in pairs])
    canonicals = encode(model, tok, pool)
    scores = (surfaces @ canonicals.T).cpu().numpy()
    top1 = sum(int(np.argmax(row) == index[gold]) for row, (_, gold) in zip(scores, pairs))
    top5 = sum(int(index[gold] in np.argsort(-row)[:5]) for row, (_, gold) in zip(scores, pairs))
    print(f"  [{tag}] JARGON top1={top1 / len(pairs):.3f} top5={top5 / len(pairs):.3f} "
          f"(n={len(pairs)} pool={len(pool)}; TRAINED-ON, so an upper bound)", flush=True)
    return top1 / len(pairs)


@torch.no_grad()
def report_antonyms(model, tok) -> None:
    """The shipped failure, checked directly: does a term still sit next to its opposite?"""
    path = EVAL / "antonym_probes.tsv"
    if not path.exists():
        return
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"), delimiter="\t")]
    if not rows:
        return
    model.eval()
    left = encode(model, tok, [r["a"] for r in rows])
    right = encode(model, tok, [r["b"] for r in rows])
    similarity = (left * right).sum(dim=1).cpu().tolist()
    print("  antonym pair similarity (lower is better):", flush=True)
    for row, score in list(zip(rows, similarity))[:10]:
        print(f"    {score:.3f}  {row['a']} <-> {row['b']}", flush=True)
    print(f"    mean={sum(similarity) / len(similarity):.3f}", flush=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def supcon(vectors: torch.Tensor, labels: list[str]) -> torch.Tensor:
    """Supervised contrastive loss: every same-category pair in the batch is a positive.

    This is the objective that shapes centroid geometry. Batches with no repeated category carry no
    signal, so the sampler groups categories deliberately rather than shuffling uniformly."""
    device = vectors.device
    ids = torch.tensor([hash(label) % (2**31) for label in labels], device=device)
    match = (ids[:, None] == ids[None, :]).float()
    match.fill_diagonal_(0)
    if match.sum() == 0:
        return torch.zeros((), device=device)
    logits = (vectors @ vectors.T) / TEMP
    logits.fill_diagonal_(-1e4)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_sample = (match * log_prob).sum(dim=1) / match.sum(dim=1).clamp(min=1)
    return -per_sample[match.sum(dim=1) > 0].mean()


def category_batches(rows: list[tuple[str, str]], rng: random.Random, size: int):
    """Yield batches that contain several members of each of a few categories.

    A uniformly shuffled batch of 128 over 16 categories gives thin, noisy positives; grouping makes
    each batch a real comparison between a handful of categories."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for text, category in rows:
        by_category[category].append(text)
    for category, terms in by_category.items():
        rng.shuffle(terms)
        del terms[PER_CATEGORY_CAP:]
    categories = list(by_category)
    per_category = 8
    groups = max(2, size // per_category)
    for _ in range(sum(len(v) for v in by_category.values()) // size):
        batch: list[tuple[str, str]] = []
        for category in rng.sample(categories, min(groups, len(categories))):
            pool = by_category[category]
            if len(pool) < 2:
                continue
            picks = rng.sample(pool, min(per_category, len(pool)))
            batch.extend((term, category) for term in picks)
        if len(batch) >= 8:
            yield batch


def main() -> int:
    rng = random.Random(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    category_rows = [(r["text"], r["category"]) for r in read_tsv("category.tsv") if r.get("text")]
    pair_rows = [(r["a"], r["b"]) for r in read_tsv("synonyms.tsv") if r.get("a") and r.get("b")]
    colloquial = [(r["a"], r["b"]) for r in read_tsv("colloquial.tsv") if r.get("a") and r.get("b")]
    hard_rows = [(r["anchor"], r["negative"]) for r in read_tsv("hardneg.tsv") if r.get("anchor")]
    from field_vocabulary import POLARITY_PAIRS

    # Upsampled hard: the mined pairs are numerous but mostly easy, so without this the curated
    # opposites contribute a rounding error to the loss and the collision they name goes untrained.
    hard_rows = hard_rows + list(POLARITY_PAIRS) * 40
    reject_rows = [r["text"] for r in read_tsv("negatives.tsv") if r.get("text")]

    # The colloquial set is three orders of magnitude smaller than the bulk pairs but carries the
    # surface forms staff actually write; without upsampling the bulk simply drowns it.
    pair_rows = pair_rows + colloquial * int(os.environ.get("COLLOQUIAL_UPSAMPLE", "8"))
    rng.shuffle(pair_rows)
    rng.shuffle(hard_rows)
    rng.shuffle(reject_rows)

    print(
        f"device={DEV} base={BASE}\n"
        f"category={len(category_rows)} pairs={len(pair_rows)} "
        f"(colloquial={len(colloquial)}x8) hardneg={len(hard_rows)} reject={len(reject_rows)}",
        flush=True,
    )

    # AutoModel with trust_remote_code so a candidate base with a custom architecture (Jina's
    # ALiBi BERT variant) loads on the same path as a stock BERT. Every candidate is read the same
    # way -- CLS of last_hidden_state, L2-normalized -- so a base swap changes the weights and
    # nothing else about how the comparison is run.
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModel.from_pretrained(BASE, trust_remote_code=True).to(DEV)

    evaluate(model, tok, "base")
    report_jargon(model, tok, "base")
    report_antonyms(model, tok)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    seeds = load_eval()[2]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batches = list(category_batches(category_rows, rng, BATCH))
        rng.shuffle(pair_rows)
        totals = defaultdict(float)

        for step, batch in enumerate(batches):
            loss = torch.zeros((), device=DEV)

            texts = [t for t, _ in batch]
            labels = [c for _, c in batch]
            vectors = encode(model, tok, texts, grad=True)
            loss_category = supcon(vectors, labels)
            loss = loss + W_CATEGORY * loss_category
            totals["category"] += float(loss_category)

            offset = (step * BATCH) % max(1, len(pair_rows) - BATCH)
            pairs = pair_rows[offset : offset + BATCH]
            if len(pairs) >= 2:
                left = encode(model, tok, [a for a, _ in pairs], grad=True)
                right = encode(model, tok, [b for _, b in pairs], grad=True)
                logits = (left @ right.T) / TEMP
                target = torch.arange(len(pairs), device=DEV)
                loss_pair = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
                loss = loss + W_PAIR * loss_pair
                totals["pair"] += float(loss_pair)

            if hard_rows:
                offset = (step * 32) % max(1, len(hard_rows) - 32)
                hard = hard_rows[offset : offset + 32]
                if hard:
                    anchor = encode(model, tok, [a for a, _ in hard], grad=True)
                    other = encode(model, tok, [b for _, b in hard], grad=True)
                    similarity = (anchor * other).sum(dim=1)
                    loss_hard = F.relu(similarity - (1.0 - HARDNEG_MARGIN)).mean()
                    loss = loss + W_HARDNEG * loss_hard
                    totals["hardneg"] += float(loss_hard)

            if reject_rows:
                offset = (step * 32) % max(1, len(reject_rows) - 32)
                junk = reject_rows[offset : offset + 32]
                if junk:
                    junk_vectors = encode(model, tok, junk, grad=True)
                    names, centroids = build_centroids(model, tok, seeds)
                    best = (junk_vectors @ centroids.T).max(dim=1).values
                    loss_reject = F.relu(best - REJECT_CEILING).mean()
                    loss = loss + W_REJECT * loss_reject
                    totals["reject"] += float(loss_reject)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(batches)} "
                    + " ".join(f"{k}={v / max(1, step + 1):.3f}" for k, v in totals.items()),
                    flush=True,
                )

        print(f"epoch {epoch} done", flush=True)
        evaluate(model, tok, f"ep{epoch}")
        report_jargon(model, tok, f"ep{epoch}")
        report_antonyms(model, tok)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"saved -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
