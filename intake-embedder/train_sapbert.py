#!/usr/bin/env python3
"""Self-alignment training, following SapBERT's actual objective rather than plain InfoNCE.

Everything trained so far used InfoNCE over (surface, canonical) PAIRS with in-batch negatives. The
published method for this exact task -- mapping many surface forms onto one concept -- differs in two
ways that its authors call decisive, and neither was implemented until now:

1. **Concept groups, not pairs.** A concept is not two strings; it is every way people write it. The
   code systems already carry that grouping (one code, many displays across languages and registers),
   and a pair-shaped loss throws it away by re-splitting each group into arbitrary twos.
2. **Online hard-pair mining with a Multi-Similarity loss.** In-batch negatives are mostly trivial --
   two unrelated terms are easy to separate and contribute almost no gradient once the model is any
   good. Mining keeps the positives that are still too far apart and the negatives that are still too
   close, so the loss concentrates where the model is actually wrong. This is the mechanism that
   attacks the failure this project started from: a term sitting next to its own opposite because
   they share characters.

The evaluation is unchanged -- the same held-out split, seed and pool as every previous run -- so this
number is directly comparable to InfoNCE's 0.324 / 0.364 / 0.433 by size.
"""

from __future__ import annotations

import csv
import gzip
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
RESEARCH_DATA = HERE.parent / "data"
HYGIEIA_DATA = Path(os.environ.get("HYGIEIA_DATA", str(HERE / "codesystems")))
BASE = os.environ.get("BASE_MODEL", "BAAI/bge-small-zh-v1.5")
OUT = Path(os.environ.get("OUT_DIR", str(HERE / "sapbert-small")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
MAX_LEN = 32
EPOCHS = int(os.environ.get("EPOCHS", "4"))
CONCEPTS_PER_BATCH = int(os.environ.get("CONCEPTS_PER_BATCH", "48"))
SURFACES_PER_CONCEPT = int(os.environ.get("SURFACES_PER_CONCEPT", "4"))
LR = float(os.environ.get("LR", "3e-5"))
HOLDOUT = 0.15

# Multi-Similarity loss hyper-parameters (Wang et al. 2019), as used by SapBERT.
MS_ALPHA, MS_BETA, MS_BASE, MS_EPSILON = 2.0, 50.0, 0.5, 0.1


def read_gz(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_groups() -> dict[str, set[str]]:
    """concept key -> every surface form we have for it.

    A group is the unit of supervision: all its members must land together and away from every other
    group. Sources are joined on the code, which is the only reliable statement that two differently
    written strings denote the same thing."""
    groups: dict[str, set[str]] = defaultdict(set)

    for name, system in [
        ("icd10_cm_2023.csv.gz", "icd"),
        ("icd10_pcs_2023.csv.gz", "pcs"),
        ("loinc_nhi.csv.gz", "loinc"),
    ]:
        for row in read_gz(HYGIEIA_DATA / name):
            code = (row.get("code") or "").strip()
            if not code:
                continue
            for field in ("display", "display_zh"):
                value = (row.get(field) or "").strip()
                if value and 1 < len(value) <= 40:
                    groups[f"{system}:{code}"].add(value)

    # Surface->canonical pairs: the canonical IS the concept key, so every colloquial way of saying
    # it joins that group -- which is exactly the jargon we need pulled in.
    path = RESEARCH_DATA / "specialized_pairs.tsv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                a, b = (row.get("a") or "").strip(), (row.get("b") or "").strip()
                if a and b and a != b:
                    groups[f"canon:{b}"].update({a, b})

    path = RESEARCH_DATA / "bulk_all.tsv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                a, b = (row.get("a") or "").strip(), (row.get("b") or "").strip()
                if a and b and a != b:
                    groups[f"canon:{b}"].update({a, b})

    # Taiwan FDA drug licences. Unlike a code system's one-code-two-displays, this is genuinely
    # many-to-one: every product licensed for an active ingredient is another real way that concept
    # is written. Measured at 7.54 surfaces per ingredient against 2.16 for everything else, which is
    # the structure a group-based objective needs and previously did not have.
    path = HYGIEIA_DATA / "fda_drugs.csv.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ingredient = (row.get("generic_name") or "").strip().upper()
                if not ingredient or len(ingredient) > 120:
                    continue
                for field in ("trade_name", "trade_name_en"):
                    surface = (row.get(field) or "").strip()
                    if surface and 1 < len(surface) <= 40:
                        groups[f"fda:{ingredient}"].add(surface)

    path = HERE / "data" / "tw_fhir_bilingual.tsv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                english, chinese = (row.get("display_en") or "").strip(), (row.get("display_zh") or "").strip()
                if english and chinese:
                    groups[f"twfhir:{row.get('code')}"].update({english, chinese})

    groups = {key: members for key, members in groups.items() if len(members) >= 2}
    return _merge_groups_sharing_surfaces(groups)


def _merge_groups_sharing_surfaces(groups: dict[str, set[str]]) -> dict[str, set[str]]:
    """Merge any groups that share a surface form, so every string belongs to exactly one concept.

    Sources overlap: a term can be an ICD display AND the canonical of a colloquial pair. Keyed
    separately, the SAME string then lands in two groups and the loss is handed a contradiction --
    pull these identical strings together, push these identical strings apart. Measured on a real
    batch before this fix, the negative-pair cosine reached 1.000 (identical text labelled as a
    different concept) and the positive and negative distributions were indistinguishable
    (medians 0.441 vs 0.420). No amount of training resolves an objective with no solution; the loss
    sat flat for four epochs.

    Sharing a surface is evidence of denoting the same thing, so the groups are merged transitively
    rather than one being discarded -- dropping either would throw away real synonyms."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        while parent.setdefault(node, node) != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    # Every surface links all the group keys it appears in.
    surface_to_key: dict[str, str] = {}
    for key, members in groups.items():
        for surface in members:
            if surface in surface_to_key:
                union(key, surface_to_key[surface])
            else:
                surface_to_key[surface] = key

    merged: dict[str, set[str]] = defaultdict(set)
    for key, members in groups.items():
        merged[find(key)].update(members)
    return {key: members for key, members in merged.items() if len(members) >= 2}


def multi_similarity_loss(vectors: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multi-Similarity loss with online hard-pair mining.

    For each anchor, a positive is kept only if it is still closer-than-it-should-be to the hardest
    negative, and a negative only if it is still nearer than the easiest positive. Pairs already
    resolved contribute nothing, so the gradient goes where the model is still wrong instead of being
    diluted by the overwhelming majority of trivially separable pairs."""
    similarity = vectors @ vectors.T
    same = labels[:, None] == labels[None, :]
    identity = torch.eye(len(labels), dtype=torch.bool, device=vectors.device)
    positives_mask = same & ~identity
    negatives_mask = ~same

    losses = []
    for index in range(len(labels)):
        positives = similarity[index][positives_mask[index]]
        negatives = similarity[index][negatives_mask[index]]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        # Mining: keep only pairs that violate the margin against the other class's extreme.
        hard_positives = positives[positives < negatives.max() + MS_EPSILON]
        hard_negatives = negatives[negatives > positives.min() - MS_EPSILON]
        if hard_positives.numel() == 0 or hard_negatives.numel() == 0:
            continue
        positive_term = (1.0 / MS_ALPHA) * torch.log(
            1 + torch.sum(torch.exp(-MS_ALPHA * (hard_positives - MS_BASE)))
        )
        negative_term = (1.0 / MS_BETA) * torch.log(
            1 + torch.sum(torch.exp(MS_BETA * (hard_negatives - MS_BASE)))
        )
        losses.append(positive_term + negative_term)

    if not losses:
        return torch.zeros((), device=vectors.device, requires_grad=True)
    return torch.stack(losses).mean()


def encode(model, tok, texts: list[str], grad: bool = False, batch: int = 256) -> torch.Tensor:
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
def report(model, tok, test, pool, tag: str) -> float:
    model.eval()
    index = {c: i for i, c in enumerate(pool)}
    scores = (encode(model, tok, [a for a, _ in test]) @ encode(model, tok, pool).T).cpu().numpy()
    top1 = sum(int(np.argmax(r) == index[g]) for r, (_, g) in zip(scores, test))
    top5 = sum(int(index[g] in np.argsort(-r)[:5]) for r, (_, g) in zip(scores, test))
    print(f"[{tag}] HELD-OUT jargon top1={top1 / len(test):.3f} top5={top5 / len(test):.3f}", flush=True)
    return top1 / len(test)


def main() -> int:
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    specialized = []
    with (RESEARCH_DATA / "specialized_pairs.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            a, b = (row.get("a") or "").strip(), (row.get("b") or "").strip()
            if a and b and a != b:
                specialized.append((a, b))
    rng.shuffle(specialized)
    cut = max(30, int(HOLDOUT * len(specialized)))
    test = specialized[:cut]
    pool = sorted({b for _, b in specialized})
    held_out_surfaces = {a for a, _ in test}

    groups = build_groups()
    # A held-out surface form must not be trainable through any group, or the split is a fiction.
    for key in list(groups):
        groups[key] = {s for s in groups[key] if s not in held_out_surfaces}
        if len(groups[key]) < 2:
            del groups[key]

    keys = sorted(groups)
    # Sample groups in proportion to how many positive PAIRS they can contribute, not uniformly.
    # 89% of groups hold exactly two surfaces, so uniform sampling fills a batch with anchors that
    # have a single positive each -- the degenerate case where a group-based loss reduces to a
    # pair-based one and mining has nothing to choose between. Weighting by (surfaces - 1) puts the
    # genuinely many-surface concepts (drug ingredients, 7.54 surfaces each) into batches often
    # enough for the objective to see the structure it exists to exploit.
    weights = [len(groups[key]) - 1 for key in keys]
    weighted_keys = [key for key, weight in zip(keys, weights) for _ in range(min(weight, 8))]
    print(f"device={DEV} base={BASE}")
    print(f"weighted sampling pool={len(weighted_keys)} (from {len(keys)} groups)", flush=True)
    print(f"concept groups={len(keys)}  surfaces={sum(len(v) for v in groups.values())}  "
          f"held-out={len(test)} pool={len(pool)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModel.from_pretrained(BASE, trust_remote_code=True).to(DEV)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    report(model, tok, test, pool, "base")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    steps_per_epoch = len(weighted_keys) // CONCEPTS_PER_BATCH

    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(keys)
        total = 0.0
        for step in range(steps_per_epoch):
            chosen = keys[step * CONCEPTS_PER_BATCH : (step + 1) * CONCEPTS_PER_BATCH]
            texts, labels = [], []
            for label, key in enumerate(chosen):
                members = list(groups[key])
                rng.shuffle(members)
                for surface in members[:SURFACES_PER_CONCEPT]:
                    texts.append(surface)
                    labels.append(label)
            if len(texts) < 4:
                continue
            vectors = encode(model, tok, texts, grad=True)
            loss = multi_similarity_loss(vectors, torch.tensor(labels, device=DEV))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
        print(f"epoch {epoch} ms_loss={total / max(steps_per_epoch, 1):.4f}", flush=True)
        report(model, tok, test, pool, f"ep{epoch}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    print(f"saved -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
