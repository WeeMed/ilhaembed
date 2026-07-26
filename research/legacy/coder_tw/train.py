#!/usr/bin/env python3
"""Light contrastive domain-adaptation of CODER on the TW clinical surface
lexicon. surface (what staff say/write) and canonical_zh (standard term) form a
synonym positive; in-batch other canonicals are negatives (InfoNCE / MNRL).

Order per the plan: FINE-TUNE here (fp32/MPS) -> quantize int8 ONNX separately.

Honest guards:
  * clean the PDF-noisy abbr canonicals (strip trailing numbers/parens/fragments)
  * hold out 15% of pairs for eval the model never trains on
  * deterministic split (seeded) so before/after is comparable
"""
import csv
import os
import re
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

D = os.path.dirname(__file__)
BASE = "GanjinZero/coder_all"
OUT = os.path.join(D, "coder-tw")
DEV = os.environ.get("HYG_DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

CJK = "一-鿿"
_frag = re.compile(rf"^[之的有是及並]")  # canonical starting like a sentence fragment


def clean_canonical(c):
    c = re.split(r"[（(]", c)[0]           # drop from first paren
    c = re.sub(r"\s+\d+\s*$", "", c)        # trailing stray number (PDF column bleed)
    c = re.sub(r"\s+", "", c).strip()
    if not re.match(rf"[{CJK}A-Za-z]", c):
        return None
    if len(c) < 2 or _frag.match(c):
        return None
    return c


def load_pairs():
    rows = list(csv.DictReader(open(os.path.join(D, "unified_med_lexicon.tsv")),
                               delimiter="\t"))
    pairs = []
    for r in rows:
        s = r["surface"].strip()
        c = clean_canonical(r["canonical_zh"])
        if s and c and s != c:
            pairs.append((s, c))
    # dedupe
    seen, uniq = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def encode(model, tok, texts, bs=64):
    vecs = []
    for i in range(0, len(texts), bs):
        ids = tok(texts[i:i+bs], max_length=32, truncation=True,
                  padding="max_length", return_tensors="pt").to(DEV)
        out = model(**ids)[1]              # CLS pooler (CODER convention)
        vecs.append(F.normalize(out, dim=1))
    return torch.cat(vecs)


@torch.no_grad()
def evaluate(model, tok, test, canon_pool):
    """For each held-out (surface, canonical): rank the true canonical among the
    full canonical pool. Report top-1 / top-5."""
    model.eval()
    surf = [s for s, _ in test]
    gold = [c for _, c in test]
    se = encode(model, tok, surf)
    ce = encode(model, tok, canon_pool)
    sims = se @ ce.T
    idx = {c: i for i, c in enumerate(canon_pool)}
    top1 = top5 = 0
    for i, g in enumerate(gold):
        order = torch.argsort(sims[i], descending=True)
        rank = (order == idx[g]).nonzero().item()
        top1 += rank == 0
        top5 += rank < 5
    return top1 / len(test), top5 / len(test)


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    pairs = load_pairs()
    random.shuffle(pairs)
    n_test = max(30, int(0.15 * len(pairs)))
    test, train = pairs[:n_test], pairs[n_test:]
    canon_pool = sorted({c for _, c in pairs})
    print(f"pairs: {len(pairs)} usable | train {len(train)} | test {len(test)} "
          f"| canonical pool {len(canon_pool)} | device {DEV}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).to(DEV)

    b, a = evaluate(model, tok, test, canon_pool)
    print(f"[base CODER]   held-out top1={b:.3f} top5={a:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    EPOCHS, BS, TEMP = 4, 32, 0.05
    for ep in range(EPOCHS):
        model.train(); random.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), BS):
            batch = train[i:i+BS]
            if len(batch) < 2:
                continue
            s = [x[0] for x in batch]; c = [x[1] for x in batch]
            se = encode(model, tok, s, bs=BS)
            ce = encode(model, tok, c, bs=BS)
            logits = (se @ ce.T) / TEMP
            labels = torch.arange(len(batch)).to(DEV)
            loss = (F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        t1, t5 = evaluate(model, tok, test, canon_pool)
        print(f"  epoch {ep+1}: loss={tot:.3f}  held-out top1={t1:.3f} top5={t5:.3f}")

    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print(f"saved fine-tuned model -> {OUT}")


if __name__ == "__main__":
    main()
