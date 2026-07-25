#!/usr/bin/env python3
"""Full-scale CODER-TW fine-tune on the RTX 4080 (self-contained, no local imports).

Data (transferred alongside):
  bulk_pairs.tsv           63.5k gov cross-lingual pairs (zh<->en)
  unified_med_lexicon.tsv  specialized surface->canonical (colloquial/slang/abbr)

Mix: specialized upsampled (so 63k bulk doesn't drown it) + all bulk.
InfoNCE / in-batch negatives, CLS pooler (CODER convention). Two eval tracks
(specialized zh->zh, cross-lingual zh->en) on seeded held-out sets."""
import csv
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast

D = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(D, "coder_all_st")  # local safetensors (avoids remote torch<2.6 .bin block)
OUT = os.path.join(D, "coder-tw-gpu")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
CJK = "一-鿿"
_frag = re.compile(rf"^[之的有是及並]")


def clean_canonical(c):
    c = re.split(r"[（(]", c)[0]
    c = re.sub(r"\s+\d+\s*$", "", c)
    c = re.sub(r"\s+", "", c).strip()
    if not re.match(rf"[{CJK}A-Za-z]", c) or len(c) < 2 or _frag.match(c):
        return None
    return c


def load_specialized():
    rows = list(csv.DictReader(open(os.path.join(D, "specialized_pairs.tsv")), delimiter="\t"))
    return [(r["a"], r["b"]) for r in rows if r["a"] and r["b"]]


def load_bulk():
    rows = list(csv.DictReader(open(os.path.join(D, "bulk_all.tsv")), delimiter="\t"))
    p = [(r["a"], r["b"]) for r in rows if r["a"] and r["b"]]
    random.Random(SEED).shuffle(p)
    return p


def encode(model, tok, texts, bs=128):
    vs = []
    for i in range(0, len(texts), bs):
        ids = tok(texts[i:i+bs], max_length=32, truncation=True,
                  padding="max_length", return_tensors="pt").to(DEV)
        with torch.no_grad():
            v = model(**ids)[1]
        vs.append(F.normalize(v, dim=1))
    return torch.cat(vs)


def topk_acc(model, tok, test, pool, ks=(1, 5)):
    model.eval()
    se = encode(model, tok, [a for a, _ in test])
    ce = encode(model, tok, pool)
    sims = (se @ ce.T).cpu().numpy()
    idx = {c: i for i, c in enumerate(pool)}
    res = {}
    for k in ks:
        hit = 0
        for i, (_, g) in enumerate(test):
            order = np.argsort(-sims[i])[:k]
            hit += idx[g] in order
        res[k] = hit / len(test)
    return res


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    sp = load_specialized(); random.shuffle(sp)
    n_test = max(30, int(0.15 * len(sp)))
    sp_test, sp_train = sp[:n_test], sp[n_test:]
    sp_pool = sorted({c for _, c in sp})

    bulk = load_bulk()
    N_BK_TEST = 500
    bk_test = bulk[:N_BK_TEST]; bk_train = bulk[N_BK_TEST:]
    bk_pool = sorted({e for _, e in bk_test})

    UP = 8
    train = sp_train * UP + bk_train
    random.shuffle(train)
    print(f"device={DEV} | train={len(train)} ({len(sp_train)}x{UP} sp + {len(bk_train)} bulk) "
          f"| sp_test={len(sp_test)} bk_test={len(bk_test)}", flush=True)

    tok = BertTokenizerFast.from_pretrained(BASE)
    model = BertModel.from_pretrained(BASE).to(DEV)

    def report(tag):
        s = topk_acc(model, tok, sp_test, sp_pool)
        x = topk_acc(model, tok, bk_test, bk_pool)
        print(f"[{tag}] specialized top1={s[1]:.3f} top5={s[5]:.3f} | "
              f"xling top1={x[1]:.3f} top5={x[5]:.3f}", flush=True)

    report("base")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    EPOCHS, BS, TEMP = 4, 128, 0.05
    for ep in range(EPOCHS):
        model.train(); random.shuffle(train); tot = 0.0
        for i in range(0, len(train), BS):
            b = train[i:i+BS]
            if len(b) < 2:
                continue
            ids_a = tok([x[0] for x in b], max_length=32, truncation=True,
                        padding="max_length", return_tensors="pt").to(DEV)
            ids_c = tok([x[1] for x in b], max_length=32, truncation=True,
                        padding="max_length", return_tensors="pt").to(DEV)
            se = F.normalize(model(**ids_a)[1], dim=1)
            ce = F.normalize(model(**ids_c)[1], dim=1)
            logits = (se @ ce.T) / TEMP
            lab = torch.arange(len(b), device=DEV)
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"epoch {ep+1} loss={tot:.1f}", flush=True); report(f"ep{ep+1}")

    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
