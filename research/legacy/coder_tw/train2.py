#!/usr/bin/env python3
"""Retrain CODER-TW on the EXPANDED mix and measure TWO tracks honestly:

  specialized (中文表面 -> 中文標準): 斷腦筋->中風. Signal not in any pretrain
      corpus. Upsampled so bulk doesn't drown it.
  cross-lingual (中文 -> 英文碼名):   霍亂->Cholera. 63.5k gov ICD/LOINC pairs;
      this is the Test-B gap CODER's pretraining only partly covers.

Same seeded specialized split as train.py, so specialized numbers are directly
comparable to the 987-pair run (base 0.189 / v1 0.297). Bulk gets its own held-out.
CPU (reliable; MPS died earlier). This is a checkpoint to decide if the full 64k
run on the 4080 is worth it."""
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from train import SEED, load_pairs, encode, evaluate

D = os.path.dirname(__file__)
BASE = "GanjinZero/coder_all"
OUT = os.path.join(D, "coder-tw-v2")
DEV = "cpu"
N_BULK_TRAIN = 8000
N_BULK_TEST = 250
UPSAMPLE = 4


def load_bulk():
    rows = list(csv.DictReader(open(os.path.join(D, "bulk_pairs.tsv")), delimiter="\t"))
    pairs = [(r["zh"], r["en"]) for r in rows if r["zh"] and r["en"]]
    random.Random(SEED).shuffle(pairs)
    return pairs


def eval_xling(model, tok, test, pool):
    model.eval()
    with torch.no_grad():
        ze = encode(model, tok, [z for z, _ in test])
        pe = encode(model, tok, pool)
    sims = (ze @ pe.T).cpu().numpy()
    idx = {e: i for i, e in enumerate(pool)}
    t1 = sum(int(np.argmax(sims[i]) == idx[e]) for i, (_, e) in enumerate(test))
    return t1 / len(test)


def main():
    random.seed(SEED); torch.manual_seed(SEED)

    # specialized: identical split to train.py
    sp = load_pairs(); random.shuffle(sp)
    n_test = max(30, int(0.15 * len(sp)))
    sp_test, sp_train = sp[:n_test], sp[n_test:]
    sp_pool = sorted({c for _, c in sp})

    # bulk cross-lingual
    bulk = load_bulk()
    bk_test = bulk[:N_BULK_TEST]
    bk_train = bulk[N_BULK_TEST:N_BULK_TEST + N_BULK_TRAIN]
    bk_pool = sorted({e for _, e in bk_test})

    train = sp_train * UPSAMPLE + bk_train
    random.shuffle(train)
    print(f"train: {len(sp_train)}x{UPSAMPLE} specialized + {len(bk_train)} bulk "
          f"= {len(train)} | sp_test {len(sp_test)} | bk_test {len(bk_test)} | {DEV}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).to(DEV)

    s1, s5 = evaluate(model, tok, sp_test, sp_pool)
    x1 = eval_xling(model, tok, bk_test, bk_pool)
    print(f"[base]     specialized top1={s1:.3f} top5={s5:.3f} | xling top1={x1:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    EPOCHS, BS, TEMP = 3, 32, 0.05
    for ep in range(EPOCHS):
        model.train(); random.shuffle(train); tot = 0.0
        for i in range(0, len(train), BS):
            b = train[i:i+BS]
            if len(b) < 2:
                continue
            se = encode(model, tok, [x[0] for x in b], bs=BS)
            ce = encode(model, tok, [x[1] for x in b], bs=BS)
            logits = (se @ ce.T) / TEMP
            lab = torch.arange(len(b)).to(DEV)
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        s1, s5 = evaluate(model, tok, sp_test, sp_pool)
        x1 = eval_xling(model, tok, bk_test, bk_pool)
        print(f"  epoch {ep+1}: loss={tot:.1f}  specialized top1={s1:.3f} top5={s5:.3f}"
              f" | xling top1={x1:.3f}")

    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
