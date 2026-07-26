#!/usr/bin/env python3
"""Shrink an embedding model by dropping vocabulary the domain never uses.

A general-purpose Chinese model carries a vocabulary sized for general Chinese. This one is a
single-domain model, and the embedding matrix is the largest single block of parameters in it --
for jina-v2-base-zh, 46M of 132M. Measured over the whole domain corpus, only ~21% of its rows are
ever touched, so the rest are paid for on every load and never read.

Pruning is lossless for text built from the kept tokens. It is NOT lossless in general: a character
outside the kept set becomes [UNK], and an [UNK] carries no meaning at all. So the kept set is
deliberately wider than "what the corpus used":

1. every token the domain corpus touches -- the tokens that actually carry the work;
2. every single-character token in the model's vocabulary -- the tokenizer's own fallback path, so an
   unseen WORD still decomposes into characters that mean something instead of collapsing to [UNK].

(2) is what makes this safe to ship into healthcare. Dropping it would buy a few more megabytes and
pay for them with silent failures on exactly the rare terms a clinical system must not lose.

The tokenizer is REBUILT over the kept tokens, not merely saved alongside them. Slicing the matrix
while keeping the original tokenizer produces a model that loads, runs, and returns embeddings for
the wrong tokens -- a total corruption with no symptom. The verification at the end is therefore not
optional decoration: it re-encodes real domain strings through both models and fails loudly unless
they agree.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ILHAEMBED_TRAIN_DATA", str(HERE / "data")))
BASE = os.environ.get("BASE_MODEL", "jinaai/jina-embeddings-v2-base-zh")
OUT = Path(os.environ.get("OUT_DIR", str(HERE / "jina-zh-pruned")))

CORPUS = [
    ("category.tsv", ["text"]),
    ("synonyms.tsv", ["a", "b"]),
    ("hardneg.tsv", ["anchor", "negative"]),
    ("negatives.tsv", ["text"]),
    ("colloquial.tsv", ["a", "b"]),
]
# Verification strings span what the model must keep reading correctly: clinical terms, abbreviated
# institutions, working phrases, mixed Latin, and a deliberately unseen rare character.
VERIFY = [
    "糖化血色素", "健康關懷站", "嘉基", "戒菸衛教", "白血球表面標記", "低血壓",
    "定期心內門診-戒菸", "U.R.I.", "CCU", "不辣咖", "顳顎關節疼痛", "膽囊切除術",
]


def domain_texts() -> list[str]:
    texts: list[str] = []
    for name, columns in CORPUS:
        path = DATA / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                texts.extend(row[c] for c in columns if row.get(c))
    return texts


def embed(model, tok, texts: list[str]) -> torch.Tensor:
    ids = tok(texts, max_length=32, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        hidden = model(**ids).last_hidden_state[:, 0]
    return F.normalize(hidden, dim=1)


def main() -> int:
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if not isinstance(tok, BertTokenizerFast):
        print(f"unsupported tokenizer {type(tok).__name__}: this prune rebuilds a WordPiece vocab only")
        return 1

    model = AutoModel.from_pretrained(BASE, trust_remote_code=True)
    model.eval()
    vocab = tok.get_vocab()
    inverse = {index: token for token, index in vocab.items()}

    keep: set[int] = set(tok.all_special_ids)
    for text in domain_texts():
        keep.update(tok(text, add_special_tokens=False)["input_ids"])
    touched = len(keep)
    for index, token in inverse.items():
        if len(token.removeprefix("##")) == 1:
            keep.add(index)

    kept = sorted(keep)
    print(f"vocab {len(inverse)} -> kept {len(kept)}  "
          f"(corpus touched {touched}, +{len(kept) - touched} single-char fallback)")

    baseline = embed(model, tok, VERIFY)
    before = sum(p.numel() for p in model.parameters())

    embeddings = model.get_input_embeddings()
    pruned = torch.nn.Embedding(len(kept), embeddings.weight.shape[1])
    with torch.no_grad():
        pruned.weight.copy_(embeddings.weight[kept])
    model.set_input_embeddings(pruned)
    model.config.vocab_size = len(kept)
    after = sum(p.numel() for p in model.parameters())
    print(f"params {before / 1e6:.1f}M -> {after / 1e6:.1f}M  ({1 - after / before:.1%} smaller)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vocab.txt").write_text(
        "\n".join(inverse[index] for index in kept) + "\n", encoding="utf-8"
    )
    new_tok = BertTokenizerFast(
        vocab_file=str(OUT / "vocab.txt"),
        do_lower_case=getattr(tok, "do_lower_case", False),
    )
    new_tok.save_pretrained(OUT)
    model.save_pretrained(OUT, safe_serialization=True)

    check = embed(model, new_tok, VERIFY)
    similarity = (baseline * check).sum(dim=1)
    worst = float(similarity.min())
    print("\nverification (original vs pruned, cosine per string):")
    for text, score in zip(VERIFY, similarity.tolist()):
        flag = "" if score > 0.999 else "   <-- DIVERGED"
        print(f"  {score:.5f}  {text}{flag}")
    if worst <= 0.999:
        print(f"\nFAILED: worst cosine {worst:.5f}. The pruned model does not reproduce the original.")
        return 1
    print(f"\nOK: worst cosine {worst:.5f}; pruned model reproduces the original on every probe.")
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
