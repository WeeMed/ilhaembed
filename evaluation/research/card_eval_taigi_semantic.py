#!/usr/bin/env python3
"""Model-card eval slot: Taigi-colloquial -> standard-clinical-concept, Han -> Han only.

This is the register that `card_eval_cpu.py`'s 2026-07-24 entry names as the honest next
benchmark. `taigi_med_lexicon.tsv` (used there as a labeled transliteration DIAGNOSTIC) maps
Han -> Tai-lo ROMANIZATION (中風 -> tiong-hong): a transliteration task, not semantics, so a
meaning embedder correctly scores ~0 on it and it stays excluded from any semantic macro.

`taigi_semantic_register.tsv` (data/) is different by construction: every pair is Han -> Han,
a Taiwanese-colloquial written form mapped to the standard clinical Han concept it names
(斷腦筋 -> 中風, 胃藥 -> 制酸劑). Same combined-pool / top-1 / self-match-excluded protocol as
card_eval_cpu.py's slang/abbrev/apposition registers, so numbers are directly comparable to
those three -- unlike taigi_med_lexicon, this register belongs in the semantic macro.

Two provenance families, both retained (see data file's `provenance`/`confidence` columns):
  * itaigi_TGB       -- 台語漢字辭典 (itaigi) crowd-sourced 華語<->台語漢字 pairs, net-vote filtered
                        (confidence tier derived from net votes: high>=10, medium>=3, low>=1),
                        restricted to a hand-reviewed clinical-concept whitelist (disease/symptom/
                        body part/procedure/drug/facility/role -- proverbs, food, kinship, and
                        general vocabulary excluded even when the vote count was strong).
  * internal-narrative -- a small (7-pair) subset mined from a proprietary, non-public clinical
                        case-narrative corpus; not part of this public release (see
                        docs/REPOSITORY.md).

Known gap (documented, not silently absent): a larger, professionally curated sovereignty corpus
is the intended primary source for this register; the application to access it is still pending
review and could not be used here. See docs/research/EXPERIMENTS.md for what it will add.

CPU-only on purpose (CUDA_VISIBLE_DEVICES="") so this can run without a GPU.

Note: the `taigi_semantic_register.tsv` data file this script reads is not included in this
public release (see SOURCES.md / README.md) -- this script is published for methodology
transparency, not as a runnable-out-of-the-box artifact.
"""
import os, csv, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU; leave the GPU to whatever is training
from sentence_transformers import SentenceTransformer

D = os.path.expanduser(os.environ.get("ILHAEMBED_RESEARCH_ROOT", "~/med-embed"))
REG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "taigi_semantic_register.tsv")
MODELS = [
    ("IlhaEmbed (pruned 97M)", os.path.join(D, "ilha-embed-97m-pruned")),
    ("bge-small-zh",           os.path.join(D, "models/base-bge")),
    ("jina-v2-base-zh",        os.path.join(D, "models/base-jina")),
    ("ckip-base",              os.path.join(D, "models/base-ckip-base")),
]


def load_register(path):
    pairs = []
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            surface, canonical = row["surface"].strip(), row["canonical"].strip()
            if surface and canonical:
                pairs.append((surface, canonical, row.get("confidence", ""), row.get("provenance", "")))
    return pairs


pairs = load_register(REG)
pool = sorted({c for _, c, _, _ in pairs})
idx = {c: i for i, c in enumerate(pool)}
by_tier = {t: [p for p in pairs if p[2] == t] for t in ("high", "medium", "low")}
print(f"taigi_semantic_register: n={len(pairs)} pool={len(pool)} "
      f"tiers=high:{len(by_tier['high'])}/medium:{len(by_tier['medium'])}/low:{len(by_tier['low'])}",
      flush=True)


def run(name, mid):
    m = SentenceTransformer(mid, device="cpu", trust_remote_code=True)
    ce = m.encode(pool, normalize_embeddings=True, show_progress_bar=False)
    print(f"\n{name}:", flush=True)
    qe = m.encode([s for s, _, _, _ in pairs], normalize_embeddings=True, show_progress_bar=False)
    sims = qe @ ce.T
    for i, (s, _, _, _) in enumerate(pairs):
        if s in idx:
            sims[i][idx[s]] = -1e9  # self-match exclusion
    hits = [idx.get(c, -2) == int(np.argmax(sims[i])) for i, (_, c, _, _) in enumerate(pairs)]
    overall = float(np.mean(hits))
    print(f"  overall top1 = {overall:.3f} (n={len(pairs)})", flush=True)
    for tier in ("high", "medium", "low"):
        tp = by_tier[tier]
        if not tp:
            continue
        tier_hits = [hits[i] for i, p in enumerate(pairs) if p[2] == tier]
        print(f"  {tier:6s} top1={np.mean(tier_hits):.3f} (n={len(tp)})", flush=True)


for n, p in MODELS:
    try:
        run(n, p)
    except Exception as e:
        print(f"{n}: ERR {str(e)[:150]}", flush=True)
