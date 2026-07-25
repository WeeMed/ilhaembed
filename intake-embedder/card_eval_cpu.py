#!/usr/bin/env python3
"""Model-card eval: held-out jargon top-1, one methodology across all models, CPU-only.

SEMANTIC benchmark (the headline). Each register is held out and its surface term retrieved against
the combined canonical pool; top-1, self-match excluded, macro-averaged. All three are the same kind
of task -- a colloquial / Taigi / abbreviated Han surface form -> its standard Han concept:
  * slang      斷腦筋 -> 中風          (Taiwanese clinical colloquialism; this is where the embedder's
                                        Taigi CLINICAL capability is actually measured)
  * abbrev     Tid    -> 一天三次       (dosing / clinical abbreviation)
  * apposition URI    -> 上呼吸道感染   (apposition-mined abbreviation)

taigi_med_lexicon is reported as a DIAGNOSTIC and is deliberately NOT in the macro -- honesty over
headline. Its "gold" is a Han -> Tai-lo ROMANIZATION (中風 -> tiong-hong), i.e. a TRANSLITERATION
task, not semantic retrieval. A semantic embedder correctly scores ~0 mapping a Han term to a
romanized string; that job belongs to the ASR / romanization model, not here. Averaging a
transliteration task into a semantic benchmark would produce a meaningless number, so it is shown
separately with its reason rather than folded in. (The earlier "self-match bug" framing was wrong:
only 31/779 taigi surfaces collide with the pool and those are already masked; the ~0 is the
task mismatch, not a maskable collision.)

CPU-only on purpose (CUDA_VISIBLE_DEVICES="") so this runs while the GPU trains something else.

Note: the four `.tsv` files this script reads (`med_slang.tsv`, `abbr_dict_v1.tsv`,
`appos_pairs.tsv`, `taigi_med_lexicon.tsv`) carry third-party / institutional copyright and are
not included in this public release (see SOURCES.md). This script is published for methodology
transparency, not as a runnable-out-of-the-box artifact.
"""
import os, csv, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU; leave the GPU to whatever is training
from sentence_transformers import SentenceTransformer

D = os.path.expanduser("~/med-embed")
MODELS = [
    ("IlhaEmbed (pruned 97M)", os.path.join(D, "ilha-embed-97m-pruned")),
    ("bge-small-zh",           os.path.join(D, "intake-embedder/base-bge")),
    ("jina-v2-base-zh",        os.path.join(D, "intake-embedder/base-jina")),
    ("ckip-base",              os.path.join(D, "intake-embedder/base-ckip-base")),
]
# kind: "semantic" counts toward the macro; "translit" is a diagnostic shown but excluded.
SRC = [("slang", "med_slang.tsv", 0, 2, "semantic"),
       ("abbrev", "abbr_dict_v1.tsv", 0, 1, "semantic"),
       ("apposition", "appos_pairs.tsv", 0, 1, "semantic"),
       ("taigi (Han->romanization, transliteration)", "taigi_med_lexicon.tsv", 0, 1, "translit")]

def load(fn, sc, cc):
    out = []
    for r in csv.reader(open(os.path.join(D, fn)), delimiter="\t"):
        if len(r) > max(sc, cc) and r[sc].strip() and r[cc].strip():
            out.append((r[sc].strip(), r[cc].strip()))
    # Drop a header row when the first cell is a column label rather than a real term.
    if out and out[0][0] in ("a", "surface", "Tid", "漢字", "slang", "term"):
        out = out[1:]
    return out

regs = {name: (load(fn, sc, cc), kind) for name, fn, sc, cc, kind in SRC}
# One combined pool over every register's canonicals (kept identical to the published run so the
# semantic numbers stay directly comparable; the romanization golds simply act as extra distractors).
pool = sorted({c for pairs, _ in regs.values() for _, c in pairs})
idx = {c: i for i, c in enumerate(pool)}
print(f"combined pool={len(pool)} | " + " ".join(f"{k.split()[0]}={len(v[0])}" for k, v in regs.items()), flush=True)

def run(name, mid):
    m = SentenceTransformer(mid, device="cpu", trust_remote_code=True)
    ce = m.encode(pool, normalize_embeddings=True, show_progress_bar=False)
    print(f"\n{name}:", flush=True)
    accs = []
    for reg, (pairs, kind) in regs.items():
        if not pairs:
            continue
        qe = m.encode([a for a, _ in pairs], normalize_embeddings=True, show_progress_bar=False)
        sims = qe @ ce.T
        for i, (q, _) in enumerate(pairs):
            if q in idx:
                sims[i][idx[q]] = -1e9  # self-match exclusion
        t1 = float(np.mean([idx.get(c, -2) == int(np.argmax(sims[i])) for i, (_, c) in enumerate(pairs)]))
        if kind == "semantic":
            accs.append(t1)
            print(f"  {reg:14s} top1={t1:.3f} (n={len(pairs)})  [semantic]", flush=True)
        else:
            print(f"  {reg} top1={t1:.3f} (n={len(pairs)})  [transliteration diagnostic — NOT a semantic task, excluded from macro]", flush=True)
    print(f"  SEMANTIC macro (slang / abbrev / apposition) = {np.mean(accs):.3f}", flush=True)

for n, p in MODELS:
    try:
        run(n, p)
    except Exception as e:
        print(f"{n}: ERR {str(e)[:150]}", flush=True)
