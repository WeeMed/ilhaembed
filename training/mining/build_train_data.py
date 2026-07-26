#!/usr/bin/env python3
"""Merge ALL mined sources into two training files for train_gpu.py.

specialized_pairs.tsv (a,b)  -- surface -> canonical, the hard/scarce signal:
   unified lexicon (colloquial/slang/abbr/taigi) + wiki redirects + wiki
   apposition + TW-medical-crawl apposition.
bulk_all.tsv (a,b)           -- formal synonym bulk:
   ICD/LOINC cross-lingual (zh<->en) + SNOMED English synonyms.
"""
import csv
import os
import re

D = os.path.dirname(os.path.abspath(__file__))
CJK = "一-鿿"
_frag = re.compile(rf"^[之的有是及並]")


def clean(c):
    c = re.split(r"[（(]", c)[0]
    c = re.sub(r"\s+\d+\s*$", "", c)
    c = re.sub(r"\s+", "", c).strip()
    if not re.match(rf"[{CJK}A-Za-z]", c) or len(c) < 2 or _frag.match(c):
        return None
    return c


def rd(name):
    p = os.path.join(D, name)
    return list(csv.DictReader(open(p), delimiter="\t")) if os.path.exists(p) else []


def main():
    spec, seen = [], set()

    def add(a, b):
        a = (a or "").strip(); b = clean(b or "")
        if a and b and a != b and (a, b) not in seen:
            seen.add((a, b)); spec.append((a, b))

    for r in rd("unified_med_lexicon.tsv"):
        add(r["surface"], r["canonical_zh"])
    for name in ("wiki_pairs.tsv", "appos_pairs.tsv", "tw_med_pairs.tsv"):
        for r in rd(name):
            add(r["surface"], r["canonical"])

    with open(os.path.join(D, "specialized_pairs.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["a", "b"]); w.writerows(spec)

    bulk, bseen = [], set()
    for r in rd("bulk_pairs.tsv"):
        k = (r["zh"], r["en"])
        if r["zh"] and r["en"] and k not in bseen:
            bseen.add(k); bulk.append(k)
    for r in rd("snomed_syn_pairs.tsv"):
        k = (r["surface"], r["canonical"])
        if r["surface"] and r["canonical"] and k not in bseen:
            bseen.add(k); bulk.append(k)

    with open(os.path.join(D, "bulk_all.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["a", "b"]); w.writerows(bulk)

    print(f"specialized_pairs.tsv: {len(spec)}  (was 987 usable)")
    print(f"bulk_all.tsv:          {len(bulk)}  (ICD/LOINC + SNOMED-syn)")


if __name__ == "__main__":
    main()
