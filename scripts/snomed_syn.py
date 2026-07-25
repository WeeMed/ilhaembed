#!/usr/bin/env python3
"""Extract English synonym pairs from the local SNOMED description table.

Each concept has one FSN (Fully Specified Name, type 900000000000003001) plus
synonyms (type 900000000000013009). synonym -> FSN(tag stripped) is a
surface->canonical pair: teaches the model that abbreviations / lay terms /
alt spellings denote the same concept (MI -> Myocardial infarction).

We prioritize the HIGH-SIGNAL surface forms (short / abbreviation-like), not all
1.37M descriptions -- bulk formal terms already saturated cross-lingual; what's
useful here is the abbreviation<->term axis.

Output: snomed_syn_pairs.tsv (surface<TAB>canonical<TAB>source)
"""
import csv
import gzip
import os
import re

DESC = os.path.expanduser(
    "~/Workspace/weemed-ai/hygieia/core/data/terminology_sources/snomed/description.csv.gz")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snomed_syn_pairs.tsv")
FSN = "900000000000003001"
SYN = "900000000000013009"
_tag = re.compile(r"\s*\([^)]*\)\s*$")   # trailing "(disorder)" etc.


def strip_tag(t):
    return _tag.sub("", t).strip()


def is_high_signal(surf, canon):
    # abbreviation-like (short, has caps) OR clearly shorter lay/alt form
    if surf == canon:
        return False
    if len(surf) <= 6 and re.search(r"[A-Z]", surf):     # MI, COPD, CABG
        return True
    if len(surf) < len(canon) * 0.6 and len(surf) >= 3:   # notably shorter alt
        return True
    return False


def main():
    concepts = {}  # concept_id -> {"fsn": str, "syns": [str]}
    with gzip.open(DESC, "rt") as f:
        for r in csv.DictReader(f):
            cid = r["concept_id"]; term = r["term"]; typ = r["type_id"]
            c = concepts.setdefault(cid, {"fsn": None, "syns": []})
            if typ == FSN:
                c["fsn"] = term
            elif typ == SYN:
                c["syns"].append(term)

    pairs, seen = [], set()
    for c in concepts.values():
        if not c["fsn"]:
            continue
        canon = strip_tag(c["fsn"])
        if not (2 <= len(canon) <= 60):
            continue
        for s in c["syns"]:
            s = s.strip()
            if is_high_signal(s, canon) and (s, canon) not in seen:
                seen.add((s, canon)); pairs.append((s, canon))

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["surface", "canonical", "source"])
        w.writerows((s, c, "snomed-syn") for s, c in pairs)
    print(f"concepts: {len(concepts)} | high-signal synonym pairs: {len(pairs)}")
    for s, c in pairs[:12]:
        print(f"  {s}  ->  {c}")


if __name__ == "__main__":
    main()
