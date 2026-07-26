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
import re
import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--description",
        type=Path,
        required=True,
        help="licensed SNOMED description.csv.gz export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("snomed_syn_pairs.tsv"),
    )
    args = parser.parse_args()

    concepts = {}  # concept_id -> {"fsn": str, "syns": [str]}
    with gzip.open(args.description, "rt") as f:
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["surface", "canonical", "source"])
        w.writerows((s, c, "snomed-syn") for s, c in pairs)
    print(f"concepts: {len(concepts)} | high-signal synonym pairs: {len(pairs)}")
    for s, c in pairs[:12]:
        print(f"  {s}  ->  {c}")


if __name__ == "__main__":
    main()
