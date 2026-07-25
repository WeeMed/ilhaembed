#!/usr/bin/env python3
"""Extract bulk cross-lingual synonym pairs from hygieia's own authoritative
terminology tables (gov ICD-10-CM/PCS, LOINC-NHI, SNOMED zh subset).

Each code's (display_zh, display_en) is a synonym positive for the same concept
-- exactly the cross-lingual signal CODER needs for the Chinese-fragment ->
English-code-display gap. No scraping, no license issue (gov + existing DB).

Output: bulk_pairs.tsv  (zh<TAB>en<TAB>source)
"""
import csv
import gzip
import os
import re

DATA = os.path.expanduser("~/Workspace/weemed-ai/hygieia/core/data")
OUT = os.path.join(os.path.dirname(__file__), "bulk_pairs.tsv")
CJK = re.compile(r"[一-鿿]")

TABLES = [
    ("icd10_cm_2023.csv.gz", "icd10cm"),
    ("icd10_pcs_2023.csv.gz", "icd10pcs"),
    ("loinc_nhi.csv.gz", "loinc"),
    ("snomed_ct.csv.gz", "snomed"),
]


def ok(zh, en):
    if not zh or not en:
        return False
    if not CJK.search(zh):          # zh side must have Chinese
        return False
    if len(zh) > 40 or len(en) > 60:  # drop pathologically long formal strings
        return False
    return True


def main():
    seen, rows = set(), []
    per = {}
    for fname, tag in TABLES:
        p = os.path.join(DATA, fname)
        if not os.path.exists(p):
            continue
        n = 0
        with gzip.open(p, "rt") as f:
            for r in csv.DictReader(f):
                zh = (r.get("display_zh") or "").strip()
                en = (r.get("display") or "").strip()
                if ok(zh, en):
                    k = (zh, en)
                    if k not in seen:
                        seen.add(k)
                        rows.append((zh, en, tag))
                        n += 1
        per[tag] = n

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["zh", "en", "source"])
        w.writerows(rows)

    print("cross-lingual pairs per table:")
    for tag, n in per.items():
        print(f"  {tag:<10} {n}")
    print(f"total unique: {len(rows)} -> bulk_pairs.tsv")
    print("samples:")
    for zh, en, tag in rows[:6]:
        print(f"  {zh}  <->  {en}  [{tag}]")


if __name__ == "__main__":
    main()
