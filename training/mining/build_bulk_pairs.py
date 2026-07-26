#!/usr/bin/env python3
"""Extract bulk cross-lingual synonym pairs from licensed terminology tables.

Each code's (display_zh, display_en) is a synonym positive for the same concept
-- the cross-lingual signal needed for the Chinese-fragment to
English-code-display gap. Obtain each input under its own licence; this script
does not grant redistribution rights.

Output: bulk_pairs.tsv  (zh<TAB>en<TAB>source)
"""
import argparse
import csv
import gzip
import re
from pathlib import Path

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing the licensed terminology .csv.gz files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("bulk_pairs.tsv"),
    )
    args = parser.parse_args()

    seen, rows = set(), []
    per = {}
    for fname, tag in TABLES:
        path = args.data_dir / fname
        if not path.exists():
            continue
        n = 0
        with gzip.open(path, "rt") as f:
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["zh", "en", "source"])
        w.writerows(rows)

    print("cross-lingual pairs per table:")
    for tag, n in per.items():
        print(f"  {tag:<10} {n}")
    print(f"total unique: {len(rows)} -> {args.output}")
    print("samples:")
    for zh, en, tag in rows[:6]:
        print(f"  {zh}  <->  {en}  [{tag}]")


if __name__ == "__main__":
    main()
