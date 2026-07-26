"""Emit FDA ingredient groups as surface-to-canonical pairs.

The group-based objective failed to learn here across three attempts, while InfoNCE over pairs
reaches 0.364 on the same data. So the FDA structure is expressed in the shape the working method
consumes: every licensed product name becomes a surface form of its active ingredient."""
import csv, gzip, sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "codesystems/fda_drugs.csv.gz")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "../data/fda_pairs.tsv")

groups: dict[str, set[str]] = {}
with gzip.open(src, "rt", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        ingredient = (row.get("generic_name") or "").strip()
        if not ingredient or len(ingredient) > 120:
            continue
        for field in ("trade_name", "trade_name_en"):
            surface = (row.get(field) or "").strip()
            if surface and 1 < len(surface) <= 40:
                groups.setdefault(ingredient, set()).add(surface)

pairs = [(surface, ingredient) for ingredient, surfaces in groups.items() for surface in surfaces]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(("a", "b"))
    writer.writerows(pairs)
print(f"{len(groups)} ingredient concepts -> {len(pairs)} surface pairs -> {out}")
