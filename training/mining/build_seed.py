#!/usr/bin/env python3
"""Build the iTaigi seed = union of the Mandarin 'full name' column of all four
tables, cleaned. iTaigi's 揣列表 takes a Mandarin keyword, so we harvest the
Mandarin-side term from each table:

  abbr_dict_v1.tsv     -> zh        (Chinese full name of the abbreviation)
  med_slang.tsv        -> meaning   (Mandarin meaning of the slang)
  taigi_med_lexicon    -> 漢字       (the term itself)
  itaigi_med_lexicon   -> 華語       (already Mandarin keywords)

Cleaning: strip trailing stray numbers/spaces, drop parenthetical asides, keep
only 1-8 char CJK terms, dedupe. Non-CJK / English-only rows are dropped (an
English abbr like 'CT' is not a Mandarin search key)."""
import csv
import os
import re

D = os.path.dirname(__file__)
CJK = "一-鿿"
clean_re = re.compile(rf"[^{CJK}]")          # keep only CJK
paren_re = re.compile(r"[（(].*?[）)]")


def norm(s):
    s = paren_re.sub("", s or "")
    # cut at first non-CJK run so '心電圖 28' -> '心電圖', '就醫：X醫院' handled
    m = re.match(rf"[{CJK}]+", s.strip())
    if not m:
        return None
    term = m.group(0)
    return term if 1 <= len(term) <= 8 else None


def col(path, field):
    p = os.path.join(D, path)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = norm(row.get(field, ""))
            if t:
                out.append(t)
    return out


def main():
    src = {
        "abbr(zh)":   col("abbr_dict_v1.tsv", "zh"),
        "slang(意思)": col("med_slang.tsv", "meaning"),
        "taigi(漢字)": col("taigi_med_lexicon.tsv", "台語漢字") or col("taigi_med_lexicon.tsv", "漢字"),
        "itaigi(華語)": col("itaigi_med_lexicon.tsv", "華語"),
    }
    seen, union = set(), []
    for name, terms in src.items():
        for t in terms:
            if t not in seen:
                seen.add(t)
                union.append(t)
    with open(os.path.join(D, "union_seed.txt"), "w") as f:
        f.write("\n".join(union))
    print("per-source term counts (pre-dedup):")
    for name, terms in src.items():
        print(f"  {name:<14} {len(terms):>5}  ({len(set(terms))} unique)")
    print(f"\nunion (deduped): {len(union)} terms -> union_seed.txt")


if __name__ == "__main__":
    main()
