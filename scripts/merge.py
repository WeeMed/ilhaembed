#!/usr/bin/env python3
"""Merge the four harvested tables into ONE unified lexicon.

Schema -- every row is a (surface -> canonical) pair, which serves BOTH:
  * ASR post-processing: map what staff actually say/write -> the standard term
  * embedder training:   surface & canonical are a synonym pair (SapBERT/CODER
                         contrastive positive), so this file is also the
                         fine-tune set for an open-sourceable medical embedder.

columns:
  surface        form as spoken/written by staff (空喙 / 摸咪 / CT / Endo)
  reading_tai_lo Tai-lo romanization when Taigi, else ''
  canonical_zh   standard Mandarin term (傷口 / 晨會 / 電腦斷層 / 氣管內管)
  canonical_en   standard English/source term when any, else ''
  type           taigi_reading | slang | jargon | abbr
  votes          iTaigi crowd net votes, else ''
  source         provenance
  tier           1 authoritative(MOE/gov/committee) 2 crowd/hospital 3 community
"""
import csv
import os

D = os.path.dirname(__file__)
COLS = ["surface", "reading_tai_lo", "canonical_zh", "canonical_en",
        "type", "votes", "source", "tier"]


def rd(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    out = []

    # 1. iTaigi crowd readings: 華語 -> 台語漢字 (台羅), with votes
    for r in rd("itaigi_med_lexicon_full.tsv"):
        out.append(dict(surface=r["台語漢字"], reading_tai_lo=r["台羅"],
                        canonical_zh=r["華語"], canonical_en="",
                        type="taigi_reading", votes=r["淨票"],
                        source=f"itaigi:{r['貢獻者']}", tier=2))

    # 2. MOE authoritative Taigi dictionary: headword + Tai-lo
    for r in rd("taigi_med_lexicon.tsv"):
        out.append(dict(surface=r["漢字"], reading_tai_lo=r["台羅"],
                        canonical_zh=r["漢字"], canonical_en="",
                        type="taigi_reading", votes="",
                        source="MOE-twblg", tier=1))

    # 3. slang / jargon: surface slang -> Mandarin meaning + English origin
    for r in rd("med_slang.tsv"):
        typ = "slang" if r["layer"] == "spoken" else "jargon"
        out.append(dict(surface=r["slang"], reading_tai_lo="",
                        canonical_zh=r["meaning"], canonical_en=r["origin"],
                        type=typ, votes="",
                        source=r["source"], tier=3 if typ == "slang" else 2))

    # 4. abbreviations: English/Latin abbr -> Mandarin full name
    for r in rd("abbr_dict_v1.tsv"):
        out.append(dict(surface=r["abbr"], reading_tai_lo="",
                        canonical_zh=r["zh"], canonical_en="",
                        type="abbr", votes="",
                        source=f"abbr:{r['source']}", tier=int(r.get("tier", 2))))

    # dedupe on (surface, canonical_zh, type); keep the lowest tier (best src)
    best = {}
    for r in out:
        k = (r["surface"], r["canonical_zh"], r["type"])
        if k not in best or int(r["tier"]) < int(best[k]["tier"]):
            best[k] = r
    rows = sorted(best.values(), key=lambda r: (r["type"], r["canonical_zh"]))

    dest = os.path.join(D, "unified_med_lexicon.tsv")
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    by_type = Counter(r["type"] for r in rows)
    print(f"unified_med_lexicon.tsv: {len(rows)} rows (deduped from {len(out)})")
    for t, n in by_type.most_common():
        print(f"  {t:<15} {n}")


if __name__ == "__main__":
    main()
