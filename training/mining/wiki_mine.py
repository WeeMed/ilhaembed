#!/usr/bin/env python3
"""Mine zh.wikipedia for medical surface->canonical pairs.

Redirects ARE synonym pairs: every redirect title -> its target article is an
alternate name for the same concept (中風 <- 腦中風/脑卒中/脑血管意外/Brain attack).
This is the classic synonym-mining technique -- clean, CC BY-SA, API-crawlable,
and exactly the Chinese surface variation the specialized track is starved of.

Seed = (a) canonical_zh from our unified lexicon + (b) members of key medical
categories (discovers concepts we don't have). Then batch prop=redirects.

Output: wiki_pairs.tsv  (surface<TAB>canonical<TAB>source)
"""
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import argparse
from pathlib import Path

API = "https://zh.wikipedia.org/w/api.php"
CJK = re.compile(r"[一-鿿]")
CATS = [
    "疾病", "症状", "医学术语", "外科手术", "医学检查", "醫學診斷",
    "精神疾病", "癌症", "感染病", "神经系统疾病", "心血管疾病",
]


def api(params):
    params.update(format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "med-lexicon-research/1 (contact: research)"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception:
            time.sleep(1.0)
    return {}


def category_members(cat, limit=500):
    out, cont = [], None
    while True:
        p = {"action": "query", "list": "categorymembers",
             "cmtitle": f"Category:{cat}", "cmlimit": "500", "cmtype": "page"}
        if cont:
            p["cmcontinue"] = cont
        d = api(p)
        out += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont or len(out) >= limit:
            break
        time.sleep(0.3)
    return out


def redirects_batch(titles):
    """titles<=50 -> list of (redirect, canonical)."""
    pairs = []
    d = api({"action": "query", "prop": "redirects", "rdlimit": "max",
             "titles": "|".join(titles)})
    for page in d.get("query", {}).get("pages", {}).values():
        canon = page.get("title", "")
        for rd in page.get("redirects", []):
            pairs.append((rd["title"], canon))
    return pairs


def seed_titles(lexicon: Path | None, terminology_dir: Path | None):
    """Seed purely from guaranteed-medical Chinese terms: lexicon canonicals +
    short ICD-10-CM/PCS Chinese displays (common disease/procedure names likely
    to have wiki articles). No category crawl -- it returns list/subcat pages and
    injects non-medical noise."""
    import gzip
    titles = set()
    if lexicon and lexicon.exists():
        with lexicon.open(encoding="utf-8") as handle:
            for r in csv.DictReader(handle, delimiter="\t"):
                c = r["canonical_zh"].strip()
                if CJK.search(c) and 2 <= len(c) <= 12:
                    titles.add(c)
    if terminology_dir:
        for name in ("icd10_cm_2023.csv.gz", "icd10_pcs_2023.csv.gz"):
            path = terminology_dir / name
            if not path.exists():
                continue
            with gzip.open(path, "rt") as handle:
                for r in csv.DictReader(handle):
                    z = (r.get("display_zh") or "").strip()
                    if z and CJK.search(z) and 2 <= len(z) <= 8 and not re.search(r"[A-Za-z0-9]", z):
                        titles.add(z)
    return sorted(titles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", type=Path, help="TSV with a canonical_zh column")
    parser.add_argument("--terminology-dir", type=Path, help="directory containing licensed ICD files")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("wiki_pairs.tsv"))
    args = parser.parse_args()

    seeds = seed_titles(args.lexicon, args.terminology_dir)
    if not seeds:
        parser.error("provide --lexicon and/or --terminology-dir with readable source data")
    print(f"seed titles: {len(seeds)}", flush=True)
    pairs, seen = [], set()
    for i in range(0, len(seeds), 50):
        for surf, canon in redirects_batch(seeds[i:i+50]):
            if surf != canon and (surf, canon) not in seen:
                seen.add((surf, canon)); pairs.append((surf, canon))
        if i % 500 == 0:
            print(f"  {i}/{len(seeds)} seeds -> {len(pairs)} pairs", flush=True)
        time.sleep(0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["surface", "canonical", "source"])
        for s, c in pairs:
            w.writerow([s, c, "zhwiki-redirect"])
    zh = sum(1 for s, _ in pairs if CJK.search(s))
    print(f"\nwiki_pairs.tsv: {len(pairs)} pairs ({zh} with CJK surface)", flush=True)
    for s, c in pairs[:12]:
        print(f"  {s} -> {c}")


if __name__ == "__main__":
    main()
