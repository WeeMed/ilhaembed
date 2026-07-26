#!/usr/bin/env python3
"""Apposition / gloss miner -- the reason a search engine "knows" 乳超.

Aliases aren't in dictionaries; they're glossed in running text:
  "心肌梗死（英語：myocardial infarction，MI）又稱心肌梗塞、心梗，俗稱心臟病發作"
One article yields many surface->canonical pairs INCLUDING abbreviations
(心梗, COPD, DM, MI) that redirects/dictionaries miss. This is Hearst-pattern
extraction over zh.wikipedia intro extracts (CC BY-SA, batched API).

Output: appos_pairs.tsv (surface<TAB>canonical<TAB>source)
"""
import csv
import gzip
import re
import time
import urllib.parse
import urllib.request
import argparse
from pathlib import Path

API = "https://zh.wikipedia.org/w/api.php"
CJK = re.compile(r"[一-鿿]")

# alias-introducing markers; capture the run of terms after them
MARK = r"(?:又稱|又名|亦稱|亦称|俗稱|俗称|簡稱|简称|缩写为|縮寫為|缩寫為|常稱為|常称为|全稱|全称|別名|别名)"
SEG = re.compile(MARK + r"[為为:：]?\s*([^，。；\s（）()]+(?:、[^，。；、\s（）()]+)*)")
# latin abbrev inside the leading paren:  （…，MI） / （…，缩写为COPD）
LATIN = re.compile(r"（[^）]*?([A-Z][A-Za-z]{1,6})[，、）]")


def api(params):
    params.update(format="json")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "med-lexicon-research/1"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return __import__("json").load(r)
        except Exception:
            time.sleep(1.0)
    return {}


def seed_titles(lexicon: Path | None, terminology_dir: Path | None):
    titles = set()
    if lexicon and lexicon.exists():
        with lexicon.open(encoding="utf-8") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            for r in rows:
                c = r["canonical_zh"].strip()
                if CJK.search(c) and 2 <= len(c) <= 12 and not re.search(r"[A-Za-z0-9]", c):
                    titles.add(c)
    if terminology_dir:
        path = terminology_dir / "icd10_cm_2023.csv.gz"
        if path.exists():
            with gzip.open(path, "rt") as handle:
                for r in csv.DictReader(handle):
                    z = (r.get("display_zh") or "").strip()
                    if z and CJK.search(z) and 2 <= len(z) <= 6 and not re.search(r"[A-Za-z0-9]", z):
                        titles.add(z)
    return sorted(titles)


def extract_aliases(title, text):
    if not text:
        return []
    lead = text[:400]                       # aliases live in the first sentence(s)
    out = []
    for seg in SEG.findall(lead):
        for a in seg.split("、"):
            a = a.strip("「」\"'　 ")
            if 2 <= len(a) <= 12 and a != title:
                out.append(a)
    for m in LATIN.findall(lead):
        if m.lower() not in ("english", "latin") and m != title:
            out.append(m)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", type=Path, help="TSV with a canonical_zh column")
    parser.add_argument("--terminology-dir", type=Path, help="directory containing icd10_cm_2023.csv.gz")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("appos_pairs.tsv"))
    args = parser.parse_args()

    seeds = seed_titles(args.lexicon, args.terminology_dir)
    if not seeds:
        parser.error("provide --lexicon and/or --terminology-dir with readable source data")
    print(f"seed medical titles: {len(seeds)}", flush=True)
    pairs, seen = [], set()
    B = 20
    for i in range(0, len(seeds), B):
        batch = seeds[i:i+B]
        d = api({"action": "query", "prop": "extracts", "exintro": 1,
                 "explaintext": 1, "redirects": 1, "exlimit": "max",
                 "titles": "|".join(batch)})
        # map possibly-redirected titles back is unnecessary; use returned title
        for page in d.get("query", {}).get("pages", {}).values():
            t = page.get("title", "")
            for a in extract_aliases(t, page.get("extract", "")):
                if (a, t) not in seen:
                    seen.add((a, t)); pairs.append((a, t))
        if i % 400 == 0:
            print(f"  {i}/{len(seeds)} -> {len(pairs)} pairs", flush=True)
        time.sleep(0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["surface", "canonical", "source"])
        for s, c in pairs:
            w.writerow([s, c, "zhwiki-appos"])
    abbr = sum(1 for s, _ in pairs if re.search(r"[A-Za-z]", s))
    print(f"\nappos_pairs.tsv: {len(pairs)} pairs ({abbr} latin-abbrev)", flush=True)
    for s, c in pairs[:16]:
        print(f"  {s} -> {c}")


if __name__ == "__main__":
    main()
