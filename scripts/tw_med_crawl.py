#!/usr/bin/env python3
"""Targeted Taiwan-medical web crawl + apposition mining (server-rendered pages).

The mechanism a search engine uses to "know 乳超": real clinical text glosses
abbreviations. We fetch server-rendered 衛教/專科 pages and apply HIGH-PRECISION
apposition patterns (avoid the noise of loose paren matching):

  1. FULLNAME（ABBR）  where ABBR is uppercase-Latin 2-6  -> 電腦斷層掃描(CT)
  2. FULLNAME（簡稱/俗稱/又稱 X）                          -> explicit marker
  3. FULLNAME，(簡稱|俗稱|又稱)X

FULLNAME = the maximal CJK run (<=10) immediately before the marker. Output pairs
are (surface=alias, canonical=fullname). Low volume but real clinical Taiwan text.

Output: tw_med_pairs.tsv
"""
import csv
import html
import os
import re
import time
import urllib.request

D = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (research; medical-terminology)"}

# rsroc = 中華民國放射線醫學會 knowledge base (imaging/exam-heavy, content.asp?ID=N)
SEED_URLS = [f"https://www.rsroc.org.tw/knowledge/news/content.asp?ID={i}" for i in range(1, 120)]

FULL = r"([一-鿿]{2,10})"
P_LATIN = re.compile(FULL + r"\s*[（(]\s*([A-Z][A-Za-z]{1,6})\s*[，,]?[^）)]{0,20}[）)]")
P_MARK_PAREN = re.compile(FULL + r"\s*[（(]\s*(?:又稱|俗稱|簡稱|简称|缩写[為为]?|亦稱)\s*[：:]?\s*([一-鿿A-Za-z]{2,10})")
P_MARK = re.compile(FULL + r"\s*[，,]?\s*(?:又稱|俗稱|簡稱|简称)\s*[「]?([一-鿿A-Za-z]{2,10})[」]?")
NOISE = re.compile(r"醫師|主任|中心|專區|頻道|上集|下集|右|左|圖|表|先生|小姐|教授|部|科$")


def fetch(url):
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8", "replace")
    except Exception:
        return ""
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


def mine(text):
    out = []
    for pat in (P_LATIN, P_MARK_PAREN, P_MARK):
        for full, alias in pat.findall(text):
            full = full.strip(); alias = alias.strip()
            if len(full) < 2 or NOISE.search(full) or alias == full:
                continue
            out.append((alias, full))
    return out


def main():
    pairs, seen = [], set()
    hits = 0
    for i, url in enumerate(SEED_URLS):
        t = fetch(url)
        if len(re.findall(r"[一-鿿]", t)) < 200:
            continue
        hits += 1
        for a, c in mine(t):
            if (a, c) not in seen:
                seen.add((a, c)); pairs.append((a, c))
        if i % 20 == 0:
            print(f"  {i}/{len(SEED_URLS)} pages ({hits} with content) -> {len(pairs)} pairs", flush=True)
        time.sleep(0.3)

    out = os.path.join(D, "tw_med_pairs.tsv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["surface", "canonical", "source"])
        for a, c in pairs:
            w.writerow([a, c, "rsroc-weiei"])
    print(f"\ntw_med_pairs.tsv: {len(pairs)} pairs from {hits} content pages", flush=True)
    for a, c in pairs[:20]:
        print(f"  {a} -> {c}")


if __name__ == "__main__":
    main()
