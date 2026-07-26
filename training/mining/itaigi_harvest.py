#!/usr/bin/env python3
"""iTaigi harvester -- pull crowd-sourced Taigi readings for medical terms.

API (reverse-engineered from i3thuan5/itaigi frontend 後端.jsx):
  GET https://itaigi.tw/平臺項目列表/揣列表?關鍵字=<華語詞>
  -> {列表:[{外語資料, 新詞文本:[{文本資料(台語漢字), 音標資料(台羅),
             貢獻者, 按呢講好, 按呢無好}]}], 其他建議:[...]}

Crowd votes (按呢講好/無好) give a quality signal MOE dict lacks. Output ranks
variants by net votes -> the top row is the community-preferred pronunciation,
exactly what an ASR biasing lexicon wants.
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request

_PATH = urllib.parse.quote("平臺項目列表/揣列表", safe="/")
_KEY = urllib.parse.quote("關鍵字")
BASE = f"https://itaigi.tw/{_PATH}?{_KEY}="

# seed: common spoken clinical Mandarin terms (what gets said in a ward meeting)
SEED = [
    "醫院", "醫生", "護士", "病人", "住院", "出院", "開刀", "手術", "打針",
    "點滴", "吊點滴", "血壓", "發燒", "食藥", "藥仔", "感冒", "咳嗽", "疼",
    "檢查", "超音波", "電腦斷層", "X光", "抽血", "驗血", "驗尿", "傷口",
    "換藥", "包紮", "麻醉", "化療", "癌症", "腫瘤", "中風", "糖尿病",
    "高血壓", "心臟病", "洗腎", "懷孕", "生產", "月內", "過敏", "住加護病房",
]


def fetch(term):
    url = BASE + urllib.parse.quote(term)
    req = urllib.request.Request(url, headers={"User-Agent": "med-lexicon-harvest/1"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def rows_for(term, data):
    out = []
    for item in data.get("列表", []):
        hua = item.get("外語資料", term)
        for t in item.get("新詞文本", []):
            good = int(t.get("按呢講好", 0))
            bad = int(t.get("按呢無好", 0))
            out.append({
                "華語": hua,
                "台語漢字": t.get("文本資料", ""),
                "台羅": t.get("音標資料", ""),
                "好": good, "無好": bad, "淨票": good - bad,
                "貢獻者": t.get("貢獻者", ""),
            })
    out.sort(key=lambda r: -r["淨票"])
    return out


def load_seed():
    p = os.path.join(os.path.dirname(__file__), "union_seed.txt")
    if os.path.exists(p):
        with open(p) as f:
            terms = [l.strip() for l in f if l.strip()]
        print(f"seed: union_seed.txt ({len(terms)} terms)")
        return terms
    print(f"seed: built-in SEED ({len(SEED)} terms)")
    return SEED


def main():
    seed = load_seed()
    all_rows = []
    for i, term in enumerate(seed):
        try:
            data = fetch(term)
            rs = rows_for(term, data)
            all_rows.extend(rs)
            print(f"  [{i+1}/{len(SEED)}] {term}: {len(rs)} variants"
                  + (f"  top={rs[0]['台語漢字']}/{rs[0]['台羅']} (+{rs[0]['淨票']})"
                     if rs else ""))
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(SEED)}] {term}: ERROR {e}")
        time.sleep(0.5)  # be polite to a community server

    out = os.path.join(os.path.dirname(__file__), "itaigi_med_lexicon_full.tsv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["華語", "台語漢字", "台羅", "好", "無好", "淨票", "貢獻者"])
        for r in all_rows:
            w.writerow([r["華語"], r["台語漢字"], r["台羅"], r["好"],
                        r["無好"], r["淨票"], r["貢獻者"]])
    print(f"\nitaigi_med_lexicon_full.tsv: {len(all_rows)} rows "
          f"from {len(seed)} seed terms")


if __name__ == "__main__":
    main()
