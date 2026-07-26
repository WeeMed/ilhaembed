#!/usr/bin/env python3
"""abbr-collector v1 -- turn heterogeneous published Taiwanese medical
abbreviation PDFs into one normalized alias table.

Pipeline per source: download -> pdftotext (layout) -> flag needs-OCR if empty
-> regex-extract (abbr, chinese) candidate rows -> tag with source+provenance.

This is a FIRST PASS. Layout-aware per-source parsing and code-joining are the
remaining work; the point here is to prove the collection is scriptable and see
real coverage. Output: abbr_dict_v1.tsv  (abbr | zh | source | tier)
"""
import re
import subprocess
import sys
from pathlib import Path

D = Path(__file__).parent

# provenance tiers: 1=gov/committee-approved (authoritative), 2=hospital nursing
# teaching material, 3=community. Collision resolution prefers the lowest tier.
SOURCES = [
    # (local_name, url, tier)
    ("nutc",    "https://nursing.nutc.edu.tw/var/file/70/1070/img/2320/481820670.pdf", 2),
    ("nutc2",   "https://nursing.nutc.edu.tw/var/file/70/1070/img/2797/995061834.pdf", 2),
    ("mhchcm",  "https://c017.mhchcm.edu.tw/var/file/17/1017/img/338/51071835.pdf", 2),
    ("sijhih",  "https://sijhih.cgh.org.tw/rwd102/store/f4/177-19.pdf", 2),
    ("wagners", "https://wagners.com.tw/wp-content/uploads/2022/04/YJ10-6.pdf", 3),
    ("kmu",     "https://respcare.kmu.edu.tw/attachments/article/187/"
                "%E9%AB%98%E9%86%AB%E9%99%84%E9%99%A2%E7%97%85%EF%A6%8C%E9%86%AB"
                "%E5%9B%91%E9%80%9A%E7%94%A8%E5%8F%8A%EF%A5%A7%E5%8F%AF%E4%BD%BF"
                "%E7%94%A8%E7%B8%AE%E5%AF%AB%EF%A6%9C%E8%A1%A8.pdf", 1),
    ("hpa_breast", "https://pportal.hpa.gov.tw/ub_file/upload_file/DOWNLOAD/"
                "115%E4%B9%B3%E7%99%8C%E7%AF%A9%E6%AA%A2%E7%B3%BB%E7%B5%B1%E6%95%99"
                "%E8%82%B2%E8%A8%93%E7%B7%B4%E6%95%99%E6%9D%90(%E9%86%AB%E7%99%82"
                "%E9%99%A2%E6%89%80%E7%AB%AF).pdf", 1),
]

CJK = r"一-鿿"
# a Latin/English abbreviation token: letters, digits, . - / and spaces, 1-6 words
ABBR = r"[A-Za-z][A-Za-z0-9./\-]{0,24}(?:\s[A-Za-z0-9./\-]{1,15}){0,3}"
# numbered row:  "13. CT 電腦斷層檢查"  or  "CT   電腦斷層"
ROW_PATTERNS = [
    re.compile(rf"^\s*\d+[.)]\s*({ABBR})\s+([{CJK}][{CJK}A-Za-z0-9()（） /]{{1,30}})"),
    re.compile(rf"^\s*({ABBR})\s{{1,}}([{CJK}][{CJK}A-Za-z0-9()（） /]{{1,30}})"),
]


def fetch(name, url):
    pdf = D / f"{name}.pdf"
    if not pdf.exists() or pdf.stat().st_size < 1000:
        subprocess.run(["curl", "-sL", "-o", str(pdf), url], check=False)
    txt = D / f"{name}.txt"
    subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), str(txt)],
                   check=False)
    if not txt.exists():
        return None, True
    lines = [l for l in txt.read_text(errors="replace").splitlines() if l.strip()]
    needs_ocr = len(lines) < 5
    return lines, needs_ocr


def extract(lines):
    rows = []
    for line in lines or []:
        for pat in ROW_PATTERNS:
            m = pat.match(line)
            if m:
                abbr = m.group(1).strip()
                zh = m.group(2).strip()
                # abbr must contain a latin letter and not be a pure chinese line
                if re.search(r"[A-Za-z]", abbr) and re.search(rf"[{CJK}]", zh):
                    rows.append((abbr, zh))
                break
    return rows


def main():
    all_rows = {}  # (abbr_lower) -> (abbr, zh, source, tier)
    report = []
    for name, url, tier in SOURCES:
        lines, needs_ocr = fetch(name, url)
        rows = extract(lines)
        report.append((name, tier, len(lines or []), needs_ocr, len(rows)))
        for abbr, zh in rows:
            key = abbr.lower()
            # keep the lowest-tier (most authoritative) source per abbr
            if key not in all_rows or tier < all_rows[key][3]:
                all_rows[key] = (abbr, zh, name, tier)

    out = D / "abbr_dict_v1.tsv"
    with out.open("w") as f:
        f.write("abbr\tzh\tsource\ttier\n")
        for abbr, zh, name, tier in sorted(all_rows.values()):
            f.write(f"{abbr}\t{zh}\t{name}\t{tier}\n")

    print("=== per-source ===")
    print(f"{'source':<12}{'tier':>5}{'lines':>7}{'ocr?':>6}{'rows':>6}")
    for name, tier, nl, ocr, nr in report:
        print(f"{name:<12}{tier:>5}{nl:>7}{'YES' if ocr else '-':>6}{nr:>6}")
    print(f"\n=== merged unique abbreviations: {len(all_rows)} -> {out.name} ===")
    # show imaging/exam-relevant subset
    exam = [v for v in all_rows.values() if re.search(
        r"超音波|斷層|磁振|核磁|攝影|內視鏡|正子|骨密|骨質|心電|掃描|篩檢", v[1])]
    print(f"\n=== exam/imaging rows ({len(exam)}) ===")
    for abbr, zh, name, tier in sorted(exam):
        print(f"  {abbr:<18} {zh:<20} [{name} t{tier}]")


if __name__ == "__main__":
    sys.exit(main())
