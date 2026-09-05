#!/usr/bin/env python3
"""Mine Taiwan MOEX Nursing Licensing Exams for IlhaEmbed.

Extracts:
1. Abbreviation and jargon pairs (surface -> canonical) from question stems & options.
2. Expert-curated hard negative triplets (anchor, positive, hard_negative) for contrastive learning.

Source: Taiwan Ministry of Examination (MOEX) Public Domain Exam Questions.
Author: WeeMed AI / IlhaEmbed Data Pipeline
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

MOEX_URL_CSV = "https://raw.githubusercontent.com/pofeng/exams_tw/main/url.csv"
CLEAN_PREFIXES = re.compile(
    r"^(?:下列|依據|有關|因|以|對於|使用|接受|針對|罹患|患有|面對|指導|禁止使用|由|受檢者|病人|個案|家屬|照顧者|產婦|幼兒|成人|處於|出現|導致|小時服用|"
    r"[張李王陳林趙黃吳周徐劉何高郭梁鄭謝宋唐許韓鄧曹彭曾蕭田董袁于余](?:先生|女士|太太|小妹|伯伯|爺爺|阿姨|童|氏)?\s*)+"
)
CLEAN_SUFFIXES = re.compile(r"(?:所致|引起|為主|之|的|病人|患者|處置|護理|指導|評估|藥物|發作|反應)$")

RE_NEGATIVE_STEM = re.compile(
    r"(?:不適宜|不適當|不正確|何者錯誤|何者非|不宜|何者不屬|何者較不|何者不包括|何者最不可能|何者不需|何者不具|何者最不適當)"
)

# Apposition regexes
RE_ZH_EN = re.compile(r"([\u4e00-\u9fa5]{2,25})\s*[（\(]([A-Za-z0-9\s\-\+\/]+)[）\)]")
RE_EN_ZH = re.compile(r"([A-Za-z0-9\s\-\+\/]{2,25})\s*[（\(]([\u4e00-\u9fa5]{2,25})[）\)]")
RE_DEFINITIONAL = re.compile(r"[「『]([A-Za-z0-9\.\-\/\s]+)[」』].*?(?:係指|代表|意義為|全名為|稱為)")


def clean_term(text: str) -> str:
    """Clean clinical terms by removing noise prefixes and suffixes."""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = CLEAN_PREFIXES.sub("", t)
    t = CLEAN_SUFFIXES.sub("", t)
    return t.strip()


def parse_answers(a_text: str) -> dict[int, str]:
    """Parse answer table from MOEX answer sheet text (supports both 第N題 and tabular numbers)."""
    ans_map = {}
    lines = a_text.splitlines()
    for i, line in enumerate(lines):
        line_s = line.strip()
        if line_s.startswith("題號"):
            q_nums = [int(x) for x in re.findall(r"第\s*(\d+)\s*題", line_s)]
            if not q_nums:
                q_nums = [int(x) for x in re.findall(r"\b(\d{1,3})\b", line_s)]
            if i + 1 < len(lines):
                ans_line = lines[i + 1]
                ans_chars = re.findall(r"([A-D＃#])", ans_line)
                for q, a in zip(q_nums, ans_chars):
                    ans_map[q] = a
    return ans_map


def parse_questions(q_text: str) -> list[dict]:
    """Parse up to 100 questions and 4 choices from MOEX question text."""
    text = q_text
    text = text.replace("\ue18c", " [A] ").replace("", " [A] ")
    text = text.replace("\ue18d", " [B] ").replace("", " [B] ")
    text = text.replace("\ue18e", " [C] ").replace("", " [C] ")
    text = text.replace("\ue18f", " [D] ").replace("", " [D] ")
    text = re.sub(r"代號：\d+.*?\n", "\n", text)
    text = re.sub(r"頁次：\d+－\d+.*?\n", "\n", text)
    text = re.sub(r"\x0c", "\n", text)

    pattern = re.compile(r"(?:^|\n)\s*(\d{1,3})\s{2,}")
    parts = pattern.split(text)

    questions = []
    for i in range(1, len(parts), 2):
        q_num = int(parts[i])
        chunk = parts[i + 1]
        c_split = re.split(r"\s*\[([A-D])\]\s*", chunk)
        if len(c_split) >= 9:
            stem = re.sub(r"\s+", " ", c_split[0].strip())
            choices = {}
            for j in range(1, len(c_split), 2):
                letter = c_split[j]
                ctext = re.sub(r"\s+", " ", c_split[j + 1].strip())
                choices[letter] = ctext
            questions.append({"number": q_num, "stem": stem, "choices": choices})
    return questions


def mine_exam(q_text: str, a_text: str, meta: dict) -> tuple[set, list]:
    """Extract jargon pairs and triplets from a single exam."""
    ans_map = parse_answers(a_text)
    questions = parse_questions(q_text)

    jargon_pairs = set()
    triplets = []

    for q in questions:
        q_num = q["number"]
        stem = q["stem"]
        choices = q["choices"]

        if q_num not in ans_map:
            continue
        ans = ans_map[q_num]
        if ans not in ("A", "B", "C", "D"):
            continue

        full_text = stem + " " + " ".join(choices.values())

        # 1. Mine Jargon from Appositions
        for m in RE_ZH_EN.finditer(full_text):
            zh_raw, en_raw = m.group(1), m.group(2).strip()
            zh = clean_term(zh_raw)
            if len(en_raw) >= 2 and len(zh) >= 2 and not re.match(r"^[A-D]$", en_raw):
                jargon_pairs.add((en_raw, zh, "abbr", "MOEX-nursing", 1))

        for m in RE_EN_ZH.finditer(full_text):
            en_raw, zh_raw = m.group(1).strip(), m.group(2)
            zh = clean_term(zh_raw)
            if len(en_raw) >= 2 and len(zh) >= 2 and not re.match(r"^[A-D]$", en_raw):
                jargon_pairs.add((en_raw, zh, "abbr", "MOEX-nursing", 1))

        # 2. Mine Definitional Jargon
        m_def = RE_DEFINITIONAL.search(stem)
        if m_def:
            term = m_def.group(1).strip()
            definition = clean_term(choices[ans])
            if len(term) >= 2 and 2 <= len(definition) <= 30:
                jargon_pairs.add((term, definition, "jargon", "MOEX-nursing", 1))

        is_neg_query = bool(RE_NEGATIVE_STEM.search(stem))
        if re.search(r"[，,、]\s*下列", stem):
            anchor = re.sub(r"[，,、]\s*下列.*", "", stem).strip()
        else:
            anchor = re.sub(r"^下列\s*(?:敘述|何者|選項)?\s*(?:為|是|屬於|構成|最|顯示)?\s*", "", stem).strip()
        anchor = re.sub(r"^(?:有關|針對|對於|為|若|此時)\s*", "", anchor).strip()
        anchor = re.sub(r"[？\?。！!]$", "", anchor).strip()
        if len(anchor) > 45:
            anchor = anchor[-45:]

        if not is_neg_query:
            # Positive = correct choice, Negatives = distractors
            pos = choices[ans]
            for l, c in choices.items():
                if l != ans and 2 <= len(c) <= 60:
                    triplets.append({
                        "anchor": anchor,
                        "positive": pos,
                        "hard_negative": c,
                        "polarity": "standard",
                        "subject": meta.get("subject", ""),
                        "year": meta.get("year", ""),
                        "q_num": q_num,
                    })
        else:
            # Inverted question: answer is the contraindicated/wrong action (Negative)
            # Other 3 choices are valid clinical guidelines (Positives)
            neg = choices[ans]
            for l, c in choices.items():
                if l != ans and 2 <= len(c) <= 60:
                    triplets.append({
                        "anchor": f"{anchor} 護理常規",
                        "positive": c,
                        "hard_negative": neg,
                        "polarity": "contraindicated",
                        "subject": meta.get("subject", ""),
                        "year": meta.get("year", ""),
                        "q_num": q_num,
                    })

    return jargon_pairs, triplets


def download_and_extract(url: str, cache_dir: Path, prefix: str) -> str:
    """Download PDF and convert to layout text using pdftotext with retry."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / f"{prefix}.pdf"
    txt_path = cache_dir / f"{prefix}.txt"

    if not txt_path.exists():
        if not pdf_path.exists():
            last_err = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        pdf_path.write_bytes(resp.read())
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(1.0)
            else:
                raise RuntimeError(f"Failed to download {url} after 3 attempts: {last_err}")
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(txt_path)], check=True)

    return txt_path.read_text(encoding="utf-8", errors="ignore")


ALL_YEARS = [str(y) for y in range(101, 115)]


def main():
    parser = argparse.ArgumentParser(description="Mine MOEX Nursing & Physician Stage 1 Exams for IlhaEmbed")
    parser.add_argument("--years", nargs="+", default=ALL_YEARS, help="Exam years (e.g. 101 .. 114)")
    parser.add_argument("--depts", nargs="+", default=["護理師", "醫師(一)", "醫師（一）"], help="Target exam departments")
    parser.add_argument("--limit", type=int, default=500, help="Max exam papers to process")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/moex_exams"), help="Cache directory")
    parser.add_argument("--out-dir", type=Path, default=Path("training/mining"), help="Output directory")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching MOEX catalog: {MOEX_URL_CSV}")
    req = urllib.request.Request(MOEX_URL_CSV, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8-sig", errors="ignore")

    reader = csv.reader(io.StringIO(content))
    header = next(reader)

    candidate_rows = []
    for r in reader:
        if r[0] in args.years and r[10] == "測驗題":
            dept = r[7]
            subj = r[9]
            if any(d in dept for d in args.depts):
                if "醫師" in dept:
                    # Only keep Physician Stage 1 Basic Medical Sciences (醫學一 & 醫學二)
                    if not (subj.startswith("醫學(一)") or subj.startswith("醫學（一）") or subj.startswith("醫學(二)") or subj.startswith("醫學（二）")):
                        continue
                candidate_rows.append(r)

    print(f"Found {len(candidate_rows)} candidate nursing & physician stage 1 exams for years {args.years}.")
    target_rows = candidate_rows[: args.limit]
    print(f"Processing top {len(target_rows)} exam papers...")

    all_jargon = set()
    all_triplets = []

    for idx, r in enumerate(target_rows, 1):
        year, code, exam_name, _, _, _, _, dept, session, subj, _, q_url, a_url, _ = r
        meta = {"year": year, "code": code, "subject": subj}
        prefix = f"{year}_{code}_{session}_{dept}_{subj}".replace("/", "_").replace(" ", "")

        print(f"[{idx}/{len(target_rows)}] {year}年 {subj} ({code})...")
        try:
            q_text = download_and_extract(q_url, args.cache_dir, f"{prefix}_Q")
            a_text = download_and_extract(a_url, args.cache_dir, f"{prefix}_A")
            jargon, triplets = mine_exam(q_text, a_text, meta)
            all_jargon.update(jargon)
            all_triplets.extend(triplets)
            print(f"   -> Mined {len(jargon)} jargon pairs, {len(triplets)} triplets")
        except Exception as e:
            print(f"   ! Error processing {prefix}: {e}")

    # Write Jargon TSV
    jargon_file = args.out_dir / "nursing_jargon_pairs.tsv"
    with jargon_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["surface", "canonical", "type", "source", "tier"])
        for row in sorted(list(all_jargon)):
            w.writerow(row)

    # Write Triplets JSONL
    triplet_file = args.out_dir / "nursing_hard_negatives.jsonl"
    with triplet_file.open("w", encoding="utf-8") as f:
        for t in all_triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("Mining Complete!")
    print(f"Total Unique Jargon Pairs: {len(all_jargon)} -> {jargon_file}")
    print(f"Total Hard Negative Triplets: {len(all_triplets)} -> {triplet_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
