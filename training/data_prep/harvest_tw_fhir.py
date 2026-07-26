#!/usr/bin/env python3
"""Harvest bilingual concept pairs from Taiwan's official FHIR implementation guides.

Why this exists. The SNOMED CT release shipped in the product has 523,502 rows and ZERO Chinese
displays, and the international terminologies carry no Taiwanese clinical wording at all. But
Taiwan's own FHIR implementation guides publish exactly what is missing: a code with BOTH its
English display and a zh-TW designation. One code carrying two surface forms IS a concept grouping,
which is the structure self-alignment training needs and the structure a flat code list cannot give.

These guides are also the authoritative source for the vocabulary that fails hardest in the field.
The department value sets (就醫科別 / 診療科別) map the national insurance department codes onto
SNOMED CT with Chinese names -- the same `care_source` category that carried eight hand-written seed
exemplars and read a department at 0.64, below the gate.

Sources (government-published, publicly downloadable, recorded with retrieval date in the output):
    TW Core IG   -- 臺灣核心實作指引, 衛福部 (twcore.mohw.gov.tw)
    TWIDIR       -- 臺灣傳染病檢驗報告實作指引, 疾管署 (twidir.cdc.gov.tw)

Output is a TSV of (system, code, display_en, display_zh, source_ig, artifact) so every pair keeps
its provenance -- a training pair whose origin cannot be named is not auditable later.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "tw_fhir_bilingual.tsv"

# A designation is Chinese if FHIR says so. Guides differ in how precisely they tag the locale, so
# all the forms Taiwan's guides actually use are accepted rather than one canonical spelling.
ZH_TAGS = {"zh-tw", "zh-hant", "zh", "zh-hant-tw", "zh_tw"}
CJK = __import__("re").compile(r"[\u4e00-\u9fff]")


def has_han(value: str | None) -> bool:
    """Whether a display is actually Chinese.

    Taiwan's guides put the Chinese term straight into `display` rather than into a designation, so
    a harvester that assumes display==English silently files every Chinese name in the English
    column. The language of a string is decided by looking at it, not by which field it sits in."""
    return bool(value) and bool(CJK.search(str(value)))


def split_by_language(display: str | None, chinese: str | None) -> tuple[str, str]:
    """Return (english, chinese) however the guide happened to arrange them."""
    display = (display or "").strip()
    chinese = (chinese or "").strip()
    if chinese:
        return (display if not has_han(display) else "", chinese)
    if has_han(display):
        return ("", display)
    return (display, "")


def is_zh(value: str | None) -> bool:
    return bool(value) and str(value).lower() in ZH_TAGS


def zh_from_designations(node: dict) -> str | None:
    """The Chinese surface form attached to a concept, if the guide provides one."""
    for designation in node.get("designation") or []:
        if is_zh(designation.get("language")) and designation.get("value"):
            return str(designation["value"]).strip()
    # Some guides put the Chinese in an extension rather than a designation.
    for extension in node.get("extension") or []:
        if "translation" in str(extension.get("url", "")).lower():
            for sub in extension.get("extension") or []:
                if sub.get("url") == "content" and sub.get("valueString"):
                    return str(sub["valueString"]).strip()
    return None


def walk_concepts(concepts: list, system: str, rows: list, ig: str, artifact: str) -> None:
    """CodeSystem concepts nest arbitrarily deep; a child concept is as real as a root one."""
    for concept in concepts or []:
        code = concept.get("code")
        display = (concept.get("display") or "").strip()
        chinese = zh_from_designations(concept)
        english, zh = split_by_language(display, chinese)
        if code and (english or zh):
            rows.append((system, code, english, zh, ig, artifact))
        walk_concepts(concept.get("concept") or [], system, rows, ig, artifact)


def harvest(package_dir: Path, ig: str) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for path in sorted(glob.glob(str(package_dir / "package" / "*.json"))):
        try:
            resource = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a malformed artifact is skipped, not fatal
            continue
        kind = resource.get("resourceType")
        artifact = resource.get("id") or Path(path).stem

        if kind == "CodeSystem":
            walk_concepts(resource.get("concept") or [], resource.get("url", ""), rows, ig, artifact)

        elif kind == "ValueSet":
            for include in (resource.get("compose") or {}).get("include") or []:
                system = include.get("system", "")
                for concept in include.get("concept") or []:
                    code = concept.get("code")
                    display = (concept.get("display") or "").strip()
                    english, zh = split_by_language(display, zh_from_designations(concept))
                    if code and (english or zh):
                        rows.append((system, code, english, zh, ig, artifact))
            for contains in (resource.get("expansion") or {}).get("contains") or []:
                code = contains.get("code")
                display = (contains.get("display") or "").strip()
                english, zh = split_by_language(display, zh_from_designations(contains))
                if code and (english or zh):
                    rows.append((contains.get("system", ""), code, english, zh, ig, artifact))

        elif kind == "ConceptMap":
            # A mapping is itself a synonym statement: a national code and an international one
            # denoting the same thing, each with its own display.
            for group in resource.get("group") or []:
                target_system = group.get("target", "")
                for element in group.get("element") or []:
                    left = (element.get("display") or "").strip()
                    for mapped in element.get("target") or []:
                        right = (mapped.get("display") or "").strip()
                        if not (left or right):
                            continue
                        english = right if not has_han(right) else (left if not has_han(left) else "")
                        zh = left if has_han(left) else (right if has_han(right) else "")
                        code = mapped.get("code") or element.get("code") or ""
                        if code and (english or zh):
                            rows.append((target_system, code, english, zh, ig, f"{artifact}:map"))
    return rows


def main() -> int:
    packages = [(Path(sys.argv[1]), sys.argv[2])] if len(sys.argv) > 2 else [
        (HERE / "igs" / "x_twcore", "tw-core-ig"),
        (HERE / "igs" / "x_twidir", "twidir"),
    ]
    rows: list[tuple[str, ...]] = []
    for directory, ig in packages:
        found = harvest(directory, ig)
        print(f"{ig}: {len(found)} concept rows")
        rows.extend(found)

    seen: set[tuple[str, ...]] = set()
    unique = []
    for row in rows:
        key = (row[0], row[1], row[2], row[3])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    bilingual = [r for r in unique if r[2] and r[3]]
    zh_only = [r for r in unique if r[3] and not r[2]]
    snomed = [r for r in unique if "snomed" in r[0].lower()]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("system", "code", "display_en", "display_zh", "source_ig", "artifact"))
        writer.writerows(unique)

    print(f"\nunique concepts : {len(unique)}")
    print(f"  bilingual (EN+ZH, usable as a synonym pair): {len(bilingual)}")
    print(f"  Chinese only                              : {len(zh_only)}")
    print(f"  SNOMED CT coded                           : {len(snomed)}")
    print(f"written -> {OUT}")

    print("\nsample bilingual pairs:")
    for row in bilingual[:12]:
        print(f"  {row[3]}  ==  {row[2]}   [{row[1]}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
