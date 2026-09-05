# Training-Data Provenance — IlhaEmbed

This document records the source families mined during the development of **IlhaEmbed**. It exists for provenance, reproducibility, evaluation hygiene, and licence review.

The row counts below describe the output of each mining step. They are not a promise that every mined row appears with the same weight in every released checkpoint. A source used for training must not also be treated as a held-out evaluation source.

Private generated pair files are intentionally not published. Public readers should not need access to another private repository to understand this ledger: public source URLs and the relevant scripts in this repository are used instead of internal filesystem paths.

## Source Families

### Specialized Clinical Language (Surface → Canonical)

| ID | Source | Public Source / Acquisition | Mined Rows | Rights Status | Use and Notes |
|----|--------|-----------------------------|-----------:|---------------|---------------|
| `moe-twblg` | 教育部臺灣台語常用詞辭典 | [g0v/moedict-data-twblg](https://github.com/g0v/moedict-data-twblg), open-data dump | 778 | MOE open-data terms apply | Taigi Han text, Tâi-lô, and Mandarin definitions; 778 medical candidates filtered from 14,489 entries |
| `itaigi` | iTaigi 愛台語 | [itaigi.tw](https://itaigi.tw/), public platform data queried by keyword | 1,288 | Source-specific CC/publication terms require verification before redistribution | Crowd-contributed Taigi readings and votes |
| `slang-blog` | Taiwanese clinical slang articles | [陳志金「巷子內醫療用語」](https://snore123.blogspot.com/2019/05/medword.html), [udn article](https://blog.udn.com/ptsafetyrm/3771916), and a Vocus article | 62 | Third-party copyright; no raw redistribution permission recorded | Short surface/canonical facts such as colloquialisms and abbreviations were extracted, not the articles |
| `abbr-pdf` | Hospital abbreviation lists and nursing teaching material | Publicly accessible hospital/school PDF documents; extracted with `pdftotext` or OCR | 398 | Hospital/author copyright; no raw redistribution permission recorded | Abbreviation pairs only; source PDFs contained layout noise. Exact source manifests should be retained with the private research data |
| `wiki-redirect` | Chinese Wikipedia redirects | [MediaWiki API](https://www.mediawiki.org/wiki/API:Redirects) | 284 | CC BY-SA; attribution/share-alike obligations apply | Medical aliases mapped to article titles |
| `wiki-appos` | Chinese Wikipedia introductory text | [MediaWiki API](https://www.mediawiki.org/wiki/Extension:TextExtracts) plus apposition patterns | 371 | CC BY-SA; attribution/share-alike obligations apply | Phrases such as 又稱／俗稱／簡稱／縮寫為 yielded clinical abbreviation pairs |
| `rsroc-weiei` | 中華民國放射線醫學會衛教文章 | [rsroc.org.tw knowledge pages](https://rsroc.org.tw/knowledge/) | 34 | Society copyright; no raw redistribution permission recorded | Short factual appositions for imaging abbreviations such as LDCT, CTA, RFA, and TACE; articles are not redistributed |
| `moex-nursing` | 考選部護理師專技高考歷屆試題 | [MOEX open data](https://wwwc.moex.gov.tw/main/Exam/wHandExamQandA_CSV.ashx) via `training/mining/nursing_exam_mine.py` | 387 pairs, 2,205 triplets | 著作權法第9條第1項第5款（公眾領域 Public Domain） | 台灣護理臨床縮寫、醫囑常規、護理量表與專家級考題誘答困難負例 |

### Formal Medical Terminology & Standards

| ID | Source | Public Source / Acquisition | Mined Rows | Rights Status | Use and Notes |
|----|--------|-----------------------------|-----------:|---------------|---------------|
| `moda-naer-13` | 數位發展部 / 國教院 13 套學術醫療名詞 | [taic.moda.gov.tw](https://taic.moda.gov.tw/), open-data dump | 111,386 | MODA Open Data Terms §3.1 apply | Traditional Chinese academic and clinical terminology pairs distilled for concept alignment |
| `icd-loinc` | MOHW ICD-10-CM/PCS Chinese releases and LOINC/NHI terminology | Obtain the current releases from MOHW/NHI publication channels and [LOINC](https://loinc.org/downloads/) | 63,529 | Government-data terms and the [LOINC licence](https://loinc.org/license/) apply independently | Chinese/English cross-lingual terminology pairs |
| `snomed-syn` | SNOMED CT description synonyms | Obtain a licensed release through [SNOMED International](https://www.snomed.org/get-snomed) | 44,973 | SNOMED CT Affiliate Licence; not an unrestricted open-data corpus | English synonym → Fully Specified Name (FSN); generated pairs are not distributed here |

## What Was Not Used

- The iTaigi 2,500-seed expansion was abandoned after repeated incomplete runs; the 1,288-row base extraction is the recorded source.
- `icd_term_bridge` (approximately 204k rows) was dropped because token-level alignment produced invalid semantic pairs such as `abandonment → 照顧或`.
- Common Crawl was not used in favour of targeted Taiwanese clinical sources.

## Third-Party Copyright and Release Posture

Open access is not the same as an open licence, and withholding raw pairs does not by itself establish that model training is authorised or is fair use. Likewise, applying Apache-2.0 to model code or weights does not grant rights in upstream material.

The three explicitly copyright-restricted source groups above contributed 494 mined rows in total:

| Source Groups | Rows | Share of Specialized Training Signals |
|---|---:|---:|
| `slang-blog` + `abbr-pdf` + `rsroc-weiei` | 494 | < 0.5% of total terminology signals |

That low aggregate share, the extraction of short factual terminology rather than article text, the non-generative nature of an embedding model, and the lack of a reading substitute for the source works are relevant facts in a source-specific copyright assessment.

Accordingly:
- Raw third-party pairs and source documents are not distributed in this repository;
- Each source still requires its own licence, permission, or documented legal assessment before reuse in a new training run;
- Redistributors must independently comply with the MOE, CC BY-SA, LOINC, and SNOMED terms that apply to their inputs; and
- This provenance record is factual disclosure, not legal advice or a warranty of non-infringement.

## Model Identity & Distribution

- **Model Name:** IlhaEmbed (v2.0)
- **Base Architecture:** IBM Granite ModernBERT, Apache-2.0
- **Method:** Contrastive learning & relational knowledge distillation over Taiwanese clinical terminology, followed by vocabulary pruning and INT8 quantization
- **Published Artifact:** [`weemed/IlhaEmbed`](https://huggingface.co/weemed/IlhaEmbed) (384 dimensions, 38.5 MB INT8 ONNX)
- **Evaluation & Usage:** See [`MODEL-CARD.md`](MODEL-CARD.md) and [`README.md`](README.md)
