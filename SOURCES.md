# Training-data provenance — IlhaEmbed

This document records the source families mined during the development of
IlhaEmbed and its historical CODER-TW predecessor. It exists for provenance,
reproducibility, evaluation hygiene, and licence review.

The row counts below describe the output of each mining step at the recorded
access date (**2026-07-18**). They are not a promise that every mined row appears
with the same weight in every released checkpoint. A source used for training
must not also be treated as a held-out evaluation source.

Private generated pair files are intentionally not published. Public readers
should not need access to another private repository to understand this ledger:
public source URLs and the relevant scripts in this repository are used instead
of internal filesystem paths.

## Source families

### Specialized clinical language (surface → canonical)

| id | source | public source / acquisition | mined rows | rights status | use and notes |
|----|--------|-----------------------------|-----------:|---------------|---------------|
| moe-twblg | 教育部臺灣台語常用詞辭典 | [g0v/moedict-data-twblg](https://github.com/g0v/moedict-data-twblg), open-data dump | 778 | MOE open-data terms apply | Taigi Han text, Tâi-lô, and Mandarin definitions; 778 medical candidates filtered from 14,489 entries |
| itaigi | iTaigi 愛台語 | [itaigi.tw](https://itaigi.tw/), public platform data queried by keyword | 1,288 | Source-specific CC/publication terms require verification before redistribution | Crowd-contributed Taigi readings and votes |
| slang-blog | Taiwanese clinical slang articles | [陳志金「巷子內醫療用語」](https://snore123.blogspot.com/2019/05/medword.html), [udn article](https://blog.udn.com/ptsafetyrm/3771916), and a Vocus article | 62 | Third-party copyright; no raw redistribution permission recorded | Short surface/canonical facts such as colloquialisms and abbreviations were extracted, not the articles |
| abbr-pdf | Hospital abbreviation lists and nursing teaching material | Publicly accessible hospital/school PDF documents; extracted with `pdftotext` or OCR | 398 | Hospital/author copyright; no raw redistribution permission recorded | Abbreviation pairs only; source PDFs contained layout noise. Exact source manifests should be retained with the private research data |
| wiki-redirect | Chinese Wikipedia redirects | [MediaWiki API](https://www.mediawiki.org/wiki/API:Redirects) | 284 | CC BY-SA; attribution/share-alike obligations apply | Medical aliases mapped to article titles |
| wiki-appos | Chinese Wikipedia introductory text | [MediaWiki API](https://www.mediawiki.org/wiki/Extension:TextExtracts) plus apposition patterns | 371 | CC BY-SA; attribution/share-alike obligations apply | Phrases such as 又稱／俗稱／簡稱／縮寫為 yielded clinical abbreviation pairs |
| rsroc-weiei | 中華民國放射線醫學會衛教文章 | [rsroc.org.tw knowledge pages](https://rsroc.org.tw/knowledge/) | 34 | Society copyright; no raw redistribution permission recorded | Short factual appositions for imaging abbreviations such as LDCT, CTA, RFA, and TACE; articles are not redistributed |

### Formal terminology

| id | source | public source / acquisition | mined rows | rights status | use and notes |
|----|--------|-----------------------------|-----------:|---------------|---------------|
| icd-loinc | MOHW ICD-10-CM/PCS Chinese releases and LOINC/NHI terminology | Obtain the current releases from the respective MOHW/NHI publication channels and [LOINC](https://loinc.org/downloads/); this repository contains the transformation pipeline, not WeeMed's private source cache | 63,529 | Government-data terms and the [LOINC licence](https://loinc.org/license/) apply independently | Chinese/English cross-lingual terminology pairs |
| snomed-syn | SNOMED CT description synonyms | Obtain a licensed release through [SNOMED International](https://www.snomed.org/get-snomed) | 44,973 | SNOMED CT Affiliate Licence; not an unrestricted open-data corpus | English synonym → Fully Specified Name (FSN), the unique concept label that includes a semantic tag such as `(disorder)` or `(procedure)`; generated pairs are not distributed here |

## What was not used

- The iTaigi 2,500-seed expansion was abandoned after repeated incomplete runs;
  the 1,288-row base extraction is the recorded source.
- `icd_term_bridge` (approximately 204k rows) was dropped because token-level
  alignment produced invalid semantic pairs such as `abandonment → 照顧或`.
- Common Crawl was not used. It was considered for web-scale apposition mining,
  then deferred in favour of targeted Taiwanese clinical sources.

## Third-party copyright and release posture

Open access is not the same as an open licence, and withholding raw pairs does
not by itself establish that model training is authorised or is fair use.
Likewise, applying Apache-2.0 to model code or weights does not grant rights in
upstream material.

The three explicitly copyright-restricted source groups above contributed 494
mined rows in total:

| source groups | rows | share of the historical 119,242-pair CODER-TW mixture |
|---------------|-----:|-------------------------------------------------------:|
| slang-blog + abbr-pdf + rsroc-weiei | 494 | approximately 0.41% before training-time resampling |

That low aggregate share, the extraction of short factual terminology rather
than article text, the non-generative nature of an embedding model, and the lack
of a reading substitute for the source works are relevant facts in a
source-specific copyright assessment. They are not a blanket legal conclusion.
The assessment must also consider how much and what qualitative portion was
taken from each individual work, the applicable terms, any training-time
resampling, and market effect.

Accordingly:

- raw third-party pairs and source documents are not distributed in this
  repository;
- each source still requires its own licence, permission, or documented legal
  assessment before reuse in a new training run;
- redistributors must independently comply with the MOE, CC BY-SA, LOINC, and
  SNOMED terms that apply to their inputs; and
- this provenance record is factual disclosure, not legal advice or a warranty
  of non-infringement.

## Current released model: IlhaEmbed

The released model is **IlhaEmbed**, not CODER-TW.

- **Base:** IBM Granite ModernBERT, Apache-2.0.
- **Method:** contrastive training plus relational knowledge distillation over
  Taiwanese clinical terminology, followed by vocabulary pruning and INT8
  quantisation.
- **Published artefact:** [`weemed/IlhaEmbed`](https://huggingface.co/weemed/IlhaEmbed),
  384 dimensions, 38.5 MB INT8 ONNX.
- **Current evaluation and limitations:** see [`MODEL-CARD.md`](MODEL-CARD.md).
- **Reproducible pipeline and research history:** see [`README.md`](README.md)
  and [`docs/research/EXPERIMENTS.md`](docs/research/EXPERIMENTS.md).

The exact released-artifact recipe, evaluation split, and checksums belong in
the model card and experiment log rather than being duplicated here. This file
is the source-provenance ledger.

## Historical experiment: CODER-TW v2 (2026-07-18)

The following result is retained only as research history. It does **not**
describe the current IlhaEmbed checkpoint.

- **Base:** CODER (`GanjinZero/coder_all`, Apache-2.0), CLS pooling,
  BERT-base 768d.
- **Training mixture:** 119,242 pairs: 1,652 specialized pairs resampled eight
  times plus 108,502 formal terminology pairs; InfoNCE/in-batch negatives,
  four epochs.
- **Held-out result:** specialized top-1 0.453 fp32 / 0.433 int8;
  specialized top-5 0.628 fp32 / 0.615 int8.
- **Historical artifact:** `coder_tw_v2_int8.onnx`, 178.7 MB.

CODER-TW was a predecessor and benchmark in the development record. References
to it must not be read as the architecture, size, licence lineage, or evaluation
result of IlhaEmbed.
