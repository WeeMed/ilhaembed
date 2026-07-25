# Training-data provenance — TW clinical terminology embedder

Every corpus mined for the CODER-TW fine-tune, recorded for **ground-truth /
reproducibility / licensing**. Rule: anything listed here is IN the training
data and must NOT be reused as a held-out test set. Access date: **2026-07-18**.

## Specialized (surface → canonical) — the scarce clinical signal

| id | source | how obtained | endpoint / file | rows | license | notes |
|----|--------|--------------|-----------------|-----:|---------|-------|
| moe-twblg | 教育部臺灣台語常用詞辭典 | open-data dump | `github.com/g0v/moedict-data-twblg` → `dict-twblg.json` | 778 | MOE open data | Taigi 漢字+台羅+華語定義; medical-regex filtered from 14,489 |
| itaigi | iTaigi 愛台語 (g0v) | reverse-eng API | `itaigi.tw/平臺項目列表/揣列表?關鍵字=` | 1,288 | CC (條目標「會使公開」) | crowd Taigi readings + votes |
| slang-blog | 陳志金「巷子內醫療用語」/ udn 詹廖明義 / vocus | manual WebFetch | `snore123.blogspot.com/2019/05/medword.html`, `blog.udn.com/ptsafetyrm/3771916`, `vocus.cc/article/6541c172…` | 62 | **作者著作權** | 口語黑話(摸咪/掐水/歐卡)+書面(Endo/Foley/NG) |
| abbr-pdf | 醫院「可使用縮寫表」+ 護理教材 | curl + pdftotext | nutc, mhchcm, sijhih, kmu(失敗), wagners(需OCR), hpa | 398 | **醫院/作者著作權** | PDF 抽取,有版面噪音 |
| wiki-redirect | 中文維基百科 重定向 | MediaWiki API | `zh.wikipedia.org/w/api.php prop=redirects` | 284 | CC BY-SA | 別名→條目;醫學 redirect 覆蓋稀疏 |
| wiki-appos | 中文維基百科 內文同位語 | MediaWiki API | `…prop=extracts&exintro` + Hearst patterns | 371 | CC BY-SA | 「又稱/俗稱/簡稱/縮寫為」→ 挖出 CVA/COPD/心梗 等縮寫 |
| rsroc-weiei | 中華民國放射線醫學會 衛教 | curl crawl | `rsroc.org.tw/knowledge/news/content.asp?ID=1..119` | 34 | **學會著作權** | 影像縮寫 LDCT/CTA/RFA/TACE;高精度 apposition |

## Bulk (formal synonym) — abundant, saturates cross-lingual

| id | source | how obtained | file | rows | license | notes |
|----|--------|--------------|------|-----:|---------|-------|
| icd-loinc | 衛福部 ICD-10-CM/PCS 中文版 + LOINC-NHI | hygieia local | `core/data/icd10_cm_2023.csv.gz` 等 | 63,529 | gov public / LOINC | zh↔en cross-lingual pairs |
| snomed-syn | SNOMED CT description synonyms | hygieia local | `core/data/terminology_sources/snomed/description.csv.gz` | 44,973 | **SNOMED CT license (UMLS/UTS)** | en synonym→FSN, high-signal subset |

## Not used / dropped (recorded so they aren't re-attempted blindly)
- iTaigi 2,500-seed expansion — process kept dying mid-run; 1,288 base run already folded in.
- icd_term_bridge.csv.gz (204k) — token-level alignment noise ("abandonment→照顧或"), unusable.
- Common Crawl — the right web-scale corpus for the apposition pattern, but a
  petabyte S3/Athena/Spark project; deferred. Targeted TW-domain crawl is the
  lighter substitute (rsroc above; extend to more hospital 衛教 domains next).

## Open-source caveat (carried from core/data/README)
Bulk gov/CC/open-licensed parts are redistributable; the **abbr-pdf / slang-blog
/ rsroc** raw text is third-party copyright — release the *trained embedder
weights* (derived work), not the raw pairs, unless per-source consent obtained.

## Model produced from these corpora (v2, 2026-07-18)

- **Base**: CODER (`GanjinZero/coder_all`, Apache-2.0), CLS pooler, BERT-base 768d.
- **Train**: 119,242 pairs = specialized 1,652×8 (upsampled) + bulk 108,502;
  InfoNCE / in-batch negatives, 4 epochs, RTX 4080.
- **Held-out (247 specialized, 500 xling), never trained on:**
  | | base | v2 fp32 | v2 int8 |
  |---|---|---|---|
  | specialized top1 | 0.279 | 0.453 | 0.433 |
  | specialized top5 | 0.429 | 0.628 | 0.615 |
  | xling top1 | 0.708 | 0.978 | — |
- **Deployable**: `coder_tw_v2_int8.onnx` 178.7 MB (25% of fp32), CPU inference, fits 2 GB tier.
- **Honest limits**: specialized top1 0.43 = usable for a suggest-with-review tier,
  not autonomous. Chinese colloquial signal is web-scale-sparse (see Common Crawl
  note); the gains came mostly from apposition-mined clinical abbreviations.
