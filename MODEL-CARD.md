---
license: apache-2.0
language:
  - zh
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - embeddings
  - clinical
  - healthcare
  - traditional-chinese
  - taiwan
  - medical
  - fhir
  - on-premise
  - onnx
base_model: ibm-granite/granite-embedding-97m-multilingual-r2
---

# IlhaEmbed (v2.0)

English | [繁體中文](#繁體中文)

A small, high-precision domain-adapted embedding model tailored for Taiwanese clinical records, health checkups, and medical terminology normalization.

Clinical records and health checkup notes in Taiwan are saturated with local jargon, hospital shorthand, medical abbreviations, and Taigi colloquialisms. `L-CT` represents low-dose chest CT screening (低劑量胸部電腦斷層), `MIF` in a checkup note signifies pending typhoid stool specimen tracking (傷寒篩檢糞便檢體), `皮蛇` refers to shingles (帶狀皰疹), `成健` is shorthand for adult preventive health checkups (成人預防保健), and `檳榔` (betel-nut chewing) is a dedicated social-history risk axis alongside tobacco and alcohol. General-purpose multilingual models consistently fail to resolve these domain terms because they were never exposed to how Taiwanese medical practitioners actually write. **IlhaEmbed v2.0** was built specifically to solve this problem on-premise without cloud API dependencies.

The name originates from *Ilha Formosa*. It reads this island's medical and clinical language.

---

## 🌟 Key Highlights (v2.0 Release)

- **NAER & MODA 13-Set Academic Medical Terminology Retrained (v2.0)**: Fully retrained with the Ministry of Digital Affairs (MODA) and National Academy for Educational Research (NAER) 13-set official medical taxonomy (`taic.moda.gov.tw`, 111,386 concepts), dramatically improving coverage over anatomy, pathology, pharmacology, and clinical lab standards.
- **Context-Conditioned Acronym & Polysemy Resolution**: Pure Latin acronyms (`MIF`, `CBC`, `BUN`, `Cr`, `eGFR`) and polysemous terms (`鈣化`) are accurately disambiguated by appending operational field context (e.g. `f"{column_hint} {fragment}"`), achieving high margin separation (+0.15 score lead over unconditioned baselines).
- **Reads Local Taiwanese Clinical & Hospital Shorthand**: Specially handles Taiwanese clinical slang (`皮蛇` → `帶狀皰疹`), departmental abbreviations (`定期心內門診-戒菸`), and Taigi colloquialisms (`斷腦筋` → `中風`).
- **On-Premise & Edge Native**: 38.5 MB (INT8 ONNX), CPU-only inference via `onnxruntime` or `sentence-transformers`. Zero GPU requirements, zero cloud dependencies, zero external API keys. Patient PHI never leaves hospital premises.
- **Documented Provenance & Open Rights**: Licensed under Apache-2.0. Upstream open-government data provenance is explicitly cataloged in the public [source ledger](https://github.com/WeeMed/ilhaembed/blob/main/SOURCES.md).

---

## 🎯 Primary Use Cases

1. **Clinical Text Normalization**: Automatically mapping raw shorthand and abbreviations to canonical concepts (e.g., `L-CT` → `低劑量胸部電腦斷層`, `MIF` → `傷寒篩檢糞便檢體`).
2. **Health Checkup Data Intake (Intake-Spine)**: Structuring dirty, unstructured report cells on import and dispatching facts into standard FHIR resource fields (Condition, MedicationStatement, Observation).
3. **Medical Terminology & Code Matching**: Matching colloquial doctor notes to standard ICD-10, LOINC, or FDA drug codes.
4. **Cross-Field Search from User Clues**: Locating patient records using natural clinical shorthand without knowing exact database column names.

In Taiwan's sovereign AI medical stack, **IlhaEmbed** handles the semantic comprehension layer: **Breeze-ASR-26** for speech transcription, **IlhaEmbed v2.0** for concept normalization, and **FHIR** for structured interoperability.

---

## 📊 Benchmark Results

**Task**: Jargon Top-1 Concept Retrieval. Given a Taiwanese clinical surface term, retrieve its canonical concept from a shared pool (self-matches excluded, held-out validation set).

| Semantic Register | IlhaEmbed v2 (fp32) | IlhaEmbed v2 (int8 ONNX) | jina-embeddings-v2-base-zh | ckip-base | bge-small-zh |
|---|---:|---:|---:|---:|---:|
| **Slang** (`皮蛇` → `帶狀皰疹`) | **0.855** | **0.806** | 0.05 | 0.00 | 0.00 |
| **Abbreviation** (`L-CT` → `低劑量胸部電腦斷層`) | **0.817** | **0.757** | 0.14 | 0.01 | 0.00 |
| **Apposition** (`傷寒` → `傷寒篩檢糞便檢體`) | **0.895** | **0.881** | 0.45 | 0.36 | 0.33 |
| **Semantic Macro Average** | **0.856** | **0.815** | 0.21 | 0.13 | 0.11 |
| **Taigi Clinical Semantic** (`斷腦筋` → `中風`) | **0.943** | **0.929** | 0.64 | 0.56 | 0.50 |

---

## 🚀 Quickstart & Usage

### 1. ONNX CPU Edge Deployment (Recommended for On-Premise Production)

```python
from tokenizers import Tokenizer
import onnxruntime as ort, numpy as np

# Load pruned 25.5k vocabulary tokenizer and 38.5MB INT8 ONNX engine
tok = Tokenizer.from_file("tokenizer.json")
session = ort.InferenceSession("model_int8.onnx")

def get_embedding(text: str) -> np.ndarray:
    encoded = tok.encode(text)
    input_ids = np.array([encoded.ids[:32]], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask[:32]], dtype=np.int64)
    
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    vec = outputs[0].mean(axis=1)[0]
    return vec / np.linalg.norm(vec)

# Vector length: 384-dim normalized vector
vec = get_embedding("健檢報告 檢體未交 MIF")
```

### 2. Sentence-Transformers Usage (Python Research Path)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("weemed/IlhaEmbed")
embeddings = model.encode(
    ["皮蛇", "L-CT", "健檢報告 檢體未交 MIF", "定期心內門診-戒菸"],
    normalize_embeddings=True
)
print(embeddings.shape)  # (4, 384)
```

---

## 🔒 Data Provenance & Open Data

IlhaEmbed incorporates knowledge distillation signals from Taiwan open government datasets, including the Ministry of Digital Affairs (MODA) 13-set Academic Medical Terminology corpus (`taic.moda.gov.tw`, 111,386 concepts). Raw training corpora remain subject to upstream open data terms (MODA Open Data Terms §3.1).

---

## ⚖️ License

Repository code and released weights are licensed under **Apache-2.0**.
Base Model: IBM Granite ModernBERT (Apache-2.0).

---

<a name="繁體中文"></a>

# IlhaEmbed (v2.0) 繁體中文說明

[English](#ilhaembed-v20) | 繁體中文

**IlhaEmbed v2.0** 是專為台灣臨床病歷、健檢報告與醫療術語正規化打造的高精度小型領域嵌入向量模型 (Domain-Adapted Embedding Model)。

台灣的醫院病歷、護理紀錄、社區健檢表上充滿在地行話與臨床縮寫：`L-CT` 代表低劑量胸部電腦斷層（肺癌篩檢）、健檢報告中的 `MIF` 代表傷寒篩檢糞便檢體未交、`皮蛇` 代表帶狀皰疹、`成健` 代表成人預防保健、`檳榔` 則是與菸酒並列的獨立社會史危險因子。通用大語言模型與一般中文模型對此類在地簡寫識別率極低。**IlhaEmbed v2.0** 專為解決此痛點而生，支援純 CPU 地端運作，零雲端依賴，保障病人 PHI 隱私不出院區。

模型名稱源自 *Ilha Formosa*（美麗島），專門讀懂這座島嶼的臨床語言。

---

## 🌟 v2.0 核心升級亮點

- **教育部/國教院 (NAER) 與數發部 (MODA) 13 大類學術醫療名詞庫重訓練**：全量導入數發部 `taic.moda.gov.tw` 111,386 筆官方醫療名詞，大幅強化解剖學、病理學、藥理學與臨床檢驗標準詞庫對齊。
- **Context-Conditioned 帶上下文動態解歧義**：對純拉丁縮寫 (`MIF`, `CBC`, `BUN`, `Cr`, `eGFR`) 與多義詞 (`鈣化`)，透過傳入業務欄位 Context (如 `f"{column_hint} {fragment}"`)，實現無字典硬編碼的高餘裕精確判讀。
- **在地化臨床與健檢簡寫精確識別**：完整覆蓋台灣臨床行話 (`皮蛇` → `帶狀皰疹`)、跨科別門診紀錄 (`定期心內門診-戒菸`) 與台語口語 (`斷腦筋` → `中風`)。
- **極致輕量地端原生 (On-Premise Native)**：INT8 ONNX 體積僅 **38.5 MB**，純 CPU 即可達秒級推論。無需 GPU、無需 API 金鑰，資料完全保留在醫院內網。
- **透明透明的資料來源清冊 (Data Provenance)**：模型權重採 Apache-2.0 授權，訓練語料來源載明於公開 [SOURCES.md 清冊](https://github.com/WeeMed/ilhaembed/blob/main/SOURCES.md)。

---

## 📊 評測成效 (Jargon Top-1 Accuracy)

| 評測語域 (Semantic Register) | IlhaEmbed v2 (fp32) | IlhaEmbed v2 (int8 ONNX) | jina-embeddings-v2-base-zh | ckip-base | bge-small-zh |
|---|---:|---:|---:|---:|---:|
| **臨床行話** (`皮蛇` → `帶狀皰疹`) | **0.855** | **0.806** | 0.05 | 0.00 | 0.00 |
| **醫療縮寫** (`L-CT` → `低劑量胸部電腦斷層`) | **0.817** | **0.757** | 0.14 | 0.01 | 0.00 |
| **臨床同位語** (`傷寒` → `傷寒篩檢糞便檢體`) | **0.895** | **0.881** | 0.45 | 0.36 | 0.33 |
| **整體語意 Macro 平均** | **0.856** | **0.815** | 0.21 | 0.13 | 0.11 |
| **台語臨床語意** (`斷腦筋` → `中風`) | **0.943** | **0.929** | 0.64 | 0.56 | 0.50 |

---

## 📜 授權條款

本模型權重與開源程式碼採用 **Apache-2.0** 寬鬆授權。
基礎模型：IBM Granite ModernBERT (Apache-2.0)。
