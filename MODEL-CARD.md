---
license: apache-2.0
language:
  - zh
  - en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - embeddings
  - traditional-chinese
  - medical
  - taiwan
  - onnx
base_model: ibm-granite/granite-embedding-97m-multilingual-r2
---

# IlhaEmbed v3 Dual Release: Flagship (311M) & Ultra-Lightweight Edge (97M)

**Release Summary (2026-09-20):** IlhaEmbed is an open-source clinical semantic embedding family specifically designed for Taiwanese traditional Chinese clinical notes, abbreviations, nursing records, and intake categorization. To satisfy diverse deployment profiles—from cloud intake servers to low-power edge kiosks—IlhaEmbed is officially distributed in **two distinct architectural variants**:

1. **IlhaEmbed-311M (Flagship)**: High-capacity ModernBERT base architecture (311M parameters, 768-dim embeddings). Grounded in 16-category FHIR resource intent anchors, achieving **100.0% (44/44)** strict zero-shot prototype routing and **98.15% (106/108)** clinical shorthand Top-1 retrieval. Ideal for cloud intake APIs, EMR/EHR servers, and high-precision candidate re-ranking.
2. **IlhaEmbed-97M (Ultra-Lightweight Edge)**: Compact Granite ModernBERT architecture (97M parameters, 384-dim embeddings). Quantized to a **38.6 MB** INT8 ONNX footprint (strictly adhering to the <40MB embedded hardware budget) with ultra-low single-text latency of **~3.2ms** on standard CPU. Ideal for community health stations (e.g. *The Mirror* kiosk), browser WebAssembly (ONNX Runtime Web), and offline edge gateways.

---

## Technical Specifications & Benchmark Comparison

| Metric / Specification | IlhaEmbed-311M (Flagship) | IlhaEmbed-97M (Ultra-Lightweight) | Release Gate / Baseline |
|---|---:|---:|---:|
| **Base Architecture** | ModernBERT Base | Granite ModernBERT Lightweight | - |
| **Parameters** | 311 Million | 97 Million | - |
| **Vector Dimension** | 768-dim | 384-dim | - |
| **INT8 ONNX Footprint** | ~85.4 MB | **38.66 MB** | ≤ 40 MB (for Edge) |
| **CPU Latency (Single / Batch-16)** | 12.5 ms / 3.4 ms | **3.2 ms / 1.7 ms** | ≤ 15 ms single |
| **16-Category FHIR Zero-Shot Routing** | **100.0% (44/44)** | 75.0% (33/44) | ≥ 95.0% (Flagship) |
| **Clinical Shorthand Top-1** | **98.15% (106/108)** | 77.8% (84/108) | ≥ 90.0% |
| **Clinical Shorthand Top-5** | **100.0% (108/108)** | 90.7% (98/108) | ≥ 95.0% |
| **Colloquial / Slang Retrieval** | **95.2% (59/62)** | 95.2% (59/62) | ≥ 85.0% |
| **Bilingual Appositions** | **93.8% (348/371)** | 87.3% (324/371) | ≥ 80.0% |
| **Taigi Medical Semantics** | **96.5% (136/141)** | 94.3% (133/141) | ≥ 90.0% |

---

## Safety & Regulatory Boundary (SaMD Exemption)

- **Intended Use**: Assistive terminology alignment, semantic routing, and candidate recommendation (Suggest-with-Review).
- **Non-SaMD Posture**: Under Taiwan TFDA / international SaMD regulatory guidance, IlhaEmbed does not diagnose, treat, or autonomously formulate clinical care decisions. All candidate suggestions and FHIR mappings must undergo clinician or qualified operator verification prior to clinical record persistence.
- **Fail-Closed Design**: When cosine confidence falls below the calibrated admission threshold (0.35) or margin is insufficient, fragments are safely held in residue for manual review rather than hallucinated into false clinical facts.

---

## Usage

### 1. Flagship (311M) — Python / Sentence-Transformers
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("weemed/IlhaEmbed-311M")
embeddings = model.encode(
    ["服藥中", "114年成健", "皮蛇", "定期心內門診-戒菸"],
    normalize_embeddings=True,
)
print(embeddings.shape)  # (4, 768)
```

### 2. Edge (97M) — INT8 ONNX CPU Execution
```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("weemed/IlhaEmbed-97M")
session = ort.InferenceSession(
    "model_int8.onnx", providers=["CPUExecutionProvider"]
)

inputs = tokenizer(
    ["皮蛇", "帶狀皰疹"],
    max_length=32,
    padding="max_length",
    truncation=True,
    return_tensors="np",
)
feed = {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
}
outputs = session.run(["last_hidden_state"], feed)[0]
# Mask-aware mean pooling & L2 normalization
mask = inputs["attention_mask"][:, :, None].astype(np.float32)
pooled = (outputs * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
normed = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
print(normed.shape)  # (2, 384)
```

---

## Data Governance & Open-Source Principles

1. **Zero Protected Health Information (PHI)**: No patient records, electronic medical records (EMR), or private institutional data are included in training datasets or model checkpoints.
2. **Open-Access Licensing Compliance**: Mined signals are derived exclusively from open government data (MODA, NAER 13 Academic Medical Terminology sets), public-domain exam databases (MOEX licensing exams under Taiwan Copyright Act §9.1.5), and licensed terminology descriptions.
3. **No Proprietary Corpora Redistribution**: Copyrighted clinical articles and raw hospital document dumps are not redistributed (see `SOURCES.md`).
4. **License**: Code and published weights are licensed under **Apache-2.0**.

---

## 繁體中文摘要

IlhaEmbed v3 正式推出**雙版本模型發布**：
1. **IlhaEmbed-311M（旗艦版）**：採用 ModernBERT Base (311M 參數)，對齊 16 類 FHIR 資源語意，於 16 類原型零樣本分流達到 **100.0% (44/44)** 準確率，臨床速記 Top-1 達到 **98.15%**。適合雲端伺服器、醫院病歷攝取管線與高精度推薦重排。
2. **IlhaEmbed-97M（超輕量邊緣版）**：採用 Granite ModernBERT (97M 參數)，INT8 量化體積嚴格限制於 **38.66 MB**（低於 40MB 邊緣硬體預算門禁），單筆 CPU 延遲僅 **3.2ms**。適合社區健康站一體機、瀏覽器 WebAssembly (WASM) 與地端離線裝置。

本系列模型嚴格遵循輔助建議（Suggest-with-Review）與人機協同定位，非自主診斷醫材（符合 SaMD 豁免原則），並堅守零 PHI、無未授權專利語料之安全開源邊界。
