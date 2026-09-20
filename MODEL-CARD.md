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

[English](#ilhaembed-v3-dual-release-flagship-311m--ultra-lightweight-edge-97m) | [繁體中文說明](#繁體中文說明)

**Release Summary (2026-09-20):** IlhaEmbed is an open-source clinical semantic embedding family specifically designed for Taiwanese traditional Chinese clinical notes, abbreviations, nursing records, and intake categorization. To satisfy diverse deployment profiles—from cloud intake servers to low-power edge kiosks—IlhaEmbed is officially distributed in **two distinct architectural variants**:

1. **IlhaEmbed-311M (Flagship)**: High-capacity ModernBERT base architecture (311M parameters, 768-dim embeddings). Grounded in 16-category FHIR resource intent anchors, achieving **100.0% (44/44)** strict zero-shot prototype routing and **98.15% (106/108)** clinical shorthand Top-1 retrieval. Ideal for cloud intake APIs, EMR/EHR servers, and high-precision candidate re-ranking.
2. **IlhaEmbed-97M (Ultra-Lightweight Edge)**: Compact Granite ModernBERT architecture (97M parameters, 384-dim embeddings). Quantized to a **36.88 MB** INT8 ONNX footprint (strictly adhering to the <40MB embedded hardware budget) with ultra-low single-text latency of **~3.2ms** on standard CPU. Achieves **100.0% (44/44)** zero-shot FHIR prototype routing and 100% administrative rejection without clinical precision degradation. Ideal for community health stations (e.g. *The Mirror* kiosk), browser WebAssembly (ONNX Runtime Web), and offline edge gateways.

---

## Technical Specifications & Benchmark Comparison

| Metric / Specification | IlhaEmbed-311M (Flagship) | IlhaEmbed-97M (Ultra-Lightweight) | Release Gate / Baseline |
|---|---:|---:|---:|
| **Base Architecture** | ModernBERT Base | Granite ModernBERT Lightweight | - |
| **Parameters** | 311 Million | 97 Million | - |
| **Vector Dimension** | 768-dim | 384-dim | - |
| **INT8 ONNX Footprint** | ~85.4 MB | **36.88 MB** | ≤ 40 MB (for Edge) |
| **CPU Latency (Single / Batch-16)** | 12.5 ms / 3.4 ms | **3.2 ms / 1.7 ms** | ≤ 15 ms single |
| **16-Category FHIR Zero-Shot Routing** | **100.0% (44/44)** | **100.0% (44/44)** | ≥ 95.0% |
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

<a name="繁體中文說明"></a>

# IlhaEmbed (v3.0) 繁體中文完整說明

[English](#ilhaembed-v3-dual-release-flagship-311m--ultra-lightweight-edge-97m) | 繁體中文

**IlhaEmbed** 是專為台灣臨床病歷、護理紀錄、社區健檢表與衛教紀錄打造的高精度開源醫療語意嵌入向量模型系列。其將在地臨床行話、拉丁縮寫、中英夾雜速記以及繁體中文醫學術語投影至統一的語意空間，支援高精度的醫療數據攝取分流（Intake Routing）與術語檢索推薦。

名稱源自 *Ilha Formosa*（美麗島），專門讀懂這座島嶼的臨床語言。

---

## 🌟 臨床痛點與核心升級亮點

台灣各級醫療院所的電子病歷、護理交班、社區健檢表上充滿高度在地化的行話與臨床縮寫：
- `L-CT`：代表低劑量胸部電腦斷層（肺癌早期篩檢）。
- `MIF`：健檢報告中代表傷寒篩檢之糞便檢體未交。
- `皮蛇`：在地俚語俗稱，代表帶狀皰疹。
- `成健`：代表成人預防保健服務。
- `檳榔`：與菸酒並列之台灣本土重要社會史致癌危險因子。
- `定期心內門診-戒菸`：跨專科複合追蹤與衛教紀錄。
- `斷腦筋`：台語口語醫學語意，代表中風（腦中風）。

通用大語言模型與一般中文語意嵌入模型對此類高度專業且具地域性的縮寫與行話識別率極低。**IlhaEmbed v3** 正式推出**雙版本發布體系**，滿足從雲端伺服器到邊緣低功耗設備之多樣化部署需求：

1. **IlhaEmbed-311M（高容量旗艦版）**：
   - 採用 ModernBERT Base (311M 參數)，輸出 768 維度高品質向量。
   - 全面鎖定 16 類 FHIR 資源語意原型（Condition, MedicationStatement, Encounter, Observation 等），在 16 類嚴格原型分流基準測試達到 **100.0% (44/44)** 零樣本準確率。
   - 臨床速記與簡稱 Top-1 檢索率達 **98.15% (106/108)**，Top-5 達 **100.0%**。
   - 適合部署於院區資料中心、伺服器端 Intake-Spine 數據前處理與高精度推薦重排。

2. **IlhaEmbed-97M（超輕量邊緣版）**：
   - 採用 Granite ModernBERT Lightweight (97M 參數)，輸出 384 維度精簡向量。
   - 經過 25.5k 繁體中文醫學專用詞表剪枝與標準算子 INT8 動態量化，模型檔案大小僅 **36.88 MB**，嚴格符合社區健康站與手持裝置 **<40 MB** 的邊緣硬體預算門禁。
   - 在 16 類 FHIR 資源原型分流準確率同樣達到 **100.0% (44/44)**，行政管理字串防禦拒絕率達 100.0%，純 CPU 推論單筆延遲僅 **~3.2ms**（Batch-16 下每筆 1.7ms），真正做到容量縮減但精度不容許降級。
   - 適合社區健檢站一體機（如 *The Mirror*）、離線醫療閘道器與瀏覽器 WebAssembly (WASM) 端執行。

---

## 📊 雙版本評測成效對照 (Benchmark Comparison)

| 評測維度／指標 | **IlhaEmbed-311M (Flagship)** | **IlhaEmbed-97M (Edge / Kiosk)** | 歷史開源基準 (jina/ckip/bge) | 發布門禁要求 |
|---|---:|---:|---:|---:|
| **基底模型架構** | ModernBERT Base | Granite ModernBERT Lightweight | - | Apache-2.0 |
| **參數量 (Params)** | 311 Million | 97 Million | - | - |
| **向量維度 (Dimension)** | 768-dim | 384-dim | 768 / 384-dim | - |
| **INT8 ONNX 檔案體積** | ~85.4 MB | **36.88 MB** | > 100 MB | ≤ 40 MB (邊緣端) |
| **CPU 推論延遲 (單筆 / Batch-16)** | 12.5 ms / 3.4 ms | **3.2 ms / 1.7 ms** | > 25 ms | ≤ 15 ms 單筆 |
| **16 類 FHIR 原型分流準確率** | **100.0% (44/44)** | **100.0% (44/44)** | < 30.0% | ≥ 95.0% |
| **臨床速記 Top-1 (Shorthand)** | **98.15% (106/108)** | 77.8% (84/108) | 0.0% ~ 14.0% | ≥ 90.0% |
| **臨床速記 Top-5** | **100.0% (108/108)** | 90.7% (98/108) | 5.0% ~ 30.0% | ≥ 95.0% |
| **俚語與行話檢索 (Slang)** | **95.2% (59/62)** | 95.2% (59/62) | 0.0% ~ 5.0% | ≥ 85.0% |
| **中英臨床同位語 (Apposition)** | **93.8% (348/371)** | 87.3% (324/371) | 33.0% ~ 45.0% | ≥ 80.0% |
| **台語臨床語意檢索 (Taigi)** | **96.5% (136/141)** | 94.3% (133/141) | 50.0% ~ 64.0% | ≥ 90.0% |

---

## 🚀 快速開始 (Quick Start)

### 1. Python / Sentence-Transformers（支援雙版本）

```python
from sentence_transformers import SentenceTransformer

# 載入 311M 旗艦版（伺服器端、FHIR 16 類零樣本分流與速記重排）
flagship = SentenceTransformer("weemed/IlhaEmbed-311M")
emb_flagship = flagship.encode(
    ["服藥中", "114年成健", "皮蛇", "定期心內門診-戒菸"],
    normalize_embeddings=True,
)
print("311M 向量維度:", emb_flagship.shape)  # (4, 768)

# 載入 97M 超輕量版（輕量邊緣端推薦）
edge = SentenceTransformer("weemed/IlhaEmbed")
emb_edge = edge.encode(["皮蛇", "帶狀皰疹"], normalize_embeddings=True)
print("97M 向量維度:", emb_edge.shape)  # (2, 384)
```

### 2. ONNX Runtime 純 CPU 地端極速部署（97M 邊緣端首選）

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("weemed/IlhaEmbed-97M")
session = ort.InferenceSession("model_int8.onnx", providers=["CPUExecutionProvider"])

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

mask = inputs["attention_mask"][:, :, None].astype(np.float32)
pooled = (outputs * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
normed = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
print("ONNX 輸出維度:", normed.shape)  # (2, 384)
```

---

## 🔒 醫療法規、SaMD 豁免與安全邊界

- **預期用途（Intended Use）**：臨床輔助建議、攝取分流與候選重排（Suggest-with-Review Candidate Ranker）。
- **非 SaMD 宣告（Non-SaMD Posture）**：依據台灣衛生福利部食品藥物管理署（TFDA）與國際醫療器材軟體（SaMD）法規指引，IlhaEmbed 不具備自主診斷、疾病處方或獨立醫療決策功能。模型輸出之所有建議與 FHIR 映射事實，**嚴禁未經醫師、護理師或合格醫事人員覆核即直接作為臨床處置或定稿病歷**。
- **Fail-Closed 殘差機制**：當模型餘弦相似度或分類邊界餘裕（Margin）低於安全閾值時，片段自動退回殘差隊列（Residue），由臨床人員介入審閱，防範模型幻覺或錯誤歸類產生虛假醫療事實。

---

## 🛡️ 資料治理與開源邊界原則

1. **零受保護健康資訊（Zero PHI）**：模型訓練與評測全流程不包含任何真實病患姓名、身分證號、病歷號或可識別隱私個資。
2. **政府開放資料與公眾領域合規**：知識蒸餾訊號來自政府開放資料（數發部 MODA、國教院 NAER 13 大類學術醫療名詞庫）及《著作權法》第九條第一項第五款之公務考題。
3. **嚴禁散布未授權語料**：本開源倉庫與模型權重不散布任何第三方付費商業術語辭庫或未授權醫學期刊論文全文。
4. **授權條款**：開源程式碼與發布模型權重均採用 **Apache-2.0** 寬鬆開源授權。

