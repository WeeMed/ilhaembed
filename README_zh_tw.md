# IlhaEmbed：台灣臨床術語與速記語意嵌入模型

[English](README.md) | 繁體中文

**IlhaEmbed** 是專為台灣臨床病歷、護理紀錄、社區健檢表與衛教紀錄打造的開源醫學語意嵌入向量模型系列。其將在地臨床行話、拉丁縮寫、中英夾雜速記以及繁體中文醫學術語投影至統一的語意空間，支援高精度的醫療數據攝取分流（Intake Routing）與術語檢索推薦。

名稱源自 *Ilha Formosa*（美麗島），專門讀懂這座島嶼的臨床與醫療語言。

- **發布模型：**
  - [`weemed/IlhaEmbed-311M`](https://huggingface.co/weemed/IlhaEmbed-311M) — 311M 旗艦版（高容量、16 類 FHIR 資源語意原型分流、伺服器端 API）
  - [`weemed/IlhaEmbed`](https://huggingface.co/weemed/IlhaEmbed)（別名 `weemed/IlhaEmbed-97M`）— 97M 超輕量邊緣版（36.88MB INT8 ONNX、100% FHIR 原型分流、社區健康站一體機、WASM）
- **基底架構：** ModernBERT Base & IBM Granite ModernBERT（Apache-2.0 授權）
- **主要語言：** 台灣繁體中文臨床語料（含在地台語口語、英數縮寫與中英夾雜）
- **安全定位：** 輔助建議與候選分流（Suggest-with-Review），非自主醫療診斷器材（符合 SaMD 豁免原則）

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

通用大語言模型與一般中文語意嵌入模型對此類高度專業且具地域性的縮寫與行話識別率極低。**IlhaEmbed v3** 推出雙版本體系，分別滿足雲端伺服器與邊緣離線裝置之需求：

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

> **備註說明：**
> - **台語醫學語意檢索**：評測在地台語臨床俗稱（如 `斷腦筋`、`不辣咖`）投影至標準漢字醫學概念（`中風`、`骨折`）之語意能力。純漢字轉羅馬拼音（漢羅轉寫）屬於 ASR 語音範疇，不計入本語意檢索指標。

---

## 🚀 快速開始 (Quick Start)

### 1. Python / Sentence-Transformers（支援雙版本）

```bash
pip install sentence-transformers
```

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

在無 GPU、低功耗或網路受限的院區內部與邊緣裝置上執行：

```bash
pip install onnxruntime transformers numpy
```

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# 載入剪枝後的輕量化 Tokenizer 與 36.88MB INT8 ONNX 模型
tokenizer = AutoTokenizer.from_pretrained("weemed/IlhaEmbed-97M")
session = ort.InferenceSession("model_int8.onnx", providers=["CPUExecutionProvider"])

def embed_texts(texts: list[str]) -> np.ndarray:
    inputs = tokenizer(
        texts,
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
    
    # 遮罩感知平均池化 (Mask-aware mean pooling) 與 L2 正則化
    mask = inputs["attention_mask"][:, :, None].astype(np.float32)
    sum_embs = (outputs * mask).sum(axis=1)
    sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = sum_embs / sum_mask
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norm, a_min=1e-9, a_max=None)

# 執行臨床行話檢索
vectors = embed_texts(["健檢報告 檢體未交 MIF", "L-CT 肺部電腦斷層"])
print("推論完成，向量維度:", vectors.shape)  # (2, 384)
```

---

## 💡 院內專有自訂詞彙擴充機制 (Concept Memory)

每家醫院或體系內部常有封閉的專屬縮寫。為避免將非公開詞彙硬編碼入模型權重，可採用專案提供的外掛式概念記憶庫機制（[`examples/concept_memory.py`](examples/concept_memory.py)）：

1. **嚴格相似度門禁**：僅在片段相似度高於設定閾值（如 0.85）時觸發候選替換。
2. **替換正規概念**：保留標準概念代碼，不污染原始模型推論特徵。
3. **無命中自動原樣直通**：未識別之文字安全通過，交由後續人工或規則覆核。

---

## 🔒 醫療法規、SaMD 豁免與安全邊界

- **預期用途（Intended Use）**：臨床輔助建議、攝取分流與候選重排（Suggest-with-Review Candidate Ranker）。
- **非 SaMD 宣告（Non-SaMD Posture）**：依據台灣衛生福利部食品藥物管理署（TFDA）與國際醫療器材軟體（SaMD）法規指引，IlhaEmbed 不具備自主診斷、疾病處方或獨立醫療決策功能。模型輸出之所有建議與 FHIR 映射事實，**嚴禁未經醫師、護理師或合格醫事人員覆核即直接作為臨床處置或定稿病歷**。
- **Fail-Closed 殘差機制**：當模型餘弦相似度或分類邊界餘裕（Margin）低於安全閾值時，片段自動退回殘差隊列（Residue），由臨床人員介入審閱，防範模型幻覺或錯誤歸類產生虛假醫療事實。

---

## 🛡️ 資料治理與開源邊界原則

1. **零受保護健康資訊（Zero PHI）**：模型訓練與評測全流程不包含任何真實病患姓名、身分證號、病歷號或可識別隱私個資。
2. **政府開放資料與公眾領域合規**：
   - 知識蒸餾訊號全數來自政府開放資料（數發部 MODA `taic.moda.gov.tw` 與教育部國教院 NAER 13 大類學術醫療名詞庫共 111,386 筆名詞）。
   - 考題評測語料依據《著作權法》第九條第一項第五款，採用公務人員高等考試、專門職業及技術人員考試之醫事人員試題（屬於不得為著作權之標的）。
3. **嚴禁散布未授權語料**：本開源倉庫與模型權重不散布任何第三方付費商業術語辭庫或未授權醫學期刊論文全文。
4. **授權條款**：開源程式碼與發布模型權重均採用 **Apache-2.0** 寬鬆開源授權。

---

## 引用 (Citation)

```bibtex
@misc{ilhaembed2026,
  title  = {IlhaEmbed: An Open Embedding Model for Taiwanese Clinical Text},
  author = {WeeMed AI},
  year   = {2026},
  url    = {https://huggingface.co/weemed/IlhaEmbed}
}
```
