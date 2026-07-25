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

# IlhaEmbed

English | [繁體中文](#繁體中文)

A small embedding model for Taiwanese clinical text.

Clinical records in Taiwan are full of local shorthand. `L-CT` is a low-dose lung CT screen, 鈣化 in a checkup context is a coronary artery calcium score, 皮蛇 is shingles (帶狀皰疹), 檳榔 (betel-nut chewing) is a first-class social-history axis alongside tobacco and alcohol, 傷寒 in a checkup note means the typhoid stool specimen is still outstanding (not the disease), 成健 is shorthand for an adult preventive-care checkup, and 定期心內門診-戒菸 packs two facts — ongoing cardiology follow-up and smoking cessation — into a single cell. General-purpose and general Chinese models mostly miss these, because they never saw how doctors, nurses, and care workers in Taiwan actually write. IlhaEmbed was trained for it.

The name is from Ilha Formosa. It reads this island's clinical language.

## Highlights

- **Reads local clinical writing.** Slang, abbreviations, Taigi, handwritten shorthand. Where a general model returns noise, this one usually returns the right concept.
- **Trained on Taiwanese clinical language**, not on any one institution's forms. It reads colloquialisms, appositions, and clinical abbreviations. A caveat worth stating plainly: an institution's *operational* shorthand — near-identical strings like `L-CT` (低劑量胸部電腦斷層, a lung screen) vs `H-CT` (頭部電腦斷層, a head CT) — is where a small int8 model is weakest, and it scores 83/110 on one real register. Fine-tuning that in makes it worse; a concept memory fixes it. See "Reading an institution's own shorthand" below.
- **Runs on-premise.** 38.5 MB, INT8, CPU only. No GPU, no cloud, no API key. Patient data stays in the hospital.
- **Clear provenance.** Apache-2.0 weights, a public and documented base model (IBM Granite ModernBERT), and a pipeline you can trace. Ready for a procurement or security review.
- **Open, permissively licensed.** The released weights (`weemed/IlhaEmbed`) are Apache-2.0, and
  the base model (IBM Granite) is Apache-2.0.

## What it is for

Turning messy clinical text into structured meaning. Common uses:

- Normalizing clinical text: mapping abbreviations and slang to standard concepts (L-CT to 低劑量胸部電腦斷層).
- Placing dirty data: on import, reading one cell and sending each fact to the field it belongs in (medication, condition, care source, and so on).
- Matching terms and codes: finding the right ICD, LOINC, or drug concept for a surface term.
- Searching across fields: finding a record from the clue at hand, without knowing the exact field name.

In a clinical pipeline built in Taiwan, it handles the "understanding" step: Breeze-ASR-26 for speech, IlhaEmbed for meaning, FHIR for structure.

## Results

Task: jargon top-1. Given a Taiwanese clinical surface term, retrieve its standard concept from a shared pool, self-matches excluded, held out from training. The score is macro-averaged over three semantic registers — slang, abbreviation, and apposition — where each surface form maps to a standard Han concept (皮蛇→帶狀皰疹, L-CT→低劑量胸部電腦斷層, 傷寒→傷寒篩檢糞便檢體). All models use the same method and the same pool; numbers are from the reproducible `card_eval_cpu.py`.

| Register | IlhaEmbed (fp32) | IlhaEmbed (int8) | jina-embeddings-v2-base-zh | ckip-base | bge-small-zh |
|---|---:|---:|---:|---:|---:|
| slang (皮蛇→帶狀皰疹) | **0.855** | **0.806** | 0.05 | 0.00 | 0.00 |
| abbrev (L-CT→低劑量胸部電腦斷層) | **0.817** | **0.757** | 0.14 | 0.01 | 0.00 |
| apposition (傷寒→傷寒篩檢糞便檢體) | **0.895** | **0.881** | 0.45 | 0.36 | 0.33 |
| **semantic macro** | **0.856** | **0.815** | 0.21 | 0.13 | 0.11 |
| Taigi-semantic (斷腦筋→中風, Han→Han, v0) | **0.943** | **0.929** | 0.64 | 0.56 | 0.50 |

The int8 column is measured natively via onnxruntime on the artifact published here (`model_int8.onnx`, sha256 `3449f8ba…`); the fp32 column comes from the SentenceTransformer path. The two pipelines differ, so read the int8 figures as the deployment path's own numbers rather than as a quantization delta against fp32. Baseline models are fp32 only, for an apples-to-apples reference point.

An earlier version of this card reported int8 as macro 0.810 / Taigi 0.950. Those came from a weight-interpolated variant that was **not** the model published here, and are withdrawn in favour of the figures above, re-measured on the published artifact itself.

On these local terms the general and general-Chinese models mostly return nothing; this model finds the right concept. It is not meant to top general leaderboards such as MTEB. It does one thing, Taiwanese clinical vocabulary, and it is small enough to run on the ward.

The method is reproducible. The raw evaluation pairs carry third-party copyright and are not redistributed.

### On Taigi, and why it is not folded into that number (nothing is hidden)

A fourth set, `taigi_med_lexicon`, maps a Han term to its **Tâi-lô romanization** (中風 → tiong-hong). Every model, including this one, scores ~0.00 on it — and that is correct: mapping a Han term to a romanized spelling is a **transliteration** task, not semantic retrieval. It belongs to a romanization / ASR model, not a meaning embedder, so it is reported as a diagnostic and kept **out of the semantic macro**; averaging a transliteration score into a semantic benchmark would only produce a meaningless number. (An earlier version of this card called the ~0 a "self-match bug" — that was wrong; only 31/779 surfaces collide with the pool and those are already excluded. The ~0 is the task mismatch.)

The embedder's **Taigi clinical capability is already in the numbers above**: the `slang` register *is* that capability — Taiwanese clinical colloquialisms (皮蛇 for shingles, 檳榔 for betel-nut chewing) mapped to their standard concepts, at 0.855 (fp32). A dedicated Taigi-colloquial → standard-Han-concept set (Han→Han, semantic) is the natural next benchmark to add.

## Usage

```python
# ONNX (deployment path, no torch, CPU)
from tokenizers import Tokenizer
import onnxruntime as ort, numpy as np

tok = Tokenizer.from_file("tokenizer.json")
sess = ort.InferenceSession("model_int8.onnx")
# encode (max_len 32), run, mean-pool, L2-normalize, giving a 384-d vector
```

```python
# sentence-transformers (research path)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("weemed/IlhaEmbed")
model.encode(["皮蛇", "L-CT", "定期心內門診-戒菸"])
```

384 dimensions, up to 32 tokens. Vocab 25.5k (pruned to Traditional Chinese and clinical). Size 38.5 MB (INT8 ONNX) or 152 MB (fp32 safetensors).

## How it was built

The base model is IBM Granite ModernBERT (Apache-2.0), permissively licensed. A Taiwanese clinical teacher signal supplies the domain knowledge through knowledge distillation, so the student learns to place Taiwanese slang, abbreviations, and Taigi next to their standard concepts.

The base model ships a 180k multilingual BPE vocabulary, most of which Traditional Chinese clinical text never uses. Pruning it to the 25.5k tokens that actually appear cut the size by 68 percent (123 MB to 38.5 MB) with no loss of accuracy: jargon top-1 held, and the real import routing came out bit-identical.

## Reading an institution's own shorthand — use a concept memory, do not fine-tune

The closed set of abbreviations a specific hospital writes on its forms is the hardest case for a small quantized model, and the reason is measurable. Quantizing to int8 perturbs an embedding by a fixed amount (a **noise floor**, 0.44 here). A retrieval decision is only as good as its **margin** over the runner-up, and on this kind of shorthand that margin is 0.14 — the model is deciding inside its own quantization error. Fine-tuning the shorthand into the weights shrinks the margin further while leaving the noise floor unchanged, so it makes things worse, monotonically; 18 weight-space configurations (PTQ variants, fp16, mixed precision, weight interpolation, gamma-migration, QAT, a quantization-constrained LoRA adapter) each either miss full-precision accuracy at a deployable size or trade the general register away for the domain one.

What works is to leave the encoder frozen and hold the vocabulary as data — a small external `surface → concept` memory, read through a gate:

```python
import numpy as np

MEMORY = {"L-CT": "低劑量胸部電腦斷層", "H-CT": "頭部電腦斷層", "皮蛇": "帶狀皰疹"}
surfaces, concepts = list(MEMORY), list(MEMORY.values())
keys = np.stack([encode_one(s) for s in surfaces])

def ground(fragment, tau=0.95):
    """The known concept this fragment is shorthand for, or None -> use the base embedding."""
    scores = keys @ encode_one(fragment)
    best = int(scores.argmax())
    return concepts[best] if scores[best] >= tau else None
```

Two details are load-bearing. **Gate first**: anything below τ passes through exactly as the base encoded it, which is why a memory cannot regress unrelated inputs. **Substitute the concept, not the matched key**: representing a gated fragment by its concept's embedding is what makes the decision high-margin — using the matched surface instead recovers none of the benefit.

Measured on this published artifact, with a 110-surface register:

| | base int8 | + concept memory |
|---|---:|---:|
| institutional shorthand (110 surfaces) | 83/110 (75.5%) | **110/110 (100%)** |
| held-out 20%, surfaces never in the memory | — | **20/22 (90.9%)** |
| general semantic macro (τ=0.99) | 0.815 | **0.815** — unchanged |
| Taigi-semantic | 0.929 | **0.929** — unchanged |
| size | 38.5 MB | **38.5 MB** + a few KB |

Concretely, on a 20-concept probe the base int8 model returns 頭部電腦斷層 for `L-CT` (cos 0.805 — wrong, it is a *lung* screen) and 糖尿病 for 鈣化 (cos 0.384 — wrong). With the memory both ground at 1.000, while 高血壓 / 糖尿病 / 上呼吸道感染 / 心肌梗塞 pass through untouched. `intake-embedder/concept_memory.py` in the repo is a runnable version of exactly this.

**Build the memory's keys through the same call your queries use.** This model's output depends on the batch a text was embedded in, because a batch pads to its longest member and the padding reaches the real tokens' outputs: the same string embedded inside a large batch and embedded alone are only ~0.88–0.97 similar. Mix the two paths and an exact match scores like a near-miss, and τ quietly stops meaning what it says.

**A lookup out of context is a fabrication.** 鈣化 is a coronary calcium-score exam on a checkup order line and plain tissue calcification in a radiology impression. The gate answers "which known form is this", never "does this reading apply here" — establish the context yourself first.

## Intended use and limits

This is a tool for semantic representation, not a diagnostic system. It offers likely meanings and matches for a person to confirm, and does not decide clinical facts on its own. It is tuned for Traditional Chinese clinical text in Taiwan. General prose and Simplified Chinese are out of scope.

## License

Apache-2.0. Base model: IBM Granite (Apache-2.0). Training-pair sources are listed in `SOURCES.md`. What we release is the trained weights, a distributable derivative; the raw third-party pairs are not included.

---

<a name="繁體中文"></a>

# IlhaEmbed（繁體中文）

[English](#ilhaembed) | 繁體中文

一個讀得懂台灣臨床文字的小型語意模型。

台灣的病歷、備註、社區照護表上，有很多在地的寫法。L-CT 是低劑量胸部電腦斷層，健檢紀錄裡的鈣化指的是冠狀動脈鈣化分數，皮蛇是帶狀皰疹，檳榔（嚼檳榔）是和菸酒並列的社會史軸線，健檢紀錄裡的傷寒指的是傷寒篩檢糞便檢體還沒收到（不是這個病），成健是成人預防保健的簡稱，定期心內門診-戒菸則一格裡藏了兩件事——持續心臟科門診追蹤、加上戒菸。這些寫法，通用模型和一般中文模型多半讀不出來，因為它們沒看過台灣的醫師、護理師、照服員實際怎麼寫。IlhaEmbed 是針對這件事訓練的。

名字取自 Ilha Formosa。它專門讀這座島的臨床語言。

## 特點

- **讀得懂在地的臨床寫法。** 行話、縮寫、台語、手寫簡寫。通用模型讀成雜訊的地方，它多半能對到正確的概念。
- **訓練的是台灣的臨床語言**，不是某一家機構的表格。口語、同位語、臨床縮寫都讀得到。但有一件事要說清楚：機構自己的**操作性**簡寫——像 `L-CT`（低劑量胸部電腦斷層，肺癌篩檢）和 `H-CT`（頭部電腦斷層）這種字面幾乎一樣的——正是小型 int8 模型最弱的地方，在一份真實語域上它只有 83/110。把它微調進權重會更差，用概念記憶才修得掉。見下方「讀懂機構自己的簡寫」。
- **完全在地端。** 38.5 MB、INT8、純 CPU。不用 GPU、雲端或 API key，病人資料不離開院內。
- **來源清楚。** Apache-2.0 權重，基礎模型是公開、有文件的 IBM Granite ModernBERT，訓練流程可追溯。採購或資安要看，都拿得出來。
- **開源、寬鬆授權。** 釋出的權重（`weemed/IlhaEmbed`）是 Apache-2.0，基礎模型（IBM Granite）也是 Apache-2.0。

## 用途

把凌亂的臨床文字整理成有結構的意義。常見的用法：

- 臨床文字正規化：把縮寫、行話對到標準概念（L-CT 對到低劑量胸部電腦斷層）。
- 髒資料歸位：匯入時讀一格內容，把裡面每個事實送到該去的欄位（用藥、疾病、就醫來源等）。
- 術語、代碼比對：幫一個詞找到對應的 ICD、LOINC 或藥品概念。
- 跨欄位搜尋：用手上僅有的線索找到紀錄，不必先知道正確的欄位名稱。

在一條台灣自己的臨床流程裡，它負責「理解」這一段：Breeze-ASR-26 聽打，IlhaEmbed 理解，FHIR 存成結構。

## 成效

測的是 jargon top-1：給一個台灣臨床的表面詞，從共用概念池裡找出對應的標準概念，排除自我匹配，且留作測試。分數是**三個語意語域**（行話 slang、縮寫 abbrev、同位語 apposition）的平均，每個都是「口語／台語／縮寫的漢字表面詞 → 標準漢字概念」（皮蛇→帶狀皰疹、L-CT→低劑量胸部電腦斷層、傷寒→傷寒篩檢糞便檢體）。四個模型用同一套方法、同一個池；數字來自可重現的 `card_eval_cpu.py`。

| 語域 | IlhaEmbed（fp32） | IlhaEmbed（int8） | jina-embeddings-v2-base-zh | ckip-base | bge-small-zh |
|---|---:|---:|---:|---:|---:|
| slang（皮蛇→帶狀皰疹） | **0.855** | **0.806** | 0.05 | 0.00 | 0.00 |
| abbrev（L-CT→低劑量胸部電腦斷層） | **0.817** | **0.757** | 0.14 | 0.01 | 0.00 |
| apposition（傷寒→傷寒篩檢糞便檢體） | **0.895** | **0.881** | 0.45 | 0.36 | 0.33 |
| **語意 macro** | **0.856** | **0.815** | 0.21 | 0.13 | 0.11 |
| Taigi-semantic（斷腦筋→中風，漢字→漢字，v0） | **0.943** | **0.929** | 0.64 | 0.56 | 0.50 |

int8 那欄是用 onnxruntime 在這裡發佈的那顆 artifact 上原生量的（`model_int8.onnx`，sha256 `3449f8ba…`）；fp32 那欄走的是 SentenceTransformer 管線。兩套管線不同，所以請把 int8 的數字當成部署路徑自己的成績，不要當成對 fp32 的量化落差。基準模型只有 fp32，作為對照。

這張 card 先前把 int8 報成 macro 0.810、Taigi 0.950。那是量在一個**權重內插的變體**上、不是這裡發佈的這顆模型，已經撤掉，改用上面在發佈的 artifact 本身重新量到的數字。

在這些在地詞彙上，通用模型和一般中文模型多半讀不出來，這個模型能對到正確概念。它不是要在通用排行榜（MTEB）上比高下，只把台灣臨床詞彙這一件事做好，而且小到能在病房的機器上跑。

這套方法可以重現。原始的評估配對有第三方著作權，不隨模型散佈。

### 關於台語，以及為什麼它不併進上面那個數字（沒有藏任何東西）

還有第四個集 `taigi_med_lexicon`，它是把漢字詞對到它的**台羅羅馬字**（中風 → tiong-hong）。所有模型（包含這個）在上面都是 ~0.00 —— 而這是**對的**：把漢字對到羅馬拼寫是**音譯**任務、不是語意檢索，那是羅馬字／ASR 模型的工作、不是語意 embedder 的工作。所以它只當診斷用、**不併進語意 macro**；把音譯分數平均進語意 benchmark 只會得到一個沒有意義的數字。（這張 card 之前把那個 ~0 稱作「self-match bug」，那是錯的；只有 31/779 的表面詞會撞到池子、而且早就被排除，~0 是任務錯配。）

這個模型的**台語臨床能力其實就在上面的數字裡**：`slang` 語域正是它 —— 台灣臨床口語（皮蛇＝帶狀皰疹、檳榔＝嚼檳榔）對到標準概念，fp32 0.855。之後自然要補的是一個專門的「台語口語 → 標準漢字概念」（漢字→漢字、語意）測試集。

## 用法

```python
# ONNX（部署路徑，不需 torch、CPU）
from tokenizers import Tokenizer
import onnxruntime as ort, numpy as np

tok = Tokenizer.from_file("tokenizer.json")
sess = ort.InferenceSession("model_int8.onnx")
# 編碼（max_len 32）、推論、mean-pool、L2 normalize，得到 384 維向量
```

```python
# sentence-transformers（研究路徑）
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("weemed/IlhaEmbed")
model.encode(["皮蛇", "L-CT", "定期心內門診-戒菸"])
```

維度 384，最長 32 個 token。詞表 25.5k（剪過，只留繁體中文和臨床會用到的）。大小 38.5 MB（INT8 ONNX）或 152 MB（fp32 safetensors）。

## 怎麼做的

基礎模型用 IBM Granite ModernBERT（Apache-2.0），授權寬鬆。用台灣臨床的 teacher 訊號做知識蒸餾，讓學生模型學會把台灣的行話、縮寫、台語，擺到對應的標準概念旁邊。

基礎模型原本帶 18 萬詞的多語 BPE 詞表，繁中臨床大多用不到。把它剪到實際會用到的 25.5k，體積少了 68%（123 MB 到 38.5 MB），準確度沒掉：jargon top-1 一樣，實際跑匯入的路由結果逐位相同。

## 讀懂機構自己的簡寫 —— 用概念記憶，不要微調

某一家醫院表格上那組封閉的簡寫，是小型量化模型最難的一關，而且難在哪裡是量得出來的。量化到 int8 會讓向量偏移一個固定的量（**噪音底線**，這裡是 0.44）。而一次檢索的可靠度只等於它對第二名的**餘裕（margin）**——這類簡寫的餘裕是 0.14，也就是模型在自己的量化誤差裡面做決定。把簡寫微調進權重，餘裕還會更小、噪音底線卻不動，所以只會更差，而且是單調地更差；18 種權重空間的做法（各種 PTQ、fp16、混合精度、權重內插、gamma-migration、QAT、量化受限的 LoRA adapter）沒有一種能在可部署的體積下追上全精度，不是追不上就是拿通用語域換掉領域語域。

有效的做法是讓編碼器**凍結不動**，把詞彙當資料放外面 —— 一份小的 `表面詞 → 概念` 記憶，透過一個閘門讀取：

```python
import numpy as np

MEMORY = {"L-CT": "低劑量胸部電腦斷層", "H-CT": "頭部電腦斷層", "皮蛇": "帶狀皰疹"}
surfaces, concepts = list(MEMORY), list(MEMORY.values())
keys = np.stack([encode_one(s) for s in surfaces])

def ground(fragment, tau=0.95):
    """這個片段是哪個已知概念的簡寫；不是就回 None，改用底座的向量。"""
    scores = keys @ encode_one(fragment)
    best = int(scores.argmax())
    return concepts[best] if scores[best] >= tau else None
```

兩個細節是關鍵。**先閘門**：低於 τ 的一律照底座原樣通過，所以加了記憶不可能讓無關的輸入變差。**替換成概念、不是替換成命中的那個 key**：用概念自己的向量來代表被接住的片段，才是讓決定變成高餘裕的原因——換成命中的表面詞，好處會完全消失。

在這顆發佈的 artifact 上、用一份 110 個表面詞的語域量到：

| | 底座 int8 | ＋概念記憶 |
|---|---:|---:|
| 機構簡寫（110 個表面詞） | 83/110（75.5%） | **110/110（100%）** |
| held-out 20%，記憶裡沒有的表面詞 | — | **20/22（90.9%）** |
| 通用語意 macro（τ=0.99） | 0.815 | **0.815** —— 沒退 |
| Taigi-semantic | 0.929 | **0.929** —— 沒退 |
| 體積 | 38.5 MB | **38.5 MB** ＋幾 KB |

具體一點：在一個 20 個概念的探針上，底座 int8 把 `L-CT` 讀成頭部電腦斷層（cos 0.805，錯——那是**肺部**篩檢）、把鈣化讀成糖尿病（cos 0.384，錯）。加上記憶之後兩個都以 1.000 接住，而高血壓／糖尿病／上呼吸道感染／心肌梗塞則原樣通過。repo 裡的 `intake-embedder/concept_memory.py` 就是這段的可執行版本。

**記憶的 key 要用跟 query 同一個呼叫路徑建。** 這個模型的輸出會受它所在的 batch 影響——batch 會 pad 到最長的那一筆，而 padding 會影響到真實 token 的輸出：同一個字串放在大 batch 裡跟單獨嵌入，相似度只有 ~0.88–0.97。兩條路徑混用，完全相同的字串會拿到近似命中的分數，τ 就靜靜地失去了它字面上的意思。

**沒有脈絡的查表就是捏造。** 鈣化在健檢加購欄是冠狀動脈鈣化分數檢查，在影像所見裡是組織鈣化。這個閘門回答的是「這是哪一個已知寫法」，從來不是「這個讀法在這裡成不成立」——脈絡要你自己先確立。

## 適用範圍

這是一個做語意表示的工具，不是診斷系統。它給的是可能的意義和比對，最後由人確認，不會自己下臨床判斷。它針對台灣的繁體中文臨床文字調過，一般文章或簡體中文不在範圍內。

## 授權

Apache-2.0。基礎模型 IBM Granite（Apache-2.0）。訓練配對的來源記在 `SOURCES.md`。釋出的是訓練後的權重（可散佈的衍生成果），原始的第三方配對不隨附。
