# IlhaEmbed — the embedding model that reads Taiwan's clinical tongue

**The first open Taiwanese clinical embedder.** It completes an all-Taiwan-blooded, sovereign,
open health-AI stack — **Breeze-ASR-26** (聯發科 / MediaTek, speech) → **IlhaEmbed** (this repo,
clinical semantics) → **MOHW TW Core FHIR** (Taiwan's national FHIR implementation guide, data
standard) — into a working, FHIR-first clinical data-transformation pipeline: hear it, understand
it, structure it.

**Weights:** [`weemed/IlhaEmbed`](https://huggingface.co/weemed/IlhaEmbed) on Hugging Face
(Apache-2.0). This repo is the **reproducible pipeline + provenance + evidence log** behind those
weights — it does not re-host them.

## Why this exists

Clinical records in Taiwan are full of local shorthand: `L-CT` is a low-dose lung CT screen, 鈣化
in a checkup context is a coronary artery calcium score, 皮蛇 is shingles (帶狀皰疹), 檳榔
(betel-nut chewing) is a first-class social-history axis alongside tobacco and alcohol, 傷寒 in a
checkup note means the typhoid stool specimen is still outstanding (not the disease), 成健 is
shorthand for an adult preventive-care checkup, and 定期心內門診-戒菸 packs two facts — ongoing
cardiology follow-up and smoking cessation — into a single cell. General-purpose and
general-Chinese embedding models mostly return nothing on this vocabulary — they never saw how
doctors, nurses, and care workers in Taiwan actually write. IlhaEmbed was built to read it.

- **Sovereign, non-PRC provenance.** Base model is [IBM Granite
  ModernBERT](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
  (Apache-2.0, US-origin). For institutions under a critical-infrastructure or sovereignty-gated
  procurement regime, model provenance is not a preference, it is a pass/fail gate — this stack
  passes it.
- **Trained on Taiwanese clinical language.** Contrastive fine-tuning plus relational distillation
  over Taiwanese clinical pairs, so the model reads registers a general Chinese model collapses —
  colloquialisms, appositions, and clinical abbreviations.
- **On-prem, CPU-deployable.** 38.5 MB, INT8 ONNX, 384-dim, no GPU and no network call at
  inference. Patient data never has to leave the building.
- **Open, permissively licensed.** The released weights (`weemed/IlhaEmbed`) are Apache-2.0, and
  the base model (IBM Granite) is Apache-2.0.

## What we learned building it: a small int8 model decides inside its own noise

Getting a small model to read *operational shorthand* — the closed set of abbreviations a specific
institution writes on its forms — looked like a fine-tuning problem. It is not, and the reason is
measurable rather than a matter of taste.

Quantizing to int8 perturbs an embedding by a fixed amount. Call that the **noise floor**. A
retrieval decision is only as reliable as the **margin** between the right concept and the runner-up.
On this shorthand the margin is several times *smaller* than the noise floor, so the model is
deciding inside its own quantization error:

| | value |
|---|---:|
| int8 embedding perturbation (noise floor) | 0.44 |
| median decision margin, direct shorthand retrieval | 0.14 |

Fine-tuning makes this **worse, monotonically**: teaching the weights the shorthand shrinks the
margin further while leaving the noise floor unchanged (measured across five fine-tuning strengths;
the int8 accuracy gap grows 3.2% → 6.2% as the weights move, r = −0.97). And it is not a knob you
can tune around — **18 weight-based configurations** (int8 per-tensor / per-channel / static QDQ,
fp16, MatMul-only and mixed-precision, weight interpolation, gamma-migration, QAT, and a
quantization-constrained LoRA adapter) all fail to reach full-precision accuracy at a deployable
size, and each one that lifts the domain register drops the general one.

The generalizable form of this: **margin per bit.** Before you quantize a small encoder, measure
whether your decisions have more margin than the quantization has noise. If they do not, no
weight-space method will save them.

## The answer: keep the encoder frozen, put the vocabulary in a memory

A closed vocabulary does not need to be *learned*. It can be *looked up*. So the domain knowledge
lives outside the weights, in a small external concept memory (`surface → concept`), read through a
two-stage gate on the **frozen** int8 encoder:

1. **Gate** — is this fragment within τ of one of my known keys? If not, it passes through exactly
   as the base encoded it, which is why adding a memory cannot regress unrelated inputs.
2. **Substitute** — a gated fragment is represented by its **concept's** embedding, not the matched
   key's. That is what turns a narrow-margin many-way decision into a high-margin one.

Both stages are load-bearing, and each was ablated: remove the gate and general performance
collapses; substitute the matched *surface* instead of the concept and the entire benefit
disappears (back to 83/110).

Measured on the exact published artifact (`model_int8.onnx`, sha256 `3449f8ba…`, 38,537,426 bytes):

| | base int8, no memory | + concept memory |
|---|---:|---:|
| institutional shorthand, 110 surfaces | 83/110 (75.5%) | **110/110 (100%)** |
| held-out 20% — surfaces never in the memory | — | **20/22 (90.9%)** |
| general clinical macro (τ=0.99) | 0.815 | **0.815** — unchanged |
| Taigi-semantic | 0.929 | **0.929** — unchanged |
| size | 38.5 MB | **38.5 MB** + a few KB |

The held-out row is the honest one: 100% on the memory's own keys is lookup, but the gate also
resolves surface variants it has never seen, via their nearest in-memory neighbour.

τ is a conservatism knob, not an accuracy trade. Raising it grounds less and converges on the
ungrounded behaviour; it never trades general accuracy for domain accuracy:

| τ | general clinical macro | general queries the gate fired on |
|---|---:|---:|
| 0.99 | 0.815 (= base) | 3 |
| 0.97 | 0.826 | 17 |
| 0.95 | 0.842 | 30 |
| 0.90 | 0.868 | 61 |

Note the direction: even when the gate fires on dozens of *general* queries, the macro goes up, not
down — the clinical memory is a useful prior for general clinical retrieval too. A deployment that
must never surprise an operator should still stay at the high end, where the gate is nearly silent.

## Results — general clinical retrieval

Task: **jargon top-1.** Given a Taiwanese clinical surface term, retrieve its standard concept
from a shared pool, self-matches excluded, held out from training. The headline score is
macro-averaged over three semantic registers — slang, abbreviation, and apposition — where each
surface form maps to a standard Han concept (皮蛇→帶狀皰疹, L-CT→低劑量胸部電腦斷層,
傷寒→傷寒篩檢糞便檢體). All models are scored with the same method and the same pool; numbers come
from the reproducible `intake-embedder/card_eval_cpu.py`.

| register | IlhaEmbed (fp32) | IlhaEmbed (int8, published) | jina-v2-base-zh | ckip-base | bge-small-zh |
|---|---:|---:|---:|---:|---:|
| slang (皮蛇→帶狀皰疹) | **0.855** | **0.806** | 0.05 | 0.00 | 0.00 |
| abbrev (L-CT→低劑量胸部電腦斷層) | **0.817** | **0.757** | 0.14 | 0.01 | 0.00 |
| apposition (傷寒→傷寒篩檢糞便檢體) | **0.895** | **0.881** | 0.45 | 0.36 | 0.33 |
| **semantic macro (headline)** | **0.856** | **0.815** | 0.21 | 0.13 | 0.11 |
| Taigi-semantic (斷腦筋→中風, Han→Han) | **0.943** | **0.929** | 0.638 | 0.560 | 0.504 |

The int8 column is measured natively via onnxruntime on the published `model_int8.onnx` itself; the
fp32 column comes from the SentenceTransformer path. The two pipelines differ, so read the int8
figures as the deployment path's own numbers rather than as a quantization delta. Baselines are fp32
only, as an apples-to-apples reference for the model itself.

Note where the int8 column is weakest — `abbrev`, 0.757. That is the register the concept memory
above exists for, and the failure is concrete rather than statistical: on a 20-concept probe the
published int8 model returns 頭部電腦斷層 for **`L-CT`** (cos 0.805, wrong — it is a *lung* screen)
and 糖尿病 for **鈣化** (cos 0.384, wrong). Both are exactly the low-margin decisions the noise floor
swallows, and both are exact after grounding.

On this vocabulary, general and general-Chinese models mostly return noise; IlhaEmbed finds the
right concept. It is not built to top general leaderboards — it does one thing, Taiwanese clinical
language, and is small enough to run on the ward.

### Honest caveats — read this before you deploy it

- **~0.82–0.86 is suggest-with-review, not autonomous coding.** Use it as a candidate generator or
  a routing signal with a human or a reranker in the loop — not as a silent auto-coder. The concept
  memory raises this to exact *only for the vocabulary you put in it*.
- **The concept memory is your data, not ours.** The 110-surface register above is one institution's
  shorthand. The mechanism generalizes; the specific mapping does not, and building yours is real
  domain work.
- **A memory is a lookup, and a lookup out of context is a fabrication.** 鈣化 means a coronary
  calcium-score exam on a checkup order line and plain tissue calcification in a radiology
  impression. The gate answers "which known form is this", never "does this reading apply here" —
  establish the context (column, document type) before consulting the memory.
- **The Taigi-semantic register is a v0.** 141 pairs, mostly derived from the
  [iTaigi](https://itaigi.tw/) crowd-sourced dictionary; a larger, professionally curated corpus is
  the intended next source, pending an access application. Expect this number to move, plausibly
  down, once the register widens — see `intake-embedder/EXPERIMENTS.md` for the full note.
- **`taigi_med_lexicon` (Han→Tâi-lô romanization) is excluded from the macro on purpose.** Mapping
  a Han term to a romanized spelling is transliteration, not semantic retrieval — every model,
  including IlhaEmbed, scores ~0 on it, and that is correct. It is a job for an ASR/romanization
  model downstream, not this one. See `intake-embedder/card_eval_cpu.py`'s docstring.
- **The bottleneck for the base encoder is data, not method.** The scarce input is real Taiwanese
  clinical colloquial pairs (~1,600 curated, vs. the reference model's ~119k). The mechanism
  (distillation + contrastive fine-tuning) is proven; scaling the data is the honest remaining work.

## Quickstart

```python
# sentence-transformers (research / fp32 path)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("weemed/IlhaEmbed")
model.encode(["皮蛇", "L-CT", "定期心內門診-戒菸"])
```

```python
# ONNX (deployment path — no torch, CPU-only)
from tokenizers import Tokenizer
import onnxruntime as ort

tok = Tokenizer.from_file("tokenizer.json")
sess = ort.InferenceSession("model_int8.onnx")
# encode (max_len 32), run, mean-pool, L2-normalize -> 384-d vector
```

Adding a concept memory is a few lines on top — no retraining, no new weights:

```python
import numpy as np

MEMORY = {"L-CT": "低劑量胸部電腦斷層", "皮蛇": "帶狀皰疹", "鈣化": "冠狀動脈鈣化分數"}
surfaces, concepts = list(MEMORY), list(MEMORY.values())
keys = np.stack([embed_one(s) for s in surfaces])       # your encode(); see note below

def ground(fragment, tau=0.95):
    """The known concept this fragment is shorthand for, or None -> use the base embedding."""
    scores = keys @ embed_one(fragment)
    best = int(scores.argmax())
    return concepts[best] if scores[best] >= tau else None
```

**Build the keys through the same call your queries use.** This model's output depends on the batch
a text was embedded in — a batch pads to its longest member and the padding reaches the real tokens'
outputs — so the same string embedded inside a large batch and embedded alone are only ~0.88–0.97
similar. Mix the two paths and an exact match scores like a near-miss, and τ silently stops meaning
what it says.

384 dimensions, up to 32 tokens, vocab pruned to 25.5k BPE (Traditional Chinese + clinical tokens
only). 38.5 MB (INT8 ONNX) or 152 MB (fp32 safetensors). Full usage details, both languages, are in
`MODEL-CARD.md` (mirrors the live Hugging Face card).

## What's here vs. what's commercial

This repository and the `weemed/IlhaEmbed` weights are the full open contribution: the base model,
the training method (contrastive fine-tuning + relational knowledge distillation from a Taiwan-origin
teacher over an unlabelled clinical-string pool), the memory-augmentation recipe above, the
evaluation harness, and the honest research record (`intake-embedder/EXPERIMENTS.md`), including
the negative results.

**Not included, and why:**

| Excluded | Reason |
|---|---|
| The raw training/eval pair files (`*.tsv` under `data/`, `intake-embedder/data/`) | Third-party / institutional copyright (hospital abbreviation sheets, blog posts, a medical society's patient-education pages) — see `SOURCES.md`. Release the derived weights, never the raw copyrighted text. |
| Any patient-identifying or organization-specific data | Never collected into this repo; the pipeline mines public dictionaries and public/licensed reference terminologies only. |
| The commercial distillation recipe (`intake-embedder/distill.py`) | Kept out of the public release deliberately (this repo's own long-standing policy — the *mechanism* is described narratively in `EXPERIMENTS.md`, the exact runnable recipe over the licensed pairs is not). |

The deployed hygieia platform, its cloud-AI tier, and the operator-confirmation flywheel that keeps
improving the model over time are a separate, commercial product built on top of this open
foundation — not part of what's in this repo.

## Repository layout

- **`scripts/`** — the mining + merge + from-scratch training pipeline: mine public sources
  (SNOMED synonyms, Chinese Wikipedia redirects/apposition, iTaigi, targeted TW-medical crawl),
  merge into training pair files, train (`train.py`, `train2.py`, `train_gpu.py`), quantize
  (`quantize.py`, `quantize_remote.py`).
- **`intake-embedder/`** — the current IlhaEmbed training + evaluation code:
  `train.py` / `train_jargon.py` / `train_sapbert.py`, `prune_vocab.py` (vocab pruning),
  `field_vocabulary.py` / `make_fda_pairs.py` / `harvest_tw_fhir.py` (public-source pair
  building), the model-card evals `card_eval_cpu.py` / `card_eval_taigi_semantic.py`, and
  benchmark scripts (`bench_all.py`, `bench_ckip.py`, `bench_coder.py`), and
  `concept_memory.py` — a runnable reference implementation of the gate described above,
  with a small public demonstration register. `EXPERIMENTS.md` is the
  round-by-round evidence log, including negative results. A handful of steps in the original
  pipeline (assembling training data from a private client corpus, harvesting real-world
  intake-column vocabulary) are internal-only and not included here — see "what's here vs.
  what's commercial" above.
- **`eval/scripts/`** — the head-to-head evaluation harness used during development (concept/care
  retrieval). `smoke_test.py` runs standalone against the released `weemed/IlhaEmbed` model; the
  others (`run_eval.py`, `concept_and_care_eval.py`) need a private evaluation corpus not included
  here (see the note at the top of each file) and are published for methodology transparency. Both
  read/write paths via `ILHAEMBED_EVAL_DATA` / `ILHAEMBED_EVAL_OUT` (default: `eval/data`,
  `eval/results` next to the script) and compare bge vs. IlhaEmbed by default — an optional,
  not-shipped legacy ONNX baseline can be added via `LEGACY_ONNX_MODEL_PATH` /
  `LEGACY_ONNX_TOKENIZER_DIR`, off unless both are set.
- **`SOURCES.md`** — per-corpus provenance for every training source: how it was obtained, its
  license, and why it is (or is not) redistributed as raw data.
- **`MODEL-CARD.md`** — the source of the public Hugging Face model card (English + 繁體中文).

## Reproduce / retrain

The pipeline is CPU/GPU-agnostic; run it on any machine with Python 3.10+, `torch`,
`transformers`, and `sentence-transformers` installed. On a CUDA GPU:

```bash
python3 scripts/build_train_data.py
python3 scripts/train_gpu.py
```

The mining scripts (`scripts/*_mine.py`, `scripts/*_harvest.py`, `scripts/*_crawl.py`) pull from
public sources directly — see `SOURCES.md` for each source's endpoint and license before
re-running them, and re-verify current terms of use, since a source's access terms can change
after this was written.

## License

Apache-2.0 (see `LICENSE`). Base model: IBM Granite (Apache-2.0). Training-pair sources are listed
in `SOURCES.md`. What is released is the trained weights (a distributable derivative) and the
pipeline/method; the raw third-party training pairs are not included.

## Citation

```bibtex
@misc{ilhaembed2026,
  title  = {IlhaEmbed: An Open Embedding Model for Taiwanese Clinical Text},
  author = {WeeMed AI},
  year   = {2026},
  url    = {https://huggingface.co/weemed/IlhaEmbed}
}
```
