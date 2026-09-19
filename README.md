# IlhaEmbed

IlhaEmbed is an open embedding model for Taiwanese clinical language. It maps
local shorthand, colloquialisms, abbreviations, and Traditional Chinese clinical
terms into a shared semantic space suitable for retrieval and candidate
generation.

- **Models:**
  - [`weemed/IlhaEmbed-311M`](https://huggingface.co/weemed/IlhaEmbed-311M) — 311M Flagship (High-Capacity, 16-Category FHIR Grounded, Server API)
  - [`weemed/IlhaEmbed-97M`](https://huggingface.co/weemed/IlhaEmbed) — 97M Ultra-Lightweight (38.6MB INT8 ONNX, Edge / Kiosk / WASM)
- **Base:** ModernBERT & IBM Granite ModernBERT, Apache-2.0
- **Deployment artifacts:**
  - **311M Flagship:** 768 dimensions, ~85 MB INT8 ONNX, 100% FHIR 16-category zero-shot accuracy
  - **97M Edge:** 384 dimensions, 38.6 MB INT8 ONNX (CPU ~3.2ms), strictly <40MB edge hardware budget
- **Primary language:** Traditional Chinese clinical text used in Taiwan
- **Safety posture:** suggest-with-review, not autonomous clinical coding (SaMD-exempt assistive re-ranker)

The repository contains the public evaluation harness, source-mining utilities,
reference concept-memory implementation, provenance record, and the research
history behind the released model. It does not contain copyrighted training
pairs, institutional data, or patient data.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
```

Use the released SentenceTransformers artifact:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("weemed/IlhaEmbed")
vectors = model.encode(
    ["皮蛇", "帶狀皰疹", "L-CT", "低劑量胸部電腦斷層"],
    normalize_embeddings=True,
)
```

Run the public smoke test:

```bash
python evaluation/smoke_test.py
```

The smoke test downloads the released model from Hugging Face unless
`ILHAEMBED_MODEL` points to a local checkpoint.

## What it is good at

IlhaEmbed is designed for semantic retrieval over Taiwanese clinical text:

- colloquial surface forms such as `皮蛇 → 帶狀皰疹`;
- clinical abbreviations such as `L-CT → 低劑量胸部電腦斷層`;
- Traditional Chinese terminology and cross-register matching;
- candidate generation for a human-reviewed workflow.

It is not a diagnostic model, a generative model, or an autonomous medical
coder. General prose, Simplified Chinese, and institution-specific shorthand
outside a supplied vocabulary are not guaranteed.

The published model-card evaluation reports a Traditional Chinese clinical
semantic macro of approximately 0.82–0.86 depending on precision and execution
path. Read [`MODEL-CARD.md`](MODEL-CARD.md) for the tasks, methodology, exact
artifact measurements, and limitations.

## Institution-specific vocabulary

A hospital's closed set of local abbreviations should remain explicit data
rather than being silently baked into model weights. The reference
implementation in [`examples/concept_memory.py`](examples/concept_memory.py)
uses a conservative similarity gate:

1. match only when a surface form clears a configured threshold;
2. substitute the canonical concept, not the matched shorthand;
3. pass unmatched text through unchanged.

```bash
python examples/concept_memory.py \
  --model /path/to/model_int8.onnx \
  --tokenizer /path/to/tokenizer.json
```

The included mapping is a public demonstration. Replace it with your own
reviewed, context-scoped terminology.

## Repository map

| Path | Public purpose |
|------|----------------|
| [`evaluation/`](evaluation/) | Runnable smoke test and evaluation methodology |
| [`examples/`](examples/) | Small, self-contained integration examples |
| [`training/mining/`](training/mining/) | Source-specific corpus-mining utilities with explicit input paths |
| [`training/data_prep/`](training/data_prep/) | Public-source preparation and vocabulary utilities |
| [`training/experiments/`](training/experiments/) | Historical research trainers; these require non-published licensed data |
| [`research/legacy/coder_tw/`](research/legacy/coder_tw/) | Superseded CODER-TW experiment code |
| [`docs/research/EXPERIMENTS.md`](docs/research/EXPERIMENTS.md) | Chronological evidence log, including failed approaches |
| [`SOURCES.md`](SOURCES.md) | Corpus provenance, rights status, and release boundaries |
| [`MODEL-CARD.md`](MODEL-CARD.md) | Intended use, measurements, limitations, and artifact details |

See [`docs/REPOSITORY.md`](docs/REPOSITORY.md) for the boundary between
runnable public code, research records, and non-published inputs.

## Publishing the Hugging Face model card

[`MODEL-CARD.md`](MODEL-CARD.md) is the content source of truth for the model
card rendered on Hugging Face. Check for drift or publish it with:

```bash
pip install -r requirements-publish.txt
python tools/sync_hf_model_card.py --check
python tools/sync_hf_model_card.py --publish
# Or use an existing Git credential without exposing a token:
python tools/sync_hf_model_card.py --publish --transport git
```

Hugging Face stores the rendered card as `README.md`; the sync tool uploads the
GitHub SSOT directly and verifies the remote content after publishing. It uses
the standard `HF_TOKEN` or `hf auth login` credential flow and never handles a
token itself.

## Reproduction boundary

This is an evidence and methodology release, not a one-command reproduction of
the exact published checkpoint.

What can be reproduced from public inputs:

- loading and exercising the released model;
- the smoke-test representation checks;
- source mining where the source licence permits independent access;
- the concept-memory mechanism;
- evaluation methodology when the caller supplies an eligible evaluation set.

What is not distributed:

- third-party or institution-owned training/evaluation pairs;
- SNOMED CT or LOINC distributions that require their own terms;
- patient-identifying or organization-specific data;
- the exact private distillation run used for the released checkpoint.

The historical experiment scripts are retained for research transparency, but
their presence does not mean all required inputs are redistributable. See
[`docs/REPRODUCING.md`](docs/REPRODUCING.md) before running them.

## Data rights

Public availability is not the same as an open licence. Each input remains
subject to its own terms. This repository does not redistribute the
third-party training pairs, and the Apache-2.0 licence on repository code or
released weights does not grant rights in upstream material.

[`SOURCES.md`](SOURCES.md) records the source-level facts, including the small
share of third-party copyrighted terminology used in historical experiments.
Those facts support source-specific review; they are not a blanket legal
conclusion.

## Research history

IlhaEmbed did not begin with its current architecture. CODER-TW, BGE, CKIP,
Jina, several objectives, vocabulary pruning strategies, quantization schemes,
and concept-memory designs were evaluated along the way. Superseded results are
kept under `research/legacy/` and `docs/research/` rather than presented as the
current model.

This separation is deliberate:

- **IlhaEmbed** is the current public model.
- **CODER-TW** is a historical predecessor and benchmark.
- Product-specific routing experiments are research context, not the public
  identity of the model.

## Experimental adaptation and release checks

The September 2026 MOEX `commercial-v2` experiment is **not a released upgrade**.
Its legacy row-level split shares question families across training and evaluation;
those scores do not establish independent-question accuracy or retention of the
released model's capabilities. Keep the legacy scripts for historical reproduction.

New adaptation uses the [retained-adaptation protocol](research/CONTINUAL_ADAPTATION.md):
`training.data_prep.grouped_nursing` freezes grouped splits and excluded replay data;
`training.experiments.train_retained` trains from explicit local weights and selects
only on validation; `evaluation.check_adaptation_release` checks a selected candidate
on the frozen test and original clinical regression suite. Run these modules with
`python -m <module> --help` from the repository root. Required private datasets are
not distributed. A passing FP32 check alone does not authorize replacing the released
INT8 artifact; the exact exported model must also pass its acceptance checks.

## Citation

```bibtex
@misc{ilhaembed2026,
  title  = {IlhaEmbed: An Open Embedding Model for Taiwanese Clinical Text},
  author = {WeeMed AI},
  year   = {2026},
  url    = {https://huggingface.co/weemed/IlhaEmbed}
}
```

## Licence

Repository code is Apache-2.0 unless a file states otherwise. The released
IlhaEmbed weights and IBM Granite base are identified as Apache-2.0. Dataset
and terminology rights are separate and documented in [`SOURCES.md`](SOURCES.md).
