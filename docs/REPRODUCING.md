# Running the public artifacts

## Environment

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
```

For training and historical research scripts:

```bash
pip install -r requirements-training.txt
```

## Released-model smoke test

```bash
python evaluation/smoke_test.py
```

Override the Hugging Face model identifier with a local checkpoint:

```bash
ILHAEMBED_MODEL=/path/to/model python evaluation/smoke_test.py
```

## Concept-memory example

Download the ONNX model and tokenizer from the released artifact, then run:

```bash
python examples/concept_memory.py \
  --model /path/to/model_int8.onnx \
  --tokenizer /path/to/tokenizer.json
```

## Mining public or independently licensed sources

Miners never assume access to another repository. Supply all local inputs
explicitly:

```bash
python training/mining/build_bulk_pairs.py \
  --data-dir /path/to/licensed/terminology \
  --output /tmp/bulk_pairs.tsv

python training/mining/snomed_syn.py \
  --description /path/to/snomed/description.csv.gz \
  --output /tmp/snomed_syn_pairs.tsv

python training/mining/wiki_mine.py \
  --lexicon /path/to/unified_med_lexicon.tsv \
  --terminology-dir /path/to/licensed/terminology \
  --output /tmp/wiki_pairs.tsv
```

Obtaining a file and running a miner do not grant redistribution rights. Check
[`SOURCES.md`](../SOURCES.md) before using or publishing generated data.

## Historical experiments

Scripts under `training/experiments/`, `evaluation/research/`, and
`research/legacy/` document the development record. Their original inputs are
not all public. Each run therefore needs:

1. independently obtained, appropriately licensed inputs;
2. paths supplied through the script's CLI or `ILHAEMBED_*` environment
   variables;
3. a fresh held-out set that was not mined into training;
4. source-specific rights review before redistributing any generated pairs.

Do not treat a historical metric as a measurement of the current released
artifact. Current measurements belong in [`MODEL-CARD.md`](../MODEL-CARD.md).
