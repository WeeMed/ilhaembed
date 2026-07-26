# Repository structure and publication boundary

This repository is organized for readers who do not have access to any WeeMed
internal checkout.

## Runnable public surface

- `evaluation/smoke_test.py` loads the released Hugging Face model and checks
  relative semantic ordering.
- `examples/concept_memory.py` demonstrates conservative grounding with a local
  ONNX artifact.
- `training/mining/` contains source-specific extraction tools. Inputs are
  always supplied explicitly; no script assumes a developer's home directory.

## Research surface

- `training/experiments/` preserves trainers used to test objectives and model
  families. These scripts may require licensed or non-published pairs.
- `evaluation/research/` preserves evaluation methodology whose original
  datasets cannot be redistributed.
- `research/legacy/coder_tw/` is the superseded CODER-TW implementation.
- `docs/research/EXPERIMENTS.md` is a chronological evidence log. Historical
  internal terminology in that log describes the experiment at the time, not
  the current repository interface.

Research code is retained because negative results and evaluation methods are
part of the evidence. It is not advertised as runnable without its declared
inputs.

## Non-published material

The repository does not contain:

- patient-identifying data;
- institution-specific vocabularies or evaluation labels;
- copyrighted article, teaching-material, or hospital abbreviation pairs;
- licensed terminology distributions such as SNOMED CT;
- an exact executable copy of the private distillation run.

The absence of those inputs is a rights and privacy boundary. It is also why
this project describes itself as an evidence and methodology release instead
of claiming one-command reproduction of the published checkpoint.
