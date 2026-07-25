"""Label->concept rescue and care-context head-to-head, bge vs IlhaEmbed.

Neither slot has a curated ground-truth label file (concept rescue and care-context are
column-level judgments, not per-fragment labels), so this evaluates them against a real corpus
header population plus a small set of constructed positive/negative examples for care-context,
since a header population alone does not contain enough bare department-vs-name pairs to measure
separation.

Note: `corpus_fragments.tsv`, `concept_and_aggregate_seeds.json`, and `taxonomy_seeds.json` are not
part of this public release (see SOURCES.md -- they are derived from a private evaluation corpus).
Published for methodology transparency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from embedders import BgeEmbedder, IlhaEmbedEmbedder, LegacyOnnxEmbedder, cosine_matrix

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ILHAEMBED_EVAL_DATA", HERE.parent / "data"))
OUT = Path(os.environ.get("ILHAEMBED_EVAL_OUT", HERE.parent / "results"))
OUT.mkdir(parents=True, exist_ok=True)

# Optional, opt-in, off by default -- see run_eval.py's note on the excluded legacy ONNX candidate.
LEGACY_MODEL_PATH = os.environ.get("LEGACY_ONNX_MODEL_PATH")
LEGACY_TOKENIZER_DIR = os.environ.get("LEGACY_ONNX_TOKENIZER_DIR")

_CONCEPT_MIN_SCORE = 0.85
_CONCEPT_MIN_MARGIN = 0.05

# care_context_share's own two-way config (semantic_embedding.py): a lower floor because it is a
# two-cluster separation (care-context vs a person name), not an N-way classification.
_CARE_CONTEXT_MIN_SCORE = 0.60
_CARE_CONTEXT_CATEGORIES = frozenset({"care_source", "care_followup", "care_event"})

# Constructed examples (not from any real file -- there is no labelled care-context set in the repo).
# Departments/care-context phrases are drawn from the taxonomy's own seeds; "names" are common
# placeholder Chinese given names that do not correspond to any real person, used only to test
# whether the model separates a department string from a name-shaped string.
_CARE_CONTEXT_POSITIVES = [
    "復健科", "心臟科", "新陳代謝科", "婦科", "骨科", "定期心內門診", "門診", "住院", "出院", "急診",
]
_CARE_CONTEXT_NEGATIVES_NAMES = [
    "王小明", "陳美玲", "林志豪", "張淑芬", "李國強", "黃雅婷", "吳建宏", "劉美惠",
]


def build_concept_centroids(model, exemplars: dict) -> tuple[list[str], np.ndarray]:
    targets, vectors = [], []
    for target, seeds in exemplars.items():
        seed_vecs = model.embed(list(seeds))
        centroid = seed_vecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        targets.append(target)
        vectors.append(centroid)
    return targets, np.array(vectors, dtype=np.float32)


def run_concept_rescue(models: dict, exemplars: dict) -> dict:
    headers = sorted(
        {
            line.split("\t", 1)[1]
            for line in (DATA / "corpus_fragments.tsv").read_text(encoding="utf-8").splitlines()
            if line.startswith("header\t")
        }
    )
    summary = {}
    rows_by_model = {}
    for model_key, model in models.items():
        targets, centroids = build_concept_centroids(model, exemplars)
        header_vecs = model.embed(headers)
        scores = cosine_matrix(header_vecs, centroids)
        rows = []
        matched = 0
        for i, header in enumerate(headers):
            order = np.argsort(scores[i])[::-1]
            top = float(scores[i][order[0]])
            second = float(scores[i][order[1]]) if len(order) > 1 else 0.0
            winner = targets[order[0]] if (top >= _CONCEPT_MIN_SCORE and top - second >= _CONCEPT_MIN_MARGIN) else None
            if winner:
                matched += 1
            rows.append({"header": header, "top_target": targets[order[0]], "top_score": round(top, 4), "matched": winner})
        rows_by_model[model_key] = rows
        summary[model_key] = {"n_headers": len(headers), "n_matched_at_production_gate": matched}
    return {"summary": summary, "rows": rows_by_model}


def run_care_context(models: dict) -> dict:
    taxo = json.loads((DATA / "taxonomy_seeds.json").read_text(encoding="utf-8"))
    taxonomy = taxo["taxonomy"]
    summary = {}
    rows_by_model = {}
    for model_key, model in models.items():
        cats = [c for c in _CARE_CONTEXT_CATEGORIES]
        seed_texts, seed_owner = [], []
        for cat in cats:
            for s in taxonomy[cat]:
                seed_texts.append(s)
                seed_owner.append(cat)
        seed_vecs = model.embed(seed_texts)
        centroid = seed_vecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)

        pos_vecs = model.embed(_CARE_CONTEXT_POSITIVES)
        neg_vecs = model.embed(_CARE_CONTEXT_NEGATIVES_NAMES)
        pos_scores = pos_vecs @ centroid
        neg_scores = neg_vecs @ centroid

        rows = [
            {"text": t, "kind": "care_context", "score": round(float(s), 4), "reads_as_care_context": bool(s >= _CARE_CONTEXT_MIN_SCORE)}
            for t, s in zip(_CARE_CONTEXT_POSITIVES, pos_scores, strict=True)
        ] + [
            {"text": t, "kind": "name", "score": round(float(s), 4), "reads_as_care_context": bool(s >= _CARE_CONTEXT_MIN_SCORE)}
            for t, s in zip(_CARE_CONTEXT_NEGATIVES_NAMES, neg_scores, strict=True)
        ]
        rows_by_model[model_key] = rows
        summary[model_key] = {
            "positive_mean": float(pos_scores.mean()),
            "positive_min": float(pos_scores.min()),
            "negative_mean": float(neg_scores.mean()),
            "negative_max": float(neg_scores.max()),
            "gap_posmin_minus_negmax": float(pos_scores.min() - neg_scores.max()),
            "positives_correctly_flagged": int((pos_scores >= _CARE_CONTEXT_MIN_SCORE).sum()),
            "negatives_wrongly_flagged": int((neg_scores >= _CARE_CONTEXT_MIN_SCORE).sum()),
        }
    return {"summary": summary, "rows": rows_by_model}


def main() -> None:
    bge = BgeEmbedder()
    ilhaembed = IlhaEmbedEmbedder()
    models = {"bge": bge, "ilhaembed": ilhaembed}
    if LEGACY_MODEL_PATH and LEGACY_TOKENIZER_DIR:
        models["legacy_onnx"] = LegacyOnnxEmbedder(LEGACY_MODEL_PATH, LEGACY_TOKENIZER_DIR)

    exemplars = json.loads((DATA / "concept_and_aggregate_seeds.json").read_text(encoding="utf-8"))["concept_exemplars"]

    concept = run_concept_rescue(models, exemplars)
    care = run_care_context(models)

    (OUT / "concept_rescue_summary.json").write_text(json.dumps(concept["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "care_context_summary.json").write_text(json.dumps(care["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    for model_key in models:
        with (OUT / f"concept_rescue_{model_key}.tsv").open("w", encoding="utf-8") as f:
            f.write("header\ttop_target\ttop_score\tmatched\n")
            for r in concept["rows"][model_key]:
                f.write(f"{r['header']}\t{r['top_target']}\t{r['top_score']}\t{r['matched']}\n")
        with (OUT / f"care_context_{model_key}.tsv").open("w", encoding="utf-8") as f:
            f.write("text\tkind\tscore\treads_as_care_context\n")
            for r in care["rows"][model_key]:
                f.write(f"{r['text']}\t{r['kind']}\t{r['score']}\t{r['reads_as_care_context']}\n")

    print(json.dumps(concept["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(care["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
