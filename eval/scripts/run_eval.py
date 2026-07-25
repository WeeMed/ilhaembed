"""Head-to-head bge-small-zh-v1.5 vs IlhaEmbed for a candidate intake semantic-routing layer.

Replicates a production-style scoring gate (centroid = mean of seed embeddings, renormalized;
fragment score = centroids @ unit_vector(fragment)) IDENTICALLY for both models, so the comparison
isolates the embedding model, not the gate logic. Writes raw per-fragment scores as TSVs (every
number auditable) and a summary JSON.

CPU-only inference, single thread -- this measures the on-prem deploy profile, not a GPU-accelerated
one.

Note: this script's DATA files (`taxonomy_seeds.json`, `semantic_labels.jsonl`, `gold_labels.jsonl`,
`code_labels.jsonl`, `icd_concept_pool.json`) are not part of this public release (see SOURCES.md --
they are derived from a private evaluation corpus). Published for methodology transparency.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from embedders import BgeEmbedder, IlhaEmbedEmbedder, LegacyOnnxEmbedder, cosine_matrix

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ILHAEMBED_EVAL_DATA", HERE.parent / "data"))
OUT = Path(os.environ.get("ILHAEMBED_EVAL_OUT", HERE.parent / "results"))
OUT.mkdir(parents=True, exist_ok=True)

# Optional, opt-in, off by default: this project's earlier, superseded ONNX candidate (a PRC-origin
# base model) is not part of this public release. Set both env vars to a locally-provided checkpoint
# to include it in the comparison; otherwise only bge and IlhaEmbed are compared.
LEGACY_MODEL_PATH = os.environ.get("LEGACY_ONNX_MODEL_PATH")
LEGACY_TOKENIZER_DIR = os.environ.get("LEGACY_ONNX_TOKENIZER_DIR")

# Production's fragment-classification gate (semantic_embedding.py): a category is accepted only when
# it clearly wins BOTH a score floor and a margin over the runner-up. Reproduced here so "would this
# fragment be classified" is asked with the identical two-part rule for both models.
_FRAGMENT_MIN_SCORE = 0.70
_FRAGMENT_MIN_MARGIN = 0.10


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_taxonomy_centroids(model, taxonomy: dict, auto_classifiable: dict) -> tuple[list[str], np.ndarray]:
    """Mirrors `_taxonomy_prototypes()`: mean of seed vectors per AUTO-CLASSIFIABLE category,
    renormalized to unit length. Non-auto-classifiable categories (health_goal, care_referral) are
    excluded from the auto-apply prototype set, exactly as production does."""
    categories, vectors = [], []
    for category, seeds in taxonomy.items():
        if not auto_classifiable.get(category, True):
            continue
        seed_vecs = model.embed(list(seeds))
        centroid = seed_vecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        categories.append(category)
        vectors.append(centroid)
    return categories, np.array(vectors, dtype=np.float32)


def classify(scores_row: np.ndarray, categories: list[str]) -> tuple[str | None, float, float]:
    """Production's classify_fragment() gate applied to one fragment's score row.
    Returns (winning category or None, top score, margin)."""
    order = np.argsort(scores_row)[::-1]
    top = float(scores_row[order[0]])
    second = float(scores_row[order[1]]) if len(order) > 1 else 0.0
    margin = top - second
    if top >= _FRAGMENT_MIN_SCORE and margin >= _FRAGMENT_MIN_MARGIN:
        return categories[order[0]], top, margin
    return None, top, margin


def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney U statistic, the standard nonparametric AUC: P(a random positive scores higher
    than a random negative). Rank-based, so it is comparable across models on different absolute
    cosine scales -- exactly the property this eval needs (different embedding models' cosines are NOT
    directly comparable, but their ranking quality is)."""
    all_scores = np.concatenate([pos_scores, neg_scores])
    ranks = np.argsort(np.argsort(all_scores)) + 1  # average-free rank (ties are rare with floats)
    pos_ranks = ranks[: len(pos_scores)]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = pos_ranks.sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def fit_threshold_zero_fp(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[float, float]:
    """The lowest score threshold that admits ZERO negatives, and the fraction of positives it still
    accepts at that threshold. This is the model's fitted operating point on its own scale -- never
    compared in absolute terms across models, only via what it costs in recall."""
    if len(neg_scores) == 0:
        threshold = float(pos_scores.min()) if len(pos_scores) else 1.0
    else:
        threshold = float(neg_scores.max()) + 1e-6
    recall = float((pos_scores >= threshold).mean()) if len(pos_scores) else float("nan")
    return threshold, recall


def _is_readable(text: str) -> bool:
    """Production's `_readable_by_model` gate (semantic_embedding.py): a fragment with no Han
    character collapses onto one vector for a Chinese model, so it must never be scored -- neither
    accepted nor counted as a refused negative, because the model said nothing about it either way."""
    return any("一" <= ch <= "鿿" for ch in text)


def run_value_to_category(models: dict, taxonomy: dict, auto_classifiable: dict) -> dict:
    """Role assignment corrected 2026-07-19 (the first run wrongly took ALL 218 semantic_labels rows,
    including 162 rows labelled "residue", as positives -- residue is not a taxonomy category, can
    never win top1 against the prototype set, and its correct behaviour is REFUSAL, so it belongs on
    the negative side, not the positive one. See EXPERIMENTS.md for the retraction).

    Positives: semantic_labels rows whose category is a real, auto-classifiable taxonomy category,
    plus gold_labels rows that carry a real category (both sides drawn from labels a human already
    confirmed, never invented here). Negatives: semantic_labels residue rows + gold_labels
    likely_overrouted/null rows -- everything a correct classifier must refuse.
    """
    auto_categories = {c for c in taxonomy if auto_classifiable.get(c, True)}
    semantic_labels = load_jsonl(DATA / "semantic_labels.jsonl")
    gold_labels = load_jsonl(DATA / "gold_labels.jsonl")

    sem_positive = [r for r in semantic_labels if r["category"] in auto_categories]
    sem_negative_residue = [r for r in semantic_labels if r["category"] == "residue"]
    sem_dropped = [
        r for r in semantic_labels if r["category"] != "residue" and r["category"] not in auto_categories
    ]  # a labelled category that exists but isn't in the auto-apply prototype set (e.g. non-auto_classifiable)

    gold_positive = [r for r in gold_labels if r.get("category") in auto_categories]
    gold_negative = [r for r in gold_labels if r.get("likely_overrouted") and r.get("category") is None]
    gold_dropped = [
        r
        for r in gold_labels
        if r.get("category") is not None and r.get("category") not in auto_categories
    ]  # e.g. "address" -- a real label, but not a taxonomy category this prototype set can win

    positives_all = [{"fragment": r["fragment"], "category": r["category"]} for r in sem_positive] + [
        {"fragment": r["fragment"], "category": r["category"]} for r in gold_positive
    ]
    negatives_all = [{"fragment": r["fragment"]} for r in sem_negative_residue] + [
        {"fragment": r["fragment"]} for r in gold_negative
    ]

    positives_readable = [r for r in positives_all if _is_readable(r["fragment"])]
    negatives_readable = [r for r in negatives_all if _is_readable(r["fragment"])]
    n_positive_excluded_unreadable = len(positives_all) - len(positives_readable)
    n_negative_excluded_unreadable = len(negatives_all) - len(negatives_readable)

    summary: dict = {}
    per_model_rows: dict[str, list[dict]] = {}

    for model_key, model in models.items():
        categories, centroids = build_taxonomy_centroids(model, taxonomy, auto_classifiable)

        pos_fragments = [row["fragment"] for row in positives_readable]
        pos_expected = [row["category"] for row in positives_readable]
        neg_fragments = [row["fragment"] for row in negatives_readable]

        pos_vecs = model.embed(pos_fragments)
        neg_vecs = model.embed(neg_fragments) if neg_fragments else np.zeros((0, model.dim), dtype=np.float32)

        pos_score_matrix = cosine_matrix(pos_vecs, centroids)  # (n_pos, n_cat)
        neg_score_matrix = (
            cosine_matrix(neg_vecs, centroids) if len(neg_fragments) else np.zeros((0, len(categories)))
        )

        rows = []
        top1_hits = top5_hits = 0
        pos_top_scores = []
        for i, frag in enumerate(pos_fragments):
            row_scores = pos_score_matrix[i]
            order = np.argsort(row_scores)[::-1]
            ranked = [(categories[j], round(float(row_scores[j]), 4)) for j in order]
            expected = pos_expected[i]
            top1 = ranked[0][0] == expected
            top5 = expected in [c for c, _ in ranked[:5]]
            top1_hits += int(top1)
            top5_hits += int(top5)
            winner, top_score, margin = classify(row_scores, categories)
            pos_top_scores.append(top_score)
            rows.append(
                {
                    "fragment": frag,
                    "expected_category": expected,
                    "top1_category": ranked[0][0],
                    "top1_score": ranked[0][1],
                    "top5_categories": ranked[:5],
                    "gate_winner": winner,
                    "gate_correct": winner == expected,
                    "margin": round(margin, 4),
                }
            )

        neg_top_scores = []
        neg_false_positives = []
        for i, frag in enumerate(neg_fragments):
            row_scores = neg_score_matrix[i]
            winner, top_score, margin = classify(row_scores, categories)
            neg_top_scores.append(top_score)
            if winner is not None:
                neg_false_positives.append({"fragment": frag, "wrongly_classified_as": winner, "score": round(top_score, 4)})

        pos_top_scores_arr = np.array(pos_top_scores)
        neg_top_scores_arr = np.array(neg_top_scores)
        auc = auc_from_scores(pos_top_scores_arr, neg_top_scores_arr)
        threshold, recall_at_fitted_threshold = fit_threshold_zero_fp(pos_top_scores_arr, neg_top_scores_arr)

        pos_p5 = float(np.percentile(pos_top_scores_arr, 5)) if len(pos_top_scores_arr) else float("nan")
        neg_p95 = float(np.percentile(neg_top_scores_arr, 95)) if len(neg_top_scores_arr) else float("nan")

        summary[model_key] = {
            "n_positive": len(pos_fragments),
            "n_negative": len(neg_fragments),
            "n_positive_excluded_unreadable": n_positive_excluded_unreadable,
            "n_negative_excluded_unreadable": n_negative_excluded_unreadable,
            "n_positive_dropped_not_auto_category": len(sem_dropped) + len(gold_dropped),
            "top1_accuracy": top1_hits / len(pos_fragments),
            "top5_accuracy": top5_hits / len(pos_fragments),
            "recall_at_production_gate_correct_category": float(np.mean([r["gate_correct"] for r in rows])),
            "recall_at_production_gate_any_accept": float(np.mean([r["gate_winner"] is not None for r in rows])),
            "auc": auc,
            "pos_score_mean": float(pos_top_scores_arr.mean()),
            "pos_score_p5": pos_p5,
            "neg_score_mean": float(neg_top_scores_arr.mean()) if len(neg_top_scores_arr) else float("nan"),
            "neg_score_p95": neg_p95,
            "separation_gap_p5pos_minus_p95neg": pos_p5 - neg_p95 if len(neg_top_scores_arr) else float("nan"),
            "fitted_threshold_zero_fp": threshold,
            "recall_at_fitted_threshold": recall_at_fitted_threshold,
            "false_positives_at_production_threshold": neg_false_positives,
            "n_false_positives_at_production_threshold": len(neg_false_positives),
        }
        per_model_rows[model_key] = rows

    # 0.6-0.7 band analysis: fragments where bge's top1 score lands in [0.60, 0.70) -- a plausible
    # production threshold band -- and where the candidate model puts the SAME fragment.
    band_fragments = []
    candidate_key = "ilhaembed" if "ilhaembed" in per_model_rows else next(
        (k for k in per_model_rows if k != "bge"), None
    )
    if "bge" in per_model_rows:
        for i, row in enumerate(per_model_rows["bge"]):
            if 0.60 <= row["top1_score"] < 0.70:
                candidate_row = (
                    per_model_rows[candidate_key][i]
                    if candidate_key and i < len(per_model_rows[candidate_key])
                    else None
                )
                band_fragments.append(
                    {
                        "fragment": row["fragment"],
                        "expected_category": row["expected_category"],
                        "bge_top1_score": row["top1_score"],
                        "bge_top1_category": row["top1_category"],
                        "candidate_top1_score": candidate_row["top1_score"] if candidate_row else None,
                        "candidate_top1_category": candidate_row["top1_category"] if candidate_row else None,
                        "candidate_gate_correct": candidate_row["gate_correct"] if candidate_row else None,
                    }
                )

    return {"summary": summary, "rows": per_model_rows, "band_0.6_0.7": band_fragments}


def run_icd(models: dict) -> dict:
    """Only the ICD-CONDITION cases (an `expect_code` against the short-code concept pool this
    reproduces) are in scope: `code_labels.jsonl` also carries exam-name cases (`expect_display`,
    matched through a different subsystem -- the Alphabetic Index / entailment tiers in
    `icd_condition_index.py`, not this embedding pool) and refusal cases (`expect: none`), neither of
    which this cosine-ranking reproduction can honestly evaluate. Excluded cases are reported, not
    silently dropped."""
    concept_pool = json.loads((DATA / "icd_concept_pool.json").read_text(encoding="utf-8"))
    all_labels = load_jsonl(DATA / "code_labels.jsonl")
    code_labels = [row for row in all_labels if "expect_code" in row]
    excluded = [
        {"fragment": row["fragment"], "reason": "no expect_code (exam-display or refusal case, different subsystem)"}
        for row in all_labels
        if "expect_code" not in row
    ]
    pool_codes = [c["code"] for c in concept_pool]
    pool_displays = [c["display_zh"] for c in concept_pool]

    summary = {}
    per_model_cases: dict[str, list[dict]] = {}
    for model_key, model in models.items():
        pool_vecs = model.embed(pool_displays, batch_size=128)
        frag_vecs = model.embed([c["fragment"] for c in code_labels])
        score_matrix = cosine_matrix(frag_vecs, pool_vecs)  # (n_frag, n_pool)

        cases = []
        top1_hits = top5_hits = 0
        for i, label in enumerate(code_labels):
            row_scores = score_matrix[i]
            order = np.argsort(row_scores)[::-1][:10]
            ranked = [
                {"code": pool_codes[j], "display_zh": pool_displays[j], "score": round(float(row_scores[j]), 4)}
                for j in order
            ]
            expect = label["expect_code"]
            top1 = ranked[0]["code"] == expect
            top5 = expect in [r["code"] for r in ranked[:5]]
            top1_hits += int(top1)
            top5_hits += int(top5)
            cases.append(
                {
                    "fragment": label["fragment"],
                    "expect_code": expect,
                    "why": label.get("why", ""),
                    "top1_correct": top1,
                    "top5_correct": top5,
                    "top10": ranked,
                }
            )
        per_model_cases[model_key] = cases
        summary[model_key] = {
            "n": len(code_labels),
            "top1_accuracy": top1_hits / len(code_labels),
            "top5_accuracy": top5_hits / len(code_labels),
        }
    return {"summary": summary, "cases": per_model_cases, "excluded": excluded}


def run_cost(models: dict) -> dict:
    """Per-fragment single-thread CPU embed latency + a rough resident-memory proxy (process RSS
    delta is unreliable across model loads in one process, so this reports model asset size on disk,
    which is the number the on-prem tier budget actually gates on)."""
    import os

    sample_texts = ["高血壓", "定期心內門診", "郵寄自取給公司", "血糖127 mg/dL", "戒菸衛教"] * 20
    cost = {}
    for model_key, model in models.items():
        # warmup
        model.embed(sample_texts[:5])
        start = time.perf_counter()
        for text in sample_texts:
            model.embed([text])
        elapsed = time.perf_counter() - start
        cost[model_key] = {
            "per_fragment_ms_single_thread": (elapsed / len(sample_texts)) * 1000,
            "dim": model.dim,
        }
    cost["bge"]["disk_size_mb"] = None  # filled by caller with fastembed cache lookup, if resolvable
    if "ilhaembed" in cost:
        cost["ilhaembed"]["disk_size_mb"] = 37  # released INT8 ONNX size; see MODEL-CARD.md
    if "legacy_onnx" in cost and LEGACY_MODEL_PATH:
        cost["legacy_onnx"]["disk_size_mb"] = round(os.path.getsize(LEGACY_MODEL_PATH) / (1024 * 1024), 1)
    return cost


def main() -> None:
    print("loading models...", flush=True)
    bge = BgeEmbedder()
    ilhaembed = IlhaEmbedEmbedder()
    models = {"bge": bge, "ilhaembed": ilhaembed}
    if LEGACY_MODEL_PATH and LEGACY_TOKENIZER_DIR:
        models["legacy_onnx"] = LegacyOnnxEmbedder(LEGACY_MODEL_PATH, LEGACY_TOKENIZER_DIR)

    taxo = json.loads((DATA / "taxonomy_seeds.json").read_text(encoding="utf-8"))
    taxonomy, auto_classifiable = taxo["taxonomy"], taxo["auto_classifiable"]

    print("running value->category eval...", flush=True)
    v2c = run_value_to_category(models, taxonomy, auto_classifiable)

    print("running ICD condition eval...", flush=True)
    icd = run_icd(models)

    print("running cost eval...", flush=True)
    cost = run_cost(models)

    (OUT / "value_to_category_summary.json").write_text(
        json.dumps(v2c["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "band_0.6_0.7.json").write_text(
        json.dumps(v2c["band_0.6_0.7"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "icd_summary.json").write_text(json.dumps(icd["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "icd_excluded.json").write_text(json.dumps(icd["excluded"], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "cost.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")

    for model_key in models:
        rows = v2c["rows"][model_key]
        with (OUT / f"value_to_category_{model_key}.tsv").open("w", encoding="utf-8") as f:
            f.write("fragment\texpected\ttop1_category\ttop1_score\tgate_winner\tgate_correct\tmargin\n")
            for r in rows:
                f.write(
                    f"{r['fragment']}\t{r['expected_category']}\t{r['top1_category']}\t{r['top1_score']}\t"
                    f"{r['gate_winner']}\t{r['gate_correct']}\t{r['margin']}\n"
                )
        cases = icd["cases"][model_key]
        with (OUT / f"icd_{model_key}.tsv").open("w", encoding="utf-8") as f:
            f.write("fragment\texpect_code\ttop1_correct\ttop5_correct\ttop1_code\ttop1_display\ttop1_score\n")
            for c in cases:
                top1 = c["top10"][0]
                f.write(
                    f"{c['fragment']}\t{c['expect_code']}\t{c['top1_correct']}\t{c['top5_correct']}\t"
                    f"{top1['code']}\t{top1['display_zh']}\t{top1['score']}\n"
                )

    (OUT / "icd_cases_full.json").write_text(json.dumps(icd["cases"], ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE. Summaries:")
    print(json.dumps(v2c["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(icd["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(cost, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
