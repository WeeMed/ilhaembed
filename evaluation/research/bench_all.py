#!/usr/bin/env python3
"""Benchmark every candidate on the judgment the product actually needs: reading the trade's jargon.

Everything measured before this was taxonomy top1 -- a proxy. The question that decides the product is
whether a model reads what staff actually write: a phonetic loan (a blood culture written as the
Taiwanese sounding-out of the English), a Taigi metaphor for a stroke, a clipped department name, an
English dosing abbreviation. None of these have a structural anchor, which is exactly why they are an
embedder's job.

Deliberately EXCLUDED: institution abbreviations. The repository already measured and rejected an
embedder there (`tw-facility-abbreviation-matching.md`) because those abbreviations DO have structural
anchors -- substring, segment-initial subsequence, registrant prefix -- and deterministic scorers beat
cosine on them. Benchmarking a model on work it should not be doing would argue for the wrong design.

CONTAMINATION IS REPORTED, NOT HIDDEN. `specialized_pairs` is CODER-TW's training data and also ours
(it is `colloquial.tsv`). Those models are answering an exam they studied. Their scores are marked
CONTAMINATED and are upper bounds, not estimates. Only the untrained models are measured cleanly, and
a trained model failing to beat an untrained one on its own training data is the finding that matters.

Each model runs in its own process. Resident memory is read from a high-water mark that never falls,
so measuring two models in one process attributes the first one's peak to the second -- an earlier run
reported 0 MB for a 102M-parameter model that way.
"""

from __future__ import annotations

import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
DATA = RESEARCH / "data"


def load_jargon() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (specialized, slang) surface->canonical pairs.

    Split by source because their contamination differs and their difficulty differs: the slang table
    is spoken hospital shorthand, the specialized table is written abbreviations and colloquial terms.
    """
    specialized: list[tuple[str, str]] = []
    path = DATA / "specialized_pairs.tsv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                a, b = (row.get("a") or "").strip(), (row.get("b") or "").strip()
                if a and b and a != b:
                    specialized.append((a, b))

    slang: list[tuple[str, str]] = []
    path = DATA / "med_slang.tsv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                surface = (row.get("slang") or "").strip()
                meaning = (row.get("meaning") or "").strip()
                # A slang entry may list variants separated by a slash; the first is the headword.
                surface = surface.split("/")[0].strip()
                if surface and meaning and surface != meaning:
                    slang.append((surface, meaning))
    return specialized, slang


def retrieval_accuracy(embed, pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Rank every canonical against each surface form; report exact-match top1/top5.

    The pool is every distinct canonical in the set, so the task gets harder as the set grows -- which
    is the honest form of the runtime question, where the right reading competes with everything else
    the model knows."""
    import numpy as np

    if not pairs:
        return {"n": 0, "top1": 0.0, "top5": 0.0}
    pool = sorted({b for _, b in pairs})
    index = {canonical: position for position, canonical in enumerate(pool)}
    surfaces = embed([a for a, _ in pairs])
    canonicals = embed(pool)
    scores = surfaces @ canonicals.T
    top1 = top5 = 0
    for row, (_, gold) in zip(scores, pairs):
        order = np.argsort(-row)
        target = index[gold]
        top1 += int(order[0] == target)
        top5 += int(target in order[:5])
    return {"n": len(pairs), "pool": len(pool), "top1": top1 / len(pairs), "top5": top5 / len(pairs)}


def main() -> int:
    model_path = sys.argv[1]
    label = sys.argv[2]
    kind = sys.argv[3] if len(sys.argv) > 3 else "hf"

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import numpy as np
    import torch

    torch.set_num_threads(1)
    import torch.nn.functional as F

    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # noqa: E731
    baseline = rss()

    if kind == "onnx":
        import onnxruntime as ort
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            os.path.join(model_path, "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        params = os.path.getsize(os.path.join(model_path, "model.onnx")) / 1e6

        def embed(texts: list[str]) -> "np.ndarray":
            out = []
            for start in range(0, len(texts), 32):
                batch = tok(
                    texts[start : start + 32], max_length=32, truncation=True,
                    padding="max_length", return_tensors="np",
                )
                feed = {i.name: batch[i.name].astype("int64") for i in session.get_inputs() if i.name in batch}
                vectors = session.run(None, feed)[0]
                if vectors.ndim == 3:
                    vectors = vectors[:, 0]
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                out.append(vectors / norms)
            return np.vstack(out)
    else:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6

        def embed(texts: list[str]) -> "np.ndarray":
            out = []
            for start in range(0, len(texts), 32):
                ids = tok(
                    texts[start : start + 32], max_length=32, truncation=True,
                    padding="max_length", return_tensors="pt",
                )
                with torch.no_grad():
                    hidden = model(**ids).last_hidden_state[:, 0]
                out.append(F.normalize(hidden, dim=1).numpy())
            return np.vstack(out)

    loaded = rss()
    specialized, slang = load_jargon()

    probe = [a for a, _ in specialized[:96]] or ["糖化血色素"] * 96
    embed(probe[:16])  # warm up
    start = time.perf_counter()
    embed(probe)
    latency = (time.perf_counter() - start) * 1000 / len(probe)

    result = {
        "model": label,
        "params_or_mb": round(params, 1),
        "resident_mb": round(loaded - baseline),
        "peak_mb": round(rss()),
        "latency_ms": round(latency, 2),
        "specialized": retrieval_accuracy(embed, specialized),
        "slang": retrieval_accuracy(embed, slang),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
