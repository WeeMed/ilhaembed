"""A pooling/normalization sanity check BEFORE any real measurement run. If pooling or normalization
is wrong, the RELATIVE ordering below breaks (a colloquial term should sit much closer to its own
standard concept than to an unrelated one) even before any absolute number means anything.

Loads the released `weemed/IlhaEmbed` from Hugging Face by default -- override with ILHAEMBED_MODEL
to point at a local checkpoint instead. This checks relative ordering only; it does not assert
specific cosine values, since those depend on the exact checkpoint and are not a fixed contract.
"""

from __future__ import annotations

import os
import sys

from embedders import IlhaEmbedEmbedder

MODEL_ID = os.environ.get("ILHAEMBED_MODEL", "weemed/IlhaEmbed")

# (surface, its standard concept, an unrelated concept) -- the surface should sit closer to its own
# concept than to the unrelated one.
TRIPLETS = [
    ("斷腦筋", "中風", "糖尿病"),
    ("乳超", "乳房超音波", "骨折"),
]


def main() -> None:
    model = IlhaEmbedEmbedder(MODEL_ID)
    texts = sorted({t for triplet in TRIPLETS for t in triplet})
    vecs = model.embed(texts)
    index = {t: v for t, v in zip(texts, vecs, strict=True)}

    ok = True
    for surface, related, unrelated in TRIPLETS:
        sim_related = float(index[surface] @ index[related])
        sim_unrelated = float(index[surface] @ index[unrelated])
        status = "OK" if sim_related > sim_unrelated else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(
            f"{surface} ~ {related}={sim_related:.4f}  {surface} ~ {unrelated}={sim_unrelated:.4f}  "
            f"[{status}]"
        )

    if not ok:
        print("SMOKE TEST FAILED -- pooling/normalization is likely wrong. Fix before proceeding.")
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
