#!/usr/bin/env python3
"""Reference implementation of the concept-memory gate described in MODEL-CARD.md.

WHY THIS EXISTS RATHER THAN A FINE-TUNING SCRIPT. A small int8 encoder can only separate concepts
whose decision margin exceeds its own quantization noise. On institutional operational shorthand
(the closed set of abbreviations one hospital writes on its forms) the margin is several times
SMALLER than the int8 noise floor, so the encoder decides inside its own error -- and teaching the
weights that vocabulary shrinks the margin further while leaving the noise unchanged. A closed
vocabulary does not need to be learned; it can be looked up. So the domain knowledge lives here, as
data, and the encoder stays frozen.

The two stages are both load-bearing, each measured:

1. GATE -- only a fragment within ``threshold`` of a stored key is grounded. Everything else is
   returned as the encoder saw it, which is why adding a memory cannot regress unrelated inputs.
2. SUBSTITUTE -- a grounded fragment is represented by its CONCEPT's embedding, not the matched
   key's. That is what converts a narrow-margin many-way decision into a high-margin one;
   substituting the matched surface instead recovers none of the benefit.

The threshold is a conservatism knob, not an accuracy trade: raising it grounds less and converges
on the ungrounded behaviour.

Run it (needs `onnxruntime`, `tokenizers`, `numpy`, and the published model + tokenizer):

    python3 concept_memory.py --model model_int8.onnx --tokenizer tokenizer.json

The register below is a small PUBLIC demonstration set, not the institutional register the model
card's numbers were measured on -- that one carries institutional provenance and is not
redistributed. Replace it with your own; the mechanism is what generalizes, the mapping is not.
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MAX_LEN = 32

# surface -> canonical concept. A public demonstration register.
DEMO_MEMORY: dict[str, str] = {
    "L-CT": "低劑量胸部電腦斷層",
    "H-CT": "頭部電腦斷層",
    "皮蛇": "帶狀皰疹",
    "檳榔": "嚼檳榔習慣",
    "成健": "成人預防保健健康檢查",
    "乳超": "乳房超音波",
    "骨密": "骨質密度檢查",
}


class Encoder:
    """Mean-pooled, L2-normalized 384-d vectors from the published int8 ONNX artifact."""

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=MAX_LEN)
        self._tokenizer.enable_padding(length=MAX_LEN)

    def encode_one(self, text: str) -> np.ndarray:
        """Embed ONE string, alone.

        Deliberately one at a time. This model's output depends on the batch a text was embedded in
        -- a batch pads to its longest member and the padding reaches the real tokens' outputs -- so
        the same string embedded inside a large batch and embedded alone are only ~0.88-0.97 similar.
        A similarity THRESHOLD compares a query against stored keys, so keys and queries must come
        from the same call shape or an exact match scores like a near-miss and the threshold stops
        meaning what it says.
        """
        encoding = self._tokenizer.encode(text)
        ids = np.array([encoding.ids], dtype=np.int64)
        mask = np.array([encoding.attention_mask], dtype=np.int64)
        hidden = np.asarray(self._session.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0])
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return pooled[0].astype(np.float32)


class ConceptMemory:
    """Keys (written surface forms) and their canonical concepts, embedded by the FROZEN encoder."""

    def __init__(self, encoder: Encoder, mapping: dict[str, str]) -> None:
        self._encoder = encoder
        self._concepts = list(mapping.values())
        # Keys go through the same single-string call a query uses -- see Encoder.encode_one.
        self._keys = np.stack([encoder.encode_one(surface) for surface in mapping])

    def resolve(self, fragment: str, threshold: float = 0.95) -> tuple[str, float] | None:
        """``(concept, score)`` when this fragment is a known written form, else None.

        Returns the concept STRING rather than a vector so a caller can both re-embed it and SHOW
        it: a decoded value must be able to prove itself by naming what it was read as.

        CALLER CONTRACT -- this answers "which known form is this", never "does this reading apply
        here". A short clinical token is not self-describing: a bare calcification token names a
        coronary calcium-score exam on a checkup order line and plain tissue calcification in a
        radiology impression. That difference is not in the token, so establish the context (the
        column, the document type) before consulting a scoped register. A lookup out of scope is a
        fabrication, not coverage.
        """
        if not fragment.strip():
            return None
        scores = self._keys @ self._encoder.encode_one(fragment)
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < threshold:
            return None
        return self._concepts[best], score

    def embed(self, fragment: str, threshold: float = 0.95) -> np.ndarray:
        """The grounded CONCEPT's embedding when the gate fires, else the fragment's own."""
        grounded = self.resolve(fragment, threshold=threshold)
        return self._encoder.encode_one(grounded[0] if grounded else fragment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to model_int8.onnx")
    parser.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    encoder = Encoder(args.model, args.tokenizer)
    memory = ConceptMemory(encoder, DEMO_MEMORY)

    # Known forms ground exactly; unrelated clinical text must pass through untouched -- that
    # second half is the point of the gate and is what makes the memory safe to add.
    probes = list(DEMO_MEMORY) + ["高血壓", "糖尿病", "上呼吸道感染", "心肌梗塞"]
    print(f"{'fragment':<10} {'grounded concept':<24} score")
    for probe in probes:
        grounded = memory.resolve(probe, threshold=args.threshold)
        if grounded:
            concept, score = grounded
            print(f"{probe:<10} {concept:<24} {score:.3f}")
        else:
            print(f"{probe:<10} {'(passes through)':<24} -")


if __name__ == "__main__":
    main()
