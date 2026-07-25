"""Embedding backends, loaded so their outputs are directly comparable: every backend produces a
unit-normalized row vector, so cosine similarity means the same thing across all of them.

bge = BAAI/bge-small-zh-v1.5, a common general-purpose Chinese baseline, loaded via fastembed
(CLS pooling + normalization are baked into the fastembed ONNX export).

IlhaEmbed = the model this repo builds (weemed/IlhaEmbed on Hugging Face), loaded via
sentence-transformers -- the released Apache-2.0 weights, not a locally-built artifact.
"""

from __future__ import annotations

import numpy as np


class BgeEmbedder:
    """Wraps fastembed's BAAI/bge-small-zh-v1.5 -- a common general-purpose baseline."""

    name = "bge-small-zh-v1.5"
    dim = 512

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding("BAAI/bge-small-zh-v1.5")

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts, batch_size=batch_size)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class IlhaEmbedEmbedder:
    """Wraps the released `weemed/IlhaEmbed` model (sentence-transformers, mean-pooled, L2-normalized,
    384-d) -- pulled from Hugging Face by default, so this backend needs no local model files."""

    name = "IlhaEmbed"
    dim = 384

    def __init__(self, model_id: str = "weemed/IlhaEmbed") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True), dtype=np.float32
        )


class LegacyOnnxEmbedder:
    """Generic wrapper for a local ONNX BERT-style model whose export bakes pooling INTO the graph
    (a single "pooler" output, fixed-length input, no exposed last_hidden_state) -- so this wraps the
    graph's own pooled output rather than re-pooling. Not used by default: this project's earlier,
    superseded ONNX candidate (a PRC-origin base model) is not part of this public release and is
    never loaded unless a caller explicitly supplies its (locally-provided, not-shipped) path."""

    name = "legacy-onnx"
    dim = 768
    _seq_len = 32  # the graph's fixed sequence dimension (inspect via the ONNX input shapes)

    def __init__(self, model_path: str, tokenizer_dir: str) -> None:
        import onnxruntime as ort
        from transformers import BertTokenizerFast

        self._tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self._tokenizer(
                batch, padding="max_length", truncation=True, max_length=self._seq_len, return_tensors="np"
            )
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self._input_names}
            outputs = self._session.run(None, feed)
            pooled = outputs[0]  # (batch, 768) -- the graph's own pooler output
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            all_vectors.append((pooled / norms).astype(np.float32))
        return np.concatenate(all_vectors, axis=0)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, d), b: (m, d), both unit-normalized -> (n, m) cosine similarity matrix."""
    return a @ b.T
