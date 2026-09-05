#!/usr/bin/env python3
"""Export and Quantize IlhaEmbed to INT8 ONNX.

Features:
1. Exports PyTorch ModernBERT model to ONNX with input_ids and attention_mask.
2. Applies dynamic INT8 quantization via onnxruntime.quantization.
3. Validates against tokenizers + onnxruntime CPUExecutionProvider.
4. Verifies clinical embeddings and cosine similarity against PyTorch reference.

Target: File size <= 40MB, drop-in replacement for Hygieia / intake-embedder.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer


class ModernBertWrapper(torch.nn.Module):
    """Wraps ModernBERT to output last_hidden_state cleanly."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def export_and_quantize(model_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = out_dir / "model.onnx"
    int8_onnx = out_dir / "model_int8.onnx"

    print(f"Loading PyTorch model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pt_model = AutoModel.from_pretrained(model_dir).eval()
    wrapped_model = ModernBertWrapper(pt_model).eval()

    dummy_text = ["低劑量胸部電腦斷層", "健檢報告 檢體未交 MIF"]
    enc = tokenizer(
        dummy_text,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    print("Exporting FP32 ONNX model...")
    t0 = time.time()
    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask),
        str(fp32_onnx),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    fp32_size_mb = fp32_onnx.stat().st_size / (1024 * 1024)
    print(f"FP32 ONNX exported: {fp32_size_mb:.2f} MB (took {time.time()-t0:.2f}s)")

    print("Quantizing to INT8 (dynamic quantization)...")
    t0 = time.time()
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        weight_type=QuantType.QInt8,
    )
    int8_size_mb = int8_onnx.stat().st_size / (1024 * 1024)
    print(f"INT8 ONNX created: {int8_size_mb:.2f} MB (took {time.time()-t0:.2f}s)")

    tokenizer.save_pretrained(out_dir)

    print("\n--- Running Smoke Test & Numerical Verification ---")
    session = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])

    test_pairs = [
        ("皮蛇", "帶狀皰疹"),
        ("L-CT", "低劑量胸部電腦斷層"),
        ("CBC", "全血球計數"),
        ("檳榔", "嚼檳榔"),
        ("心肌梗塞", "急性冠狀動脈症候群"),
        ("心肌梗塞", "香港腳"),
    ]

    def embed_pt(text: str) -> np.ndarray:
        e = tokenizer([text], max_length=32, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = pt_model(**e).last_hidden_state
            mask = e["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            normed = F.normalize(pooled, p=2, dim=1)
            return normed.cpu().numpy()[0]

    def embed_onnx(text: str) -> np.ndarray:
        e = tokenizer([text], max_length=32, padding="max_length", truncation=True, return_tensors="np")
        feed = {
            "input_ids": e["input_ids"].astype(np.int64),
            "attention_mask": e["attention_mask"].astype(np.int64),
        }
        out = session.run(["last_hidden_state"], feed)[0]
        mask = e["attention_mask"][:, :, None].astype(np.float32)
        pooled = (out * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        normed = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return normed[0]

    cos_sims_pt = []
    cos_sims_onnx = []
    alignments = []

    for t1, t2 in test_pairs:
        v1_pt = embed_pt(t1)
        v2_pt = embed_pt(t2)
        sim_pt = float(np.dot(v1_pt, v2_pt))

        v1_onnx = embed_onnx(t1)
        v2_onnx = embed_onnx(t2)
        sim_onnx = float(np.dot(v1_onnx, v2_onnx))

        cos_sims_pt.append(sim_pt)
        cos_sims_onnx.append(sim_onnx)

        fid1 = float(np.dot(v1_pt, v1_onnx))
        alignments.append(fid1)

        print(f"'{t1}' vs '{t2}': PyTorch={sim_pt:+.4f} | INT8-ONNX={sim_onnx:+.4f} | Diff={abs(sim_pt-sim_onnx):.4f}")

    mean_alignment = np.mean(alignments)
    print(f"\nMean Cosine Fidelity (PyTorch vs INT8 ONNX): {mean_alignment:.4f}")
    assert mean_alignment > 0.98, f"Quantization fidelity too low: {mean_alignment}"
    assert int8_size_mb <= 40.0, f"INT8 model exceeds 40MB limit: {int8_size_mb:.2f} MB"
    print(f"\nAll assertions PASSED! Final INT8 model size: {int8_size_mb:.2f} MB (<= 40MB limit).")


def main():
    parser = argparse.ArgumentParser(description="Export and Quantize IlhaEmbed to INT8 ONNX")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to fine-tuned PyTorch checkpoint")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for ONNX models")
    args = parser.parse_args()

    export_and_quantize(args.model_dir, args.out_dir)


if __name__ == "__main__":
    main()
