#!/usr/bin/env python3
"""Quantize the fine-tuned CODER-TW to int8 ONNX and measure accuracy retention.

Order (per plan): fine-tune done -> quantize here.
Exports the CLS *pooler* output (CODER convention, outputs[1]) via a thin wrapper
so the ONNX graph matches training exactly; then onnxruntime dynamic int8.
Re-runs the SAME held-out split as train.py to report top-1/top-5 before/after
quantization -- honest accuracy retention, not just file size."""
import os
import random

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoTokenizer

from train import SEED, load_pairs  # identical data + split

D = os.path.dirname(__file__)
FT = os.path.join(D, "coder-tw")
FP32 = os.path.join(D, "coder_tw_fp32.onnx")
INT8 = os.path.join(D, "coder_tw_int8.onnx")
MAXLEN = 32


class PoolerWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids)[1]   # CLS pooler


def export_onnx(tok):
    model = AutoModel.from_pretrained(FT).eval()
    wrap = PoolerWrap(model).eval()
    enc = tok(["範例"], max_length=MAXLEN, truncation=True,
              padding="max_length", return_tensors="pt")
    torch.onnx.export(
        wrap, (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
        FP32, input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["pooler"],
        dynamic_axes={k: {0: "b"} for k in
                      ["input_ids", "attention_mask", "token_type_ids", "pooler"]},
        opset_version=14, dynamo=False,
    )


def onnx_encode(sess, tok, texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], max_length=MAXLEN, truncation=True,
                padding="max_length", return_tensors="np")
        v = sess.run(["pooler"], {
            "input_ids": e["input_ids"].astype(np.int64),
            "attention_mask": e["attention_mask"].astype(np.int64),
            "token_type_ids": e["token_type_ids"].astype(np.int64)})[0]
        v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
        out.append(v)
    return np.vstack(out)


def evaluate(sess, tok, test, pool):
    se = onnx_encode(sess, tok, [s for s, _ in test])
    ce = onnx_encode(sess, tok, pool)
    sims = se @ ce.T
    idx = {c: i for i, c in enumerate(pool)}
    t1 = t5 = 0
    for i, (_, g) in enumerate(test):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == idx[g])[0][0])
        t1 += rank == 0; t5 += rank < 5
    return t1/len(test), t5/len(test)


def main():
    import onnxruntime as ort
    tok = AutoTokenizer.from_pretrained(FT)

    # rebuild identical split
    random.seed(SEED)
    pairs = load_pairs(); random.shuffle(pairs)
    n_test = max(30, int(0.15*len(pairs)))
    test = pairs[:n_test]; pool = sorted({c for _, c in pairs})

    print("exporting fp32 ONNX (pooler output)...")
    export_onnx(tok)
    print("dynamic int8 quantization...")
    quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)

    def mb(p): return os.path.getsize(p)/1e6
    sess32 = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(INT8, providers=["CPUExecutionProvider"])
    e32 = evaluate(sess32, tok, test, pool)
    e8 = evaluate(sess8, tok, test, pool)

    st = os.path.getsize(os.path.join(FT, "model.safetensors"))/1e6
    print("\n=== size ===")
    print(f"  fine-tuned fp32 (safetensors): {st:6.1f} MB")
    print(f"  ONNX fp32:                     {mb(FP32):6.1f} MB")
    print(f"  ONNX int8:                     {mb(INT8):6.1f} MB  "
          f"({mb(INT8)/mb(FP32)*100:.0f}% of fp32)")
    print("=== held-out accuracy (same 148-pair test set) ===")
    print(f"  ONNX fp32:  top1={e32[0]:.3f} top5={e32[1]:.3f}")
    print(f"  ONNX int8:  top1={e8[0]:.3f} top5={e8[1]:.3f}")


if __name__ == "__main__":
    main()
