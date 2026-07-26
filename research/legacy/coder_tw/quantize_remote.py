#!/usr/bin/env python3
"""Remote (on the 4080 box) ONNX export + int8 quantization of coder-tw-gpu.
Self-contained. Runs on CPU (quantization targets CPU inference); only the small
int8 file is shipped back. Reports size + specialized held-out accuracy retention."""
import csv
import os
import random
import re

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import BertModel, BertTokenizerFast

D = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(D, "coder-tw-gpu")
FP32 = os.path.join(D, "coder_tw_gpu_fp32.onnx")
INT8 = os.path.join(D, "coder_tw_gpu_int8.onnx")
SEED = 42
MAXLEN = 32
CJK = "一-鿿"
_frag = re.compile(rf"^[之的有是及並]")


def clean_canonical(c):
    c = re.split(r"[（(]", c)[0]
    c = re.sub(r"\s+\d+\s*$", "", c)
    c = re.sub(r"\s+", "", c).strip()
    if not re.match(rf"[{CJK}A-Za-z]", c) or len(c) < 2 or _frag.match(c):
        return None
    return c


def load_specialized():
    # SAME data + columns as train_gpu.py so the eval split matches training
    rows = list(csv.DictReader(open(os.path.join(D, "specialized_pairs.tsv")), delimiter="\t"))
    return [(r["a"], r["b"]) for r in rows if r["a"] and r["b"]]


class Pooler(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids)[1]


def export(tok):
    m = BertModel.from_pretrained(FT).eval()
    w = Pooler(m).eval()
    e = tok(["範例"], max_length=MAXLEN, truncation=True, padding="max_length", return_tensors="pt")
    torch.onnx.export(w, (e["input_ids"], e["attention_mask"], e["token_type_ids"]), FP32,
                      input_names=["input_ids", "attention_mask", "token_type_ids"],
                      output_names=["pooler"],
                      dynamic_axes={k: {0: "b"} for k in
                                    ["input_ids", "attention_mask", "token_type_ids", "pooler"]},
                      opset_version=14)


def enc(sess, tok, texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], max_length=MAXLEN, truncation=True, padding="max_length", return_tensors="np")
        v = sess.run(["pooler"], {"input_ids": e["input_ids"].astype(np.int64),
                                  "attention_mask": e["attention_mask"].astype(np.int64),
                                  "token_type_ids": e["token_type_ids"].astype(np.int64)})[0]
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9))
    return np.vstack(out)


def acc(sess, tok, test, pool):
    se = enc(sess, tok, [s for s, _ in test]); ce = enc(sess, tok, pool)
    sims = se @ ce.T; idx = {c: i for i, c in enumerate(pool)}
    t1 = t5 = 0
    for i, (_, g) in enumerate(test):
        o = np.argsort(-sims[i])
        r = int(np.where(o == idx[g])[0][0]); t1 += r == 0; t5 += r < 5
    return t1/len(test), t5/len(test)


def main():
    import onnxruntime as ort
    random.seed(SEED)
    sp = load_specialized(); random.shuffle(sp)
    n = max(30, int(0.15*len(sp))); test = sp[:n]; pool = sorted({c for _, c in sp})
    tok = BertTokenizerFast.from_pretrained(FT)
    print("exporting fp32 onnx...", flush=True); export(tok)
    print("int8 quantizing...", flush=True); quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
    s32 = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(INT8, providers=["CPUExecutionProvider"])
    e32 = acc(s32, tok, test, pool); e8 = acc(s8, tok, test, pool)
    mb = lambda p: os.path.getsize(p)/1e6
    print(f"SIZE fp32_onnx={mb(FP32):.1f}MB int8_onnx={mb(INT8):.1f}MB ({mb(INT8)/mb(FP32)*100:.0f}%)", flush=True)
    print(f"ACC specialized fp32 top1={e32[0]:.3f} top5={e32[1]:.3f} | int8 top1={e8[0]:.3f} top5={e8[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
