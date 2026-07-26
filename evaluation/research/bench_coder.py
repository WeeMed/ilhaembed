"""CODER-TW on the same jargon benchmark, using the graph's own pooled output.

Its ONNX export bakes pooling in: one output named "pooler", fixed 32-token sequence, no exposed
last_hidden_state. That is the readout its own evaluation pipeline uses, so it is measured the way it
would actually be deployed rather than re-pooled into a shape it was never trained for."""
import os, resource, time, json, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from bench_all import load_jargon, retrieval_accuracy

MODEL_DIR = os.path.expanduser(os.environ.get("CODER_TW_MODEL_DIR", "./models/coder-tw-onnx"))
ONNX = os.path.join(MODEL_DIR, "model.onnx")

rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
baseline = rss()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
sess = ort.InferenceSession(ONNX, opts, providers=["CPUExecutionProvider"])
loaded = rss()
names = {i.name for i in sess.get_inputs()}

def embed(texts):
    out = []
    for s in range(0, len(texts), 32):
        b = tok(texts[s:s+32], max_length=32, truncation=True, padding="max_length", return_tensors="np")
        feed = {k: b[k].astype("int64") for k in names if k in b}
        v = sess.run(None, feed)[0]
        if v.ndim == 3:
            v = v[:, 0]
        n = np.linalg.norm(v, axis=1, keepdims=True); n[n == 0] = 1.0
        out.append(v / n)
    return np.vstack(out)

spec, slang = load_jargon()
probe = [a for a, _ in spec[:96]]
embed(probe[:16])
t0 = time.perf_counter(); embed(probe); lat = (time.perf_counter()-t0)*1000/len(probe)
print(json.dumps({"model": "coder-tw-int8", "params_or_mb": round(os.path.getsize(ONNX)/1e6, 1),
                  "resident_mb": round(loaded-baseline), "peak_mb": round(rss()), "latency_ms": round(lat, 2),
                  "specialized": retrieval_accuracy(embed, spec), "slang": retrieval_accuracy(embed, slang)},
                 ensure_ascii=False))
