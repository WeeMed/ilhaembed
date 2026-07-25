"""Why the multi-similarity loss did not move: measure the distribution it was configured against.

A loss that is flat across four epochs is not a result about the method, it is a mis-configuration.
MS loss is defined around a margin `base`: positives below it and negatives above it are penalised.
If the real cosine distribution sits far from that base, one term saturates, the other underflows,
and the gradient is negligible regardless of how wrong the model is."""
import os, random, csv, gzip, sys
from collections import defaultdict
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sapbert import build_groups, encode, MS_BASE, MS_ALPHA, MS_BETA, MS_EPSILON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5").to(DEV).eval()

groups = build_groups()
keys = sorted(groups)
random.Random(0).shuffle(keys)

texts, labels = [], []
for label, key in enumerate(keys[:48]):
    for surface in list(groups[key])[:4]:
        texts.append(surface); labels.append(label)

with torch.no_grad():
    vectors = encode(model, tok, texts)
labels_t = torch.tensor(labels, device=DEV)
sim = vectors @ vectors.T
same = labels_t[:, None] == labels_t[None, :]
eye = torch.eye(len(labels), dtype=torch.bool, device=DEV)
pos = sim[same & ~eye]
neg = sim[~same]

q = lambda t, p: torch.quantile(t.float(), p).item()
print(f"batch: {len(texts)} surfaces / {len(set(labels))} concepts")
print(f"POSITIVE cosine  min={pos.min():.3f}  p5={q(pos,.05):.3f}  median={q(pos,.5):.3f}  p95={q(pos,.95):.3f}  max={pos.max():.3f}")
print(f"NEGATIVE cosine  min={neg.min():.3f}  p5={q(neg,.05):.3f}  median={q(neg,.5):.3f}  p95={q(neg,.95):.3f}  max={neg.max():.3f}")
print(f"\nconfigured MS_BASE={MS_BASE}  alpha={MS_ALPHA} beta={MS_BETA} eps={MS_EPSILON}")
print(f"positives BELOW base (penalised): {(pos < MS_BASE).float().mean():.1%}")
print(f"negatives ABOVE base (penalised): {(neg > MS_BASE).float().mean():.1%}")

# How many anchors actually survive mining and contribute gradient?
contributing = 0
for i in range(len(labels)):
    p = sim[i][same[i] & ~eye[i]]; n = sim[i][~same[i]]
    if p.numel() == 0 or n.numel() == 0: continue
    hp = p[p < n.max() + MS_EPSILON]; hn = n[n > p.min() - MS_EPSILON]
    if hp.numel() and hn.numel(): contributing += 1
print(f"anchors contributing gradient: {contributing}/{len(labels)}")
