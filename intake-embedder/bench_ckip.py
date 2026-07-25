import os, resource, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

FR = ["糖化血色素", "健康關懷站", "嘉基", "戒菸衛教", "白血球表面標記", "低血壓",
      "定期心內門診-戒菸", "U.R.I.", "CCU", "不辣咖", "顳顎關節疼痛", "膽囊切除術"] * 8
HOME = os.path.expanduser("~")
MODELS = [f"{HOME}/med-embed/bert-tiny-chinese-st", f"{HOME}/med-embed/bert-base-chinese-st"]


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


for name in MODELS:
    base = rss()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    loaded = rss()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    for _ in range(2):
        ids = tok(FR[:8], max_length=32, truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            model(**ids)
    start = time.perf_counter()
    for i in range(0, len(FR), 16):
        ids = tok(FR[i:i + 16], max_length=32, truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            F.normalize(model(**ids).last_hidden_state[:, 0], dim=1)
    ms = (time.perf_counter() - start) * 1000 / len(FR)
    print(f"{os.path.basename(name)}\n  params={params:.0f}M resident={loaded - base:.0f}MB peak={rss():.0f}MB latency={ms:.2f} ms")
    del model, tok
