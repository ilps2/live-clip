"""
fuse_v2.py — V2 fusion with scene classifier weights.
Replaces fuse_mini.py. Uses scene_classifier for per-second weights.
"""
import json, subprocess, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

# ---- Args ----
WD = Path(sys.argv[1]) 
VIDEO = sys.argv[2]
FORMAT = sys.argv[3] if len(sys.argv) > 3 else "livestream"  # or "launch"

sys.path.insert(0, str(Path(__file__).parent.parent))  # motion-signature/
from scene_classifier import SceneClassifier

W2, W3, THRESH = 1.5, 0.5, 1.0

def zscore(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

# ---- Scene weights ----
sc = SceneClassifier(WD / "mv.npz")
# Determine orientation from MV meta
try:
    meta = np.load(WD / "mv.npz", allow_pickle=True)["meta"].item()
    w, h = meta.get("width", 720), meta.get("height", 1280)
    orient = "vertical" if h > w else "horizontal"
except:
    orient = "vertical"
sc.classify(orientation=orient)
scene_w = sc.weights(format_type=FORMAT)
bounds = sc.boundaries()

print(f"[scene] {dict(sc.per_second().value_counts())} | orient={orient} | boundaries={len(bounds)}")

# ---- Soft promo (same as fuse_mini.py) ----
SOFT = [
    ("拍下", 0.7), ("可以拍", 0.6), ("拍起来", 0.6), ("下单", 0.8),
    ("号链接", 0.8), ("几号", 0.4), ("多少钱", 0.5), ("价格", 0.5),
    ("优惠", 0.6), ("福利", 0.6), ("券", 0.5), ("包邮", 0.6),
    ("发货", 0.4), ("库存", 0.6), ("放心拍", 0.7), ("去下单", 0.8),
    ("去拍", 0.6), ("带", 0.3), ("拍", 0.3), ("号", 0.3),
    ("合适价格", 0.5), ("看好这个价格", 0.5), ("尺码", 0.3), ("颜色", 0.3),
]

segs = [json.loads(l) for l in open(WD / "transcript.jsonl")]
promo = []
for s in segs:
    w = sum(wt for pat, wt in SOFT if pat in s["text"])
    if w > 0:
        promo.append({"t": (s["start"] + s["end"]) / 2, "weight": min(w, 2.5),
                      "text": s["text"]})
pe = WD / "promo_events.jsonl"
if pe.exists() and pe.stat().st_size > 0:
    for l in open(pe):
        e = json.loads(l)
        promo.append({"t": (e["start"] + e["end"]) / 2, "weight": e["weight"] + 0.8,
                      "text": e["text"]})
promo.sort(key=lambda x: x["t"])
print(f"[promo] {len(promo)} events")

# ---- Voice activity ----
dur = sc.dur
va = np.zeros(int(dur) + 2)
for s in segs:
    for sec in range(int(s["start"]), min(int(s["end"]) + 1, len(va))):
        va[sec] = 1

# ---- Per-second scoring ----
secs = np.arange(0, int(dur) + 1)
promo_s = np.zeros(len(secs))
for e in promo:
    promo_s[int(e["t"])] += e["weight"]
promo_z = zscore(pd.Series(promo_s).rolling(15, center=True, min_periods=1).sum().to_numpy())
va_s = va[:len(secs)]

# FUSION: scene_weight * (1 + promo + voice)
score = scene_w[:len(secs)] * (1.0 + W2 * np.maximum(promo_z, 0) + W3 * va_s)
tl = pd.DataFrame({"sec": secs, "scene_weight": scene_w[:len(secs)],
                   "promo_density": promo_z, "voice": va_s, "score": score})
tl.to_csv(WD / "timeline_scores.csv", index=False)

# ---- Candidates ----
hot = tl[tl.score >= THRESH]
cands = []
if len(hot):
    hs = hot.sec.to_numpy()
    bounds_arr = np.where(np.diff(hs) > 5)[0] + 1
    groups = np.split(np.arange(len(hot)), bounds_arr) if len(bounds_arr) else [np.arange(len(hot))]
    for gi in groups:
        if len(gi) == 0: continue
        g = hot.iloc[gi]
        s0, e0 = int(g.sec.min()), int(g.sec.max()) + 1
        s0, e0 = max(0, s0 - 8), min(int(dur), e0 + 8)
        if e0 - s0 < 15:
            c = (s0 + e0) // 2
            s0, e0 = max(0, c - 15), min(int(dur), c + 15)
        if e0 - s0 > 75:
            e0 = s0 + 75
        peak = float(g.score.max())
        comp = []
        if g.scene_weight.max() > 1.0: comp.append("高价值画面")
        if promo_s[s0:e0].sum() > 0.5: comp.append("促单话术")
        if va_s[s0:e0].mean() > 0.7: comp.append("语音密集")
        cands.append({"id": len(cands) + 1, "start": s0, "end": e0,
                      "dur": e0 - s0, "peak_score": round(peak, 2),
                      "signals": "+".join(comp)})

# Merge overlaps
merged = []
for c in sorted(cands, key=lambda x: x["start"]):
    if merged and c["start"] <= merged[-1]["end"]:
        m = merged[-1]
        m["end"] = max(m["end"], c["end"])
        m["dur"] = m["end"] - m["start"]
        m["peak_score"] = max(m["peak_score"], c["peak_score"])
        m["signals"] = "+".join(sorted(set(m["signals"].split("+") + c["signals"].split("+")) - {""}))
    else:
        merged.append(dict(c))
for i, m in enumerate(merged):
    m["id"] = i + 1

cdf = pd.DataFrame(merged)
cdf.to_csv(WD / "candidates.csv", index=False)
print(f"[fuse] {len(cdf)} candidates:")
print(cdf.to_string(index=False))

# ---- V2 filtering: msc needs motion_hot AND promo_active ----
# Motion hot = scene_weight > 0.8 (face/product)
# Promo active = any promo event within ±12s
v2_cands = []
for _, c in cdf.iterrows():
    s0, e0 = int(c["start"]), int(c["end"])
    motion_hot = scene_w[s0:e0].max() > 0.8
    promo_active = any(abs(e["t"] - (s0 + e0) / 2) < 12 for e in promo)
    if motion_hot and promo_active:
        # Tighten boundaries to nearest visual boundaries
        bs = sc.nearest_visual_boundary(s0, 5)
        be = sc.nearest_visual_boundary(e0, 5)
        s0, e0 = min(s0, bs), max(e0, be)  # expand, don't shrink
        v2_cands.append({"id": len(v2_cands) + 1, "start": s0, "end": e0,
                         "dur": e0 - s0, "peak_score": c["peak_score"],
                         "signals": "运动高峰+促单话术(段级汇聚)"})

v2df = pd.DataFrame(v2_cands) if v2_cands else pd.DataFrame()
v2df.to_csv(WD / "v2_candidates.csv", index=False)
print(f"\n[V2] {len(v2df)} candidates:")
if len(v2df):
    print(v2df.to_string(index=False))

# ---- Annotation sheet ----
ann = v2df.copy() if len(v2df) else cdf.copy()
ann["is_highlight(是/否)"] = ""
ann["备注"] = ""
ann.to_csv(WD / "annotation_sheet.csv", index=False)
print(f"\n[done] outputs in {WD}/")
