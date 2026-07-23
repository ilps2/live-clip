"""
v3_clip_v2.py — V3 clip with visual boundary cross-validation.
Reads V2 candidates, uses SceneClassifier for intelligent cut points.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))  # motion-signature/
from scene_classifier import SceneClassifier

WD = Path(sys.argv[1])
VIDEO = sys.argv[2]
MAX_DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 55

# ---- Load V2 candidates ----
v2_csv = WD / "v2_candidates.csv"
if not v2_csv.exists():
    print("No v2_candidates.csv. Run fuse_v2.py first.")
    sys.exit(1)

import pandas as pd
cdf = pd.read_csv(v2_csv)
cands = cdf.to_dict("records")
if not cands:
    print("No V2 candidates.")
    sys.exit(0)

# ---- Load transcript ----
segs = [json.loads(l) for l in open(WD / "transcript.jsonl")]

# ---- Scene classifier for visual boundaries ----
sc = SceneClassifier(WD / "mv.npz")
try:
    meta = np.load(WD / "mv.npz", allow_pickle=True)["meta"].item()
    w, h = meta.get("width", 720), meta.get("height", 1280)
    orient = "vertical" if h > w else "horizontal"
except:
    orient = "vertical"
sc.classify(orientation=orient)
vbound = set(sc.boundaries(min_gap=1))

# ---- Merge nearby candidates ----
cands.sort(key=lambda c: c["start"])
GAP = 15
merged = []
for c in cands:
    if merged and c["start"] - merged[-1]["end"] < GAP:
        merged[-1]["end"] = c["end"]
        merged[-1]["peak_score"] = max(merged[-1].get("peak_score", 0), c.get("peak_score", 0))
    else:
        merged.append(dict(c))

print(f"Merged: {len(merged)} events")
for m in merged:
    print(f"  {m['start']}-{m['end']}s ({m['end']-m['start']}s) peak={m.get('peak_score',0):.1f}")

# ---- Pick best event, find optimal boundaries ----
best = max(merged, key=lambda m: m.get("peak_score", 0))
t0_raw, t1_raw = best["start"], best["end"]

# Cross-validate: find visual boundaries near desired cut points
# STEP 1: Start boundary — prefer visual boundary near t0_raw
t0 = t0_raw
for b in sorted(vbound):
    if t0_raw - 5 <= b <= t0_raw + 3:
        t0 = b
        break
# Validate with speech: find first complete sentence after t0
for s in segs:
    if s["start"] >= t0 - 2 and s["start"] <= t0 + 5:
        if any(s["text"].strip().endswith(p) for p in ["。", "！", "？"]):
            t0 = s["start"]
            break

# STEP 2: End boundary — prefer visual boundary near t0 + MAX_DUR, capped at t1_raw
target_end = min(t0 + MAX_DUR, t1_raw)
t1 = target_end
best_dist = float("inf")
for b in sorted(vbound):
    if target_end - 10 <= b <= target_end + 5:
        dist = abs(b - target_end)
        if dist < best_dist:
            t1 = b
            best_dist = dist
# Refine with speech: nearest sentence end to t1 (within ±5s)
for s in segs:
    if abs(s["end"] - t1) < 5:
        if any(s["text"].strip().endswith(p) for p in ["。", "！", "？", "了", "的", "吧", "呢", "吗", "行", "带", "看", "拍", "单"]):
            t1 = s["end"]
            break

# STEP 3: Check for price mention in the window. If none, extend to include nearest.
has_price = any(
    any(kw in s["text"] for kw in ["价格", "元", "多少钱", "优惠", "下单", "拍", "链接"])
    for s in segs if t0 <= s["start"] <= t1
)
if not has_price:
    for s in segs:
        if s["start"] > t1 and any(kw in s["text"] for kw in ["价格", "元", "多少钱", "优惠", "下单", "拍", "链接"]):
            t1 = min(s["end"] + 2, t0 + MAX_DUR + 10)
            break

# Enforce duration limits
if t1 - t0 < 30:
    t1 = min(t0 + 30, t1_raw)
if t1 - t0 > 90:
    t1 = t0 + 90

# STEP 4: Check for model/product name at start — if missing, extend backwards
has_name = any(
    any(kw in s["text"] for kw in ["就是", "介绍", "这款", "新品", "新成员", "上市", "首发"])
    for s in segs if s["start"] <= t0 + 10 and s["start"] >= t0 - 5
)
if not has_name:
    # Look backwards for product intro
    for s in reversed(segs):
        if s["start"] < t0 and any(kw in s["text"] for kw in ["就是", "介绍", "这款", "新品", "新成员"]):
            t0 = max(s["start"] - 0.5, t0 - 8)
            break

print(f"\nV3 clip: {t0:.0f}s - {t1:.0f}s ({t1-t0:.0f}s)")
print(f"Visual boundaries used: {t0 in vbound}/{t1 in vbound}")
print(f"Price mentioned: {has_price}")

# Show content
print(f"Content:")
for s in segs:
    if t0 <= s["start"] <= t1:
        scene = sc.per_second().get(int(s["start"]), "?")
        print(f"  {s['start']:.0f}s [{scene:7s}]: {s['text'][:80]}")

# ---- Cut ----
out = WD / "final_cut.mp4"
subprocess.run([
    "ffmpeg", "-y", "-v", "error",
    "-ss", str(t0), "-to", str(t1),
    "-i", VIDEO,
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-c:a", "aac", "-b:a", "128k",
    str(out)
], check=True)

r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(out)],
                   capture_output=True, text=True)
info = json.loads(r.stdout)
print(f"\nDone: {out} | {float(info['format']['duration']):.0f}s | {int(info['format']['size'])/1024:.0f}KB")
