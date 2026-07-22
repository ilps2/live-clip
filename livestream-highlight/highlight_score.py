#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""highlight_score.py — 廉价信号融合，产出高光分数与候选片段。

信号:
    1. 弹幕信号: 30s 滑动窗口弹幕速率 -> z-score -> 峰值
    2. 促单信号: promo 事件加权密度（30s 窗口内权重和 -> z-score）
融合: score = w1 * z_danmaku + w2 * z_promo  (w 可配; 无转录时 w2=0)

用法:
    python highlight_score.py --danmaku results/danmaku.jsonl \
        [--promo results/promo_events.jsonl] \
        --scores results/scores.csv --segments results/segments.json \
        [--ground-truth results/ground_truth.csv]

ground truth CSV schema: start,end  (秒, 与 rel_t 同一时间轴)
片段级命中判定: 预测片段与任一 GT 片段 IoU > 0.5 视为命中。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 30.0     # 滑动窗口（秒）
STEP = 5.0        # 滑窗步长（秒）
THRESHOLD = 1.0   # 融合分数阈值（z 分空间），可在实验 A 中调参


def sliding_rate(times: np.ndarray, t_max: float,
                 window: float = WINDOW, step: float = STEP) -> pd.DataFrame:
    """滑动窗口事件速率（次/秒）。"""
    starts = np.arange(0, max(t_max - window, step), step)
    rows = []
    for s in starts:
        n = int(((times >= s) & (times < s + window)).sum())
        rows.append({"t_start": s, "t_end": s + window, "rate": n / window})
    return pd.DataFrame(rows)


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def compute_scores(danmaku_path: str | None, promo_path: str | None,
                   w1: float = 1.0, w2: float = 1.0,
                   window: float = WINDOW, step: float = STEP) -> pd.DataFrame:
    times = []
    if danmaku_path and Path(danmaku_path).exists():
        with open(danmaku_path, encoding="utf-8") as f:
            times = [json.loads(l)["rel_t"] for l in f]
    promo_weight_at = []  # (t, weight)
    if promo_path and Path(promo_path).exists():
        with open(promo_path, encoding="utf-8") as f:
            for l in f:
                e = json.loads(l)
                promo_weight_at.append(((e["start"] + e["end"]) / 2, e["weight"]))
    else:
        w2 = 0.0  # 无促单信号降级：纯弹幕信号

    t_max = max(times + [t for t, _ in promo_weight_at]) if (times or promo_weight_at) else window * 2
    df = sliding_rate(np.array(sorted(times)), t_max, window, step)
    df = df.rename(columns={"rate": "dm_rate"})

    pw = np.array([w for _, w in promo_weight_at]) if promo_weight_at else np.array([])
    pt = np.array([t for t, _ in promo_weight_at]) if promo_weight_at else np.array([])
    promo_dens = []
    for _, r in df.iterrows():
        m = (pt >= r.t_start) & (pt < r.t_end)
        promo_dens.append(float(pw[m].sum()) / window if len(pw) else 0.0)
    df["promo_density"] = promo_dens

    df["z_dm"] = zscore(df["dm_rate"].to_numpy())
    df["z_promo"] = zscore(df["promo_density"].to_numpy())
    df["score"] = w1 * df["z_dm"] + w2 * df["z_promo"]
    df.attrs["w1"], df.attrs["w2"] = w1, w2
    return df


def extract_segments(df: pd.DataFrame, threshold: float = THRESHOLD,
                     min_dur: float = 15.0, pad: float = 5.0) -> list:
    """超过阈值的连续区间 -> 候选高光片段（前后各 pad 秒，合并重叠）。"""
    hot = df[df["score"] >= threshold]
    segs = []
    if hot.empty:
        return segs
    cur_s, cur_e, cur_peak = hot.iloc[0].t_start, hot.iloc[0].t_end, hot.iloc[0].score
    for _, r in hot.iloc[1:].iterrows():
        if r.t_start <= cur_e + 1e-6:
            cur_e = r.t_end
            cur_peak = max(cur_peak, r.score)
        else:
            segs.append((cur_s, cur_e, cur_peak))
            cur_s, cur_e, cur_peak = r.t_start, r.t_end, r.score
    segs.append((cur_s, cur_e, cur_peak))
    out = []
    for s, e, p in segs:
        s2, e2 = max(0.0, s - pad), e + pad
        if e2 - s2 < min_dur:
            c = (s2 + e2) / 2
            s2, e2 = max(0.0, c - min_dur / 2), c + min_dur / 2
        # 与上一个合并重叠
        if out and s2 <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], e2)
            out[-1]["peak_score"] = max(out[-1]["peak_score"], round(p, 3))
        else:
            out.append({"start": round(s2, 1), "end": round(e2, 1),
                        "peak_score": round(p, 3)})
    return out


def load_ground_truth(csv_path: str) -> list:
    gt = pd.read_csv(csv_path)
    return [{"start": float(r.start), "end": float(r.end)} for _, r in gt.iterrows()]


def _iou(a, b) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def evaluate(pred_segs: list, gt_segs: list, iou_thresh: float = 0.5) -> dict:
    """片段级 P/R/F1：预测片段与任一 GT IoU>iou_thresh 为 TP。"""
    tp = sum(1 for p in pred_segs if any(_iou(p, g) > iou_thresh for g in gt_segs))
    fp = len(pred_segs) - tp
    fn = sum(1 for g in gt_segs if not any(_iou(p, g) > iou_thresh for p in pred_segs))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--danmaku", default=None)
    ap.add_argument("--promo", default=None)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--segments", default="results/segments.json")
    ap.add_argument("--ground-truth", default=None)
    args = ap.parse_args()

    df = compute_scores(args.danmaku, args.promo, args.w1, args.w2)
    df.to_csv(args.scores, index=False)
    segs = extract_segments(df, args.threshold)
    Path(args.segments).write_text(
        json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[score] windows={len(df)} segments={len(segs)} -> {args.scores}, {args.segments}")

    if args.ground_truth and Path(args.ground_truth).exists():
        gt = load_ground_truth(args.ground_truth)
        metrics = evaluate(segs, gt)
        print(f"[eval] {metrics}")


if __name__ == "__main__":
    main()
