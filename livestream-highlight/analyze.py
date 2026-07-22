#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze.py — 时间轴可视化：弹幕速率 + 促单事件 + 融合分数 + 阈值 + GT 区间。

用法:
    python analyze.py --scores results/scores.csv \
        [--promo results/promo_events.jsonl] \
        [--ground-truth results/ground_truth.csv] \
        --out results/timeline.png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__import__("sys").executable).parent.parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from daimon_runtime import setup_plot

setup_plot()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--promo", default=None)
    ap.add_argument("--ground-truth", default=None)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--out", default="results/timeline.png")
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    t = (df["t_start"] + df["t_end"]) / 2

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 2]})

    # 1. 弹幕速率
    ax = axes[0]
    ax.plot(t, df["dm_rate"], color="#4C72B0", lw=1.2)
    ax.set_ylabel("弹幕速率 (条/秒)")
    ax.set_title("直播高光时刻检测 · 廉价信号时间线")
    ax.grid(alpha=0.3)

    # 2. 促单事件（stem 标记，高度=权重）
    ax = axes[1]
    promo_t, promo_w = [], []
    if args.promo and Path(args.promo).exists():
        with open(args.promo, encoding="utf-8") as f:
            for l in f:
                e = json.loads(l)
                promo_t.append((e["start"] + e["end"]) / 2)
                promo_w.append(e["weight"])
    if promo_t:
        ax.stem(promo_t, promo_w, linefmt="#DD8452", markerfmt="o", basefmt=" ")
        ax.set_ylim(0, max(promo_w) * 1.3)
    else:
        ax.text(0.5, 0.5, "无促单事件（ASR 未运行或无命中）",
                transform=ax.transAxes, ha="center", va="center", color="#888")
    ax.set_ylabel("促单事件强度")
    ax.grid(alpha=0.3)

    # 3. 融合分数 + 阈值 + GT
    ax = axes[2]
    ax.plot(t, df["score"], color="#55A868", lw=1.5, label="融合分数")
    ax.axhline(args.threshold, color="#C44E52", ls="--", lw=1,
               label=f"阈值={args.threshold}")
    ax.fill_between(t, df["score"], args.threshold,
                    where=df["score"] >= args.threshold,
                    color="#C44E52", alpha=0.25, label="候选高光")
    if args.ground_truth and Path(args.ground_truth).exists():
        gt = pd.read_csv(args.ground_truth)
        ymax = df["score"].max()
        for i, (_, r) in enumerate(gt.iterrows()):
            ax.axvspan(r.start, r.end, color="gold", alpha=0.35,
                       label="人工标注高光 (GT)" if i == 0 else None)
    ax.set_ylabel("融合分数 (z)")
    ax.set_xlabel("时间 (秒)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=120)
    print(f"[analyze] saved -> {args.out}")


if __name__ == "__main__":
    main()
