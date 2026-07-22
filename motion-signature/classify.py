#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""classify.py — 运动签名特征提取 + 4 类事件帧级分类实验。

特征（每帧 8 维，全部来自编码器 MV 网格）:
    1. global_mag      全局平均运动幅度
    2. move_ratio      运动块占比（幅度 > 1.0px）
    3. centroid_x/y    运动质心（幅度加权，归一化）
    4. spread          运动空间集中度（质心周围加权方差）
    5. center_edge     中心区域 / 边缘区域平均运动比
    6. dir_consist     方向一致性 |Σ单位向量| / N
    7. max_mag         最大块运动幅度

用法:
    python classify.py --mv results/synth_mv.npz --labels results/synth_labels.csv \
        --report results/classification.txt --cm results/confusion_matrix.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = ["global_mag", "move_ratio", "centroid_x", "centroid_y",
            "spread", "center_edge", "dir_consist", "max_mag"]


def frame_features(mag, dirx, diry):
    gh, gw = mag.shape
    ys, xs = np.mgrid[0:gh, 0:gw]
    tot = mag.sum()
    if tot < 1e-6:
        return dict.fromkeys(FEATURES, 0.0)
    cx = (mag * xs).sum() / tot / gw
    cy = (mag * ys).sum() / tot / gh
    var = (mag * ((xs / gw - cx) ** 2 + (ys / gh - cy) ** 2)).sum() / tot
    moving = mag > 1.0
    mr = float(moving.mean())
    # 中心 50% 区域 vs 边缘
    cm = np.zeros_like(mag, bool)
    cm[gh // 4: 3 * gh // 4, gw // 4: 3 * gw // 4] = True
    center = mag[cm].mean()
    edge = mag[~cm].mean()
    ce = center / (edge + 1e-6)
    if moving.any():
        dc = float(np.hypot(dirx[moving].sum(), diry[moving].sum()) / moving.sum())
    else:
        dc = 0.0
    return {"global_mag": float(mag.mean()), "move_ratio": mr,
            "centroid_x": float(cx), "centroid_y": float(cy),
            "spread": float(np.sqrt(var)), "center_edge": float(ce),
            "dir_consist": dc, "max_mag": float(mag.max())}


def build_dataset(npz_path, labels_path):
    d = np.load(npz_path, allow_pickle=True)
    labels = pd.read_csv(labels_path)
    n = min(len(d["mag"]), len(labels))
    rows = []
    for i in range(n):
        if d["pict"][i] == "I":   # I 帧无 MV，跳过（或用前后帧插值，这里简单跳过）
            continue
        f = frame_features(d["mag"][i], d["dirx"][i], d["diry"][i])
        f["frame"] = i
        f["event"] = labels.iloc[i]["event"]
        f["seg_id"] = labels.iloc[i]["seg_id"]
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mv", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--report", default="results/classification.txt")
    ap.add_argument("--cm", default="results/confusion_matrix.csv")
    ap.add_argument("--feat", default="results/frame_features.csv")
    args = ap.parse_args()

    df = build_dataset(args.mv, args.labels)
    # 时间平滑（段内，±2 帧滚动均值）：抑制单帧 MV 噪声，更贴近"事件"尺度
    df = df.sort_values("frame")
    df[FEATURES] = (df.groupby("seg_id")[FEATURES]
                      .transform(lambda s: s.rolling(5, center=True, min_periods=1).mean()))
    df.to_csv(args.feat, index=False)

    # 段级划分：每类 3 段，2 段训练 1 段测试（避免同段相邻帧泄漏）
    test_segs = df.groupby("event")["seg_id"].max()
    test = df[df.apply(lambda r: r.seg_id == test_segs[r.event], axis=1)]
    train = df.drop(test.index)

    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(train[FEATURES], train["event"])
    pred = clf.predict(test[FEATURES])
    acc = float((pred == test["event"]).mean())

    classes = sorted(df["event"].unique())
    cm = confusion_matrix(test["event"], pred, labels=classes)
    rep = classification_report(test["event"], pred, digits=3)

    txt = (f"帧级分类（段级划分 train={len(train)} test={len(test)}）\n"
           f"准确率: {acc:.3f}\n\n混淆矩阵 (行=真实, 列=预测)\nclasses={classes}\n{cm}\n\n{rep}\n"
           f"特征重要性: {dict(zip(FEATURES, np.round(clf.feature_importances_,3)))}\n")
    print(txt)
    Path(args.report).write_text(txt, encoding="utf-8")
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(args.cm)


if __name__ == "__main__":
    main()
