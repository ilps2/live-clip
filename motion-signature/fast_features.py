#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fast_features.py — 向量化批量计算 8 维运动签名特征。

等价于 classify.frame_features 的逐帧循环，但对 (T,H,W) 整体 numpy 向量化，
18k 帧从数分钟降到秒级。用法:
    python fast_features.py results/tp_pair1   # 读 _mv.npz + _labels.csv, 写 _feats.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["global_mag", "move_ratio", "centroid_x", "centroid_y",
            "spread", "center_edge", "dir_consist", "max_mag"]


def batch_features(mag, dirx, diry):
    T, gh, gw = mag.shape
    ys, xs = np.mgrid[0:gh, 0:gw]
    xs_n = xs / gw
    ys_n = ys / gh
    tot = mag.sum(axis=(1, 2))                      # (T,)
    valid = tot >= 1e-6
    safe_tot = np.where(valid, tot, 1.0)
    cx = (mag * xs_n).sum(axis=(1, 2)) / safe_tot   # (T,)
    cy = (mag * ys_n).sum(axis=(1, 2)) / safe_tot
    dx2 = (xs_n[None] - cx[:, None, None]) ** 2 + (ys_n[None] - cy[:, None, None]) ** 2
    var = (mag * dx2).sum(axis=(1, 2)) / safe_tot
    spread = np.sqrt(var)
    moving = mag > 1.0
    mr = moving.mean(axis=(1, 2))
    cm = np.zeros((gh, gw), bool)
    cm[gh // 4: 3 * gh // 4, gw // 4: 3 * gw // 4] = True
    center = mag[:, cm].mean(axis=1)
    edge = mag[:, ~cm].mean(axis=1)
    ce = center / (edge + 1e-6)
    ce = np.clip(ce, 0, 20)  # 边缘运动≈0 时比值爆炸, 封顶防垃圾特征
    cnt = moving.sum(axis=(1, 2))
    safe_cnt = np.where(cnt > 0, cnt, 1)
    dc = np.hypot(np.where(moving, dirx, 0).sum(axis=(1, 2)),
                  np.where(moving, diry, 0).sum(axis=(1, 2))) / safe_cnt
    dc = np.where(cnt > 0, dc, 0.0)
    out = np.stack([mag.mean(axis=(1, 2)), mr, cx, cy, spread, ce, dc,
                    mag.max(axis=(1, 2))], axis=1).astype(np.float32)
    out[~valid] = 0.0
    return out


def main():
    prefix = sys.argv[1]
    d = np.load(f"{prefix}_mv.npz", allow_pickle=True)
    labels = pd.read_csv(f"{prefix}_labels.csv")
    n = min(len(d["mag"]), len(labels))
    feats = batch_features(d["mag"][:n], d["dirx"][:n], d["diry"][:n])
    np.savez_compressed(f"{prefix}_feats.npz", feats=feats,
                        seg_ids=labels["seg_id"].to_numpy()[:n],
                        events=labels["event"].to_numpy()[:n])
    print(f"[fast] {prefix}: {feats.shape}")


if __name__ == "__main__":
    main()
