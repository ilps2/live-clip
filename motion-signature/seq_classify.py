#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""seq_classify.py — 时序分类 vs 帧级分类 & 帧率-准确率曲线。

回答两个问题:
    1. "连起来看"(GRU 序列分类) 是否比 "单帧看"(帧级 RF + 段内投票) 信息更多?
    2. 每秒 1 帧是否足够保留运动的动态发展信息? (1 / 5 / 15 / 30fps 对比)

实验设置:
    - 数据: synth_mv.npz (4 类 x 3 段 x 5 秒)
    - 特征: 每采样帧 8 维运动签名 (classify.frame_features, 不做时间平滑——
      把时间建模交给模型, 保证对比公平)
    - 序列模型: GRU(hidden=32) → 最后隐状态 → Linear(4), torch
    - 帧级基线: RandomForest(每帧) → 段内多数投票
    - 验证: Leave-One-Segment-Out CV (12 折), 报平均段级准确率

用法:
    python seq_classify.py
输出:
    results/seq_vs_frame.csv
    results/fps_curve.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from daimon_runtime import setup_plot

from classify import frame_features, FEATURES

setup_plot()
torch.manual_seed(0)
np.random.seed(0)
CLASSES = ["camera", "gesture", "product", "static"]
SEG_LEN_S = 5


def load_features(npz_path="results/synth_mv.npz", labels_path="results/synth_labels.csv"):
    d = np.load(npz_path, allow_pickle=True)
    labels = pd.read_csv(labels_path)
    n = min(len(d["mag"]), len(labels))
    feats = np.zeros((n, len(FEATURES)), np.float32)
    for i in range(n):
        f = frame_features(d["mag"][i], d["dirx"][i], d["diry"][i])
        feats[i] = [f[k] for k in FEATURES]
    seg_ids = labels["seg_id"].to_numpy()[:n]
    events = labels["event"].to_numpy()[:n]
    return feats, seg_ids, events


class GRUClassifier(nn.Module):
    def __init__(self, in_dim=8, hidden=32, n_classes=4):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(h[:, -1])


def train_gru(X_train, y_train, epochs=60, lr=1e-2, n_classes=4):
    model = GRUClassifier(n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    X = torch.tensor(X_train)
    y = torch.tensor(y_train)
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(model(X), y)
        loss.backward()
        opt.step()
    return model


def sample_segment(feats, seg_mask, fps, src_fps=30):
    """在段内按 fps 抽帧索引。"""
    idx = np.where(seg_mask)[0]
    step = max(1, round(src_fps / fps))
    return idx[::step]


def run_loso(feats, seg_ids, events, fps, src_fps=30, epochs=60, n_folds=None,
             aug_reps=6, classes=None):
    """段级交叉验证：默认逐段留一；n_folds 指定时分组留多段（大数据省时）。"""
    # 标准化（全体数据估计，特征工程层面，无标签泄漏）
    mu, sd = feats.mean(0), feats.std(0) + 1e-9
    F = (feats - mu) / sd
    cls_idx = {c: i for i, c in enumerate(classes or CLASSES)}
    segs = sorted(set(seg_ids))
    if n_folds and n_folds < len(segs):
        folds = [set(segs[i::n_folds]) for i in range(n_folds)]
    else:
        folds = [{s} for s in segs]
    for fold in folds:
        test_mask = np.isin(seg_ids, list(fold))
        train_mask = ~test_mask
        train_segs = [tr for tr in segs if tr not in fold]
        # --- 序列分类 (GRU) ---
        tr_idx = [sample_segment(F, seg_ids == tr, fps, src_fps)
                  for tr in train_segs]
        Xtr = np.stack([F[i] for i in tr_idx])
        ytr = np.array([cls_idx[events[seg_ids == tr][0]] for tr in train_segs])
        # 数据增强：时间窗随机裁剪 + 特征噪声
        Xa, ya = [x for x in Xtr], list(ytr)
        rngg = np.random.default_rng(0)
        for rep in range(aug_reps):
            for xi, yi in zip(Xtr, ytr):
                n = len(xi)
                if n >= 8:
                    st = rngg.integers(0, n // 4 + 1)
                    xi = xi[st: st + max(4, n - n // 4)]
                Xa.append(xi + rngg.normal(0, 0.05, xi.shape).astype(np.float32))
                ya.append(yi)
        # 补齐到同一长度（右端零填充）
        L = max(len(x) for x in Xa)
        Xp = np.zeros((len(Xa), L, Xtr.shape[-1]), np.float32)
        for k, x in enumerate(Xa):
            Xp[k, :len(x)] = x
        model = train_gru(Xp, np.array(ya), epochs=epochs,
                          n_classes=len(cls_idx))
        model.eval()
        # --- 帧级基线: RF + 段内投票 ---
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(F[train_mask], [cls_idx[e] for e in events[train_mask]])
        for s in fold:
            sm = seg_ids == s
            te_i = sample_segment(F, sm, fps, src_fps)
            Xte = torch.tensor(F[te_i][None])
            with torch.no_grad():
                pred_seq = int(model(Xte).argmax(1)[0])
            votes = rf.predict(F[te_i])
            pred_frame = int(np.bincount(votes).argmax())
            true = cls_idx[events[sm][0]]
            yield s, pred_seq == true, pred_frame == true


def main():
    feats, seg_ids, events = load_features()
    rows = []
    for fps in [1, 5, 15, 30]:
        seq_ok, frame_ok = [], []
        for s, so, fo in run_loso(feats, seg_ids, events, fps, epochs=60):
            seq_ok.append(so); frame_ok.append(fo)
        rows.append({"fps": fps,
                     "seq_gru_acc": np.mean(seq_ok),
                     "frame_rf_vote_acc": np.mean(frame_ok)})
        print(f"[fps={fps:>2}] GRU序列: {np.mean(seq_ok):.3f}  "
              f"帧级RF+投票: {np.mean(frame_ok):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/seq_vs_frame.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df.fps, df.seq_gru_acc, "o-", label="GRU 序列分类 (连起来看)")
    ax.plot(df.fps, df.frame_rf_vote_acc, "s--", label="帧级 RF + 投票 (单帧看)")
    ax.set_xscale("log"); ax.set_xticks(df.fps); ax.set_xticklabels(df.fps)
    ax.set_xlabel("采样帧率 (fps)"); ax.set_ylabel("段级分类准确率 (LOSO)")
    ax.set_ylim(0, 1.05)
    ax.set_title("运动动画帧率 vs 分类准确率")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/fps_curve.png", bbox_inches="tight", dpi=110)
    print("[seq] saved results/seq_vs_frame.csv, results/fps_curve.png")


if __name__ == "__main__":
    main()
