#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""visualize.py — 运动签名可视化。

产出:
    results/quiver_<event>.png   每类事件典型帧的 MV 网格箭头叠加图（x4）
    results/quiver_real_tXX.png  真实录像典型帧 MV 箭头图
    results/timeline_synth.png   合成视频：运动量/中心边缘比曲线 + 事件标注
    results/timeline_real.png    真实录像：运动量/中心边缘比曲线
    results/signature_radar.png  4 类事件签名特征对比（分组柱状图）

用法:
    python visualize.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from daimon_runtime import setup_plot

from extract_mv import extract
from classify import FEATURES

setup_plot()
R = Path("results")
EVENT_ZH = {"product": "展示商品(中心小物体)", "gesture": "主播手势(大块不规则)",
            "camera": "镜头移动(全画面)", "static": "静止(近零运动)"}


def get_video_frame(video_path, target_idx):
    c = av.open(video_path)
    img = None
    for i, frame in enumerate(c.decode(c.streams.video[0])):
        if i == target_idx:
            img = frame.to_ndarray(format="rgb24")
            break
    c.close()
    return img


def quiver_overlay(video, npz_path, frame_idx, out, title):
    d = np.load(npz_path, allow_pickle=True)
    mag, dx, dy = d["mag"][frame_idx], d["dirx"][frame_idx], d["diry"][frame_idx]
    gh, gw = mag.shape
    img = get_video_frame(video, frame_idx)
    fig, ax = plt.subplots(figsize=(11, 6))
    if img is not None:
        ax.imshow(img, extent=[0, gw, gh, 0], alpha=0.55, aspect="auto")
    ys, xs = np.mgrid[0:gh, 0:gw]
    step = 2  # 隔格画箭头，避免太密
    m = mag[::step, ::step]
    scale = np.percentile(m, 95) + 1e-6
    m_disp = np.minimum(m, scale)  # 封顶显示长度，避免长射程匹配箭头刷屏
    ax.quiver(xs[::step, ::step] + 0.5, ys[::step, ::step] + 0.5,
              dx[::step, ::step] * m_disp / scale,
              dy[::step, ::step] * m_disp / scale,
              m, cmap="autumn", clim=[0, scale],
              angles="xy", scale_units="xy", scale=0.8, width=0.004)
    ax.set_xlim(0, gw); ax.set_ylim(gh, 0)
    ax.set_title(f"运动矢量场 · {title} (帧 {frame_idx})")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"[viz] {out}")


def timeline(npz_path, out, labels_path=None):
    d = np.load(npz_path, allow_pickle=True)
    mag = d["mag"]
    gh, gw = mag.shape[1:]
    gm = mag.mean(axis=(1, 2))
    cm = np.zeros((gh, gw), bool)
    cm[gh // 4: 3 * gh // 4, gw // 4: 3 * gw // 4] = True
    ce = mag[:, cm].mean(axis=1) / (mag[:, ~cm].mean(axis=1) + 1e-6)
    t = np.arange(len(gm)) / 30.0

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(t, gm, color="#4C72B0", lw=1)
    axes[0].set_ylabel("全局运动量 (px/帧)")
    axes[1].plot(t, ce, color="#55A868", lw=1)
    axes[1].set_ylabel("中心/边缘运动比")
    axes[1].set_xlabel("时间 (秒)")
    if labels_path:
        labels = pd.read_csv(labels_path)
        colors = {"product": "#DD8452", "gesture": "#937860",
                  "camera": "#C44E52", "static": "#8172B3"}
        for ax in axes:
            for (seg, ev), g in labels.groupby(["seg_id", "event"]):
                ax.axvspan(g.frame.min() / 30, g.frame.max() / 30,
                           color=colors[ev], alpha=0.22)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[e], alpha=0.4)
                   for e in colors]
        axes[0].legend(handles, [EVENT_ZH[e] for e in colors],
                       ncol=4, fontsize=9, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("运动签名时间轴")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"[viz] {out}")


def signature_bars(feat_csv, out):
    df = pd.read_csv(feat_csv)
    # 归一化到每特征最大值，便于同图比较
    norm = df.groupby("event")[FEATURES].mean()
    norm = norm / norm.max()
    x = np.arange(len(FEATURES))
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.2
    palette = {"product": "#DD8452", "gesture": "#937860",
               "camera": "#C44E52", "static": "#8172B3"}
    for k, (ev, row) in enumerate(norm.iterrows()):
        ax.bar(x + k * width, row.values, width,
               label=EVENT_ZH.get(ev, ev), color=palette.get(ev))
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(FEATURES, rotation=20)
    ax.set_ylabel("归一化特征均值")
    ax.set_title("四类事件的运动签名对比")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"[viz] {out}")


def main():
    synth = R / "synth.mp4"
    labels = pd.read_csv(R / "synth_labels.csv")
    # 每类取中段中间帧做 quiver
    for ev in ["product", "gesture", "camera", "static"]:
        seg = labels[labels.event == ev]
        mid_seg = seg.seg_id.unique()[len(seg.seg_id.unique()) // 2]
        fidx = int(seg[seg.seg_id == mid_seg].frame.iloc[60])
        quiver_overlay(synth, R / "synth_mv.npz", fidx,
                       R / f"quiver_{ev}.png", EVENT_ZH[ev])
    timeline(R / "synth_mv.npz", R / "timeline_synth.png", R / "synth_labels.csv")
    signature_bars(R / "frame_features.csv", R / "signature_bars.png")

    real = Path("../livestream-highlight/results/live_clip.flv")
    if (R / "real_mv.npz").exists() and real.exists():
        d = np.load(R / "real_mv.npz")
        # 真实录像取运动量最大/最小的代表帧
        gm = d["mag"].mean(axis=(1, 2))
        for tag, fidx in [("peak", int(np.argmax(gm))), ("calm", int(np.argmin(gm + 1)))]:
            quiver_overlay(real, R / "real_mv.npz", fidx,
                           R / f"quiver_real_{tag}.png", f"真实录像-{tag}")
        timeline(R / "real_mv.npz", R / "timeline_real.png")


if __name__ == "__main__":
    main()
