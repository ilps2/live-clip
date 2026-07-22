#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""synth_events.py — 程序化生成 4 类视觉事件的合成视频（真实 H.264 编码）。

4 类事件（对应带货直播典型场景）:
    product : 中心小物体往复运动   （主播举起商品在镜头前晃动展示）
    gesture : 大块不规则运动       （主播手势/身体动作）
    camera  : 全画面一致运动       （镜头移动/推拉/场景切换）
    static  : 近零运动             （静止画面/口播固定机位）

生成: numpy 渲染帧 -> ffmpeg stdin 编码 libx264（CRF 23，真实 MV）
输出: video.mp4 + labels.csv (frame,event,seg_id)

用法:
    python synth_events.py --out results/synth.mp4 --labels results/synth_labels.csv
"""
import argparse
import csv
import subprocess

import numpy as np

W, H, FPS = 640, 360, 30
SEG_DUR = 5  # 每段秒数
rng = np.random.default_rng(7)


_NOISE = rng.normal(0, 6, (H, W))  # 固定噪声纹理（静态背景，MV 应≈0）
_HARD = False

def background(t):
    """带噪纹理背景。easy: 噪声固定; hard: 逐帧微变(编码器 MV 噪声)。"""
    y, x = np.mgrid[0:H, 0:W]
    bg = (np.sin(x / 37) + np.cos(y / 29) + 2) * 40
    bg += _NOISE
    if _HARD:
        bg += rng.normal(0, 3.5, (H, W))
    return np.clip(bg, 0, 255).astype(np.uint8)


def gen_segment(event, n_frames):
    """生成一段 n_frames 的灰度帧。"""
    frames = []
    # camera: 预生成一张大图做平移
    if event == "camera":
        big = background(0)
        big = np.tile(big, (3, 3))[: H * 2, : W * 2]
        if _HARD:  # 慢镜头与手势易混; 偶尔斜向
            vx = rng.uniform(1.2, 5.5)
            vy = rng.uniform(-2, 2) * rng.choice([0.3, 1.0])
        else:
            vx, vy = rng.uniform(3, 6), rng.uniform(-2, 2)
    elif event == "gesture":
        bx, by = W * 0.3, H * 0.5
    rr = 35 if not _HARD else int(rng.integers(22, 40))
    pause_at = rng.integers(30, n_frames - 30) if _HARD else -1  # 中途停顿
    for f in range(n_frames):
        img = background(f)
        t = f / FPS
        if event == "product":
            amp = 0.0 if (_HARD and pause_at <= f < pause_at + 20) else 1.0
            cx = int(W / 2 + amp * 60 * np.sin(2 * np.pi * t / 1.5))
            cy = int(H / 2 + amp * 20 * np.sin(2 * np.pi * t / 0.9))
            img[max(0, cy-rr):cy+rr, max(0, cx-rr):cx+rr] = 230
        elif event == "gesture":
            bx += rng.normal(0, 8); by += rng.normal(0, 6)
            bx = np.clip(bx, 120, W-120); by = np.clip(by, 100, H-100)
            rw, rh = 110, 80
            img[int(by-rh):int(by+rh), int(bx-rw):int(bx+rw)] = \
                120 + 80 * np.sin(t * 5)
        elif event == "camera":
            ox = int(vx * f) % W
            oy = int(vy * f) % H
            img = big[oy:oy+H, ox:ox+W].copy()
        elif event == "static":
            pass
        frames.append(img)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/synth.mp4")
    ap.add_argument("--labels", default="results/synth_labels.csv")
    ap.add_argument("--hard", action="store_true",
                    help="困难模式: 背景噪声逐帧变化(模拟真实传感器噪声)、"
                         "事件参数随机化(慢镜头/小物体/中途停顿)")
    ap.add_argument("--repeat", type=int, default=1, help="事件计划重复次数(增加段数)")
    args = ap.parse_args()

    global _HARD
    _HARD = args.hard

    base = ["static", "product", "gesture", "camera",
            "product", "camera", "static", "gesture",
            "gesture", "static", "camera", "product"]  # 每类 3 段
    plan = base * args.repeat
    n_per = SEG_DUR * FPS

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
         "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    labels = []
    fi = 0
    for seg_id, event in enumerate(plan):
        for fr in gen_segment(event, n_per):
            proc.stdin.write(fr.tobytes())
            labels.append((fi, event, seg_id))
            fi += 1
    proc.stdin.close()
    proc.wait()

    with open(args.labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "event", "seg_id"])
        w.writerows(labels)
    print(f"[synth] {fi} frames, {len(plan)} segments -> {args.out}, {args.labels}")


if __name__ == "__main__":
    main()
