#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""temporal_pairs.py — "时间维决战"合成事件对生成器。

4 对事件，每对两类在所有单帧统计上尽量一致，只有时间结构不同:
    pair1: osc(往复振荡)      vs ramp(单向锯齿-移动后跳回)
    pair2: cresc(幅度渐强)    vs dim(幅度渐弱)      —— 总运动量相同
    pair3: rhythm(严格周期开关) vs random(随机开关, 占空比相同)
    pair4: ce(先中心后边缘)    vs ec(先边缘后中心)  —— 内部时间顺序相反

渲染: 深灰纹理背景 + 移动方块(中心块或双块)。与 synth_events.py 相同的
编码管线(真实 H.264 libx264)。

用法:
    python temporal_pairs.py --segs-per-class 100
输出 (每对):
    results/tp_<pair>.mp4 + results/tp_<pair>_labels.csv   (分类实验用)
    results/tp_<pair>_demo.mp4                              (2 段演示: A,B 各一)
"""
import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

W, H, FPS = 640, 360, 30
SEG_S = 3
SEG_FR = SEG_S * FPS
rng = np.random.default_rng(11)
_NOISE = rng.normal(0, 5, (H, W))

PAIRS = {
    "pair1": ("osc", "ramp"),
    "pair2": ("cresc", "dim"),
    "pair3": ("rhythm", "random"),
    "pair4": ("ce", "ec"),
}


def bg():
    y, x = np.mgrid[0:H, 0:W]
    img = (np.sin(x / 41) + np.cos(y / 31) + 2) * 40 + _NOISE
    return np.clip(img, 0, 255).astype(np.uint8)


def put(img, cx, cy, r, val=225):
    img[max(0, cy-r):cy+r, max(0, cx-r):cx+r] = val


def block_x(phase, amp):
    """水平位置: 中心±amp*phase, phase∈[-1,1]。"""
    return int(W / 2 + amp * phase)


def gen_segment(event, seed):
    r = rng  # 段内随机性
    n = SEG_FR
    speed = 100.0          # 基准速度 px/s
    amp_px = 110           # 运动幅度 px
    boxr, cy = 22, H // 2
    frames = []
    if event in ("osc", "ramp"):
        # 三角波(osc,周期2T) vs 锯齿波(ramp,周期T/2): 范围同为[-1,1],
        # |v| 同为 4/T 常数, 随机相位 → 位置边缘分布同为均匀。
        # 唯一宏观差异 = 轨迹的时间结构(往返 vs 单向跳回);
        # 微观差异仅 ramp 每段 2 个跳回帧(~2%)。
        T = SEG_S
        phi = r.uniform(0, 1)
        for f in range(n):
            t = f / FPS
            if event == "osc":
                s = (t / T + phi) % 1.0      # 三角周期 T: 斜率 4/T
                ph = 4 * s - 1 if s < 0.5 else 3 - 4 * s
            else:
                s = (t / (T / 2) + phi) % 1.0
                ph = 2 * s - 1
            img = bg()
            put(img, block_x(ph, amp_px), cy, boxr)
            frames.append(img)
    elif event in ("cresc", "dim"):
        # 三角波位置 × 幅度包络: |v(t)| ∝ g(t), cresc 的 g 升 dim 的 g 降,
        # 两者 |v| 多重集合相同(顺序相反)。单帧不可分, 趋势相反。
        per = 18  # 三角波周期(帧)
        f0 = int(r.integers(0, per))  # 随机相位
        for f in range(n):
            g = f / n if event == "cresc" else 1 - f / n
            tri = 2 * (((f + f0) % per) / per) - 1      # ∈[-1,1] 锯齿位置
            ph = g * tri
            img = bg()
            put(img, block_x(ph, amp_px), cy, boxr)
            frames.append(img)
    elif event in ("rhythm", "random"):
        if event == "rhythm":
            on = [(f // 5) % 2 == 0 for f in range(n)]  # 动5帧停5帧
        else:
            on = list(r.random(n) < 0.5)                 # 同占空比随机
        x = W / 2 - amp_px
        v = 2 * speed / FPS * 2   # on 时匀速右移, off 静止; 速度补偿占空比
        for f in range(n):
            img = bg()
            put(img, int(x), cy, boxr)
            if on[f]:
                x += v
                if x > W / 2 + amp_px:
                    x = W / 2 - amp_px
            frames.append(img)
    elif event in ("ce", "ec"):
        r2 = 18
        cx_c, cy_c = W // 2, H // 2
        cx_e, cy_e = W // 6, H // 6
        for f in range(n):
            half = f < n // 2
            # ce: 前半中心动; ec: 前半边缘动。动的块做小幅往复
            ph = np.sin(2 * np.pi * (f % 30) / 30)
            move_c = (event == "ce") == half
            img = bg()
            if move_c:
                put(img, cx_c + int(60 * ph), cy_c, boxr)
                put(img, cx_e, cy_e, r2, 160)
            else:
                put(img, cx_c, cy_c, boxr)
                put(img, cx_e + int(60 * ph), cy_e, r2, 160)
            frames.append(img)
    return frames


def encode(frames_iter, out):
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
         "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = 0
    for fr in frames_iter:
        proc.stdin.write(fr.tobytes())
        n += 1
    proc.stdin.close()
    proc.wait()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs-per-class", type=int, default=100)
    args = ap.parse_args()
    R = Path("results")

    for pair, (ea, eb) in PAIRS.items():
        labels = []
        fi = 0

        def seg_stream():
            nonlocal fi
            for ev in [ea, eb]:
                for k in range(args.segs_per_class):
                    for fr in gen_segment(ev, k):
                        labels.append((fi, ev, fi // SEG_FR))
                        fi += 1
                        yield fr

        n = encode(seg_stream(), str(R / f"tp_{pair}.mp4"))
        with open(R / f"tp_{pair}_labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "event", "seg_id"])
            w.writerows(labels)
        # 演示视频: 每类一段
        demo_frames = list(gen_segment(ea, 0)) + list(gen_segment(eb, 0))
        encode(iter(demo_frames), str(R / f"tp_{pair}_demo.mp4"))
        print(f"[{pair}] {n} frames, {2*args.segs_per_class} segs + demo")


if __name__ == "__main__":
    main()
