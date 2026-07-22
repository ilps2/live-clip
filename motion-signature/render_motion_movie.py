#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""render_motion_movie.py — 把块级运动场序列渲染成"运动动画"。

猜想: 每秒 1 帧的块级运动场连起来播放，本身就是一部低分辨率动画，
人眼可以从中读出事件的类型、起止和发展过程。

两种"动画语言"对照:
    quiver  : 网格上画箭头（方向=MV方向, 长度+颜色=幅度），静止块不画
    heatmap : 块亮度=运动幅度的热度图

用法:
    python render_motion_movie.py --input video.mp4 --name synth \
        --fps-sample 1 [--fps-sample 30 --max-seconds 5]
输出:
    results/movie_<name>_quiver_<fps>fps.mp4/.gif
    results/movie_<name>_heatmap_<fps>fps.mp4/.gif
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from daimon_runtime import setup_plot

from extract_mv import extract

setup_plot()


def sample_indices(n_frames: int, src_fps: float, fps_sample: float,
                   max_seconds: float | None):
    step = max(1, round(src_fps / fps_sample))
    idx = np.arange(0, n_frames, step)
    if max_seconds:
        idx = idx[idx / src_fps <= max_seconds]
    return idx, src_fps / step  # 实际播放帧率


def render_quiver(mag, dx, dy, ax, title):
    gh, gw = mag.shape
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, gw); ax.set_ylim(gh, 0)
    ax.set_xticks(np.arange(0, gw + 1, 4)); ax.set_yticks(np.arange(0, gh + 1, 3))
    ax.grid(color="#333355", lw=0.4, alpha=0.6)
    ys, xs = np.mgrid[0:gh, 0:gw]
    thr = 0.8  # 静止块阈值(px)，不画
    m = mag
    moving = m > thr
    scale = np.percentile(m[moving], 98) + 1e-6 if moving.any() else 1.0
    md = np.minimum(m, scale)
    ax.quiver(xs[moving] + 0.5, ys[moving] + 0.5,
              dx[moving] * md[moving] / scale, dy[moving] * md[moving] / scale,
              md[moving], cmap="turbo", clim=[0, scale],
              angles="xy", scale_units="xy", scale=0.9, width=0.005)
    ax.set_title(title, fontsize=10, color="white")


def render_heatmap(mag, ax, title, vmax=None):
    gh, gw = mag.shape
    ax.set_xlim(0, gw); ax.set_ylim(gh, 0)
    ax.set_xticks(np.arange(0, gw + 1, 4)); ax.set_yticks(np.arange(0, gh + 1, 3))
    ax.grid(color="#333355", lw=0.4, alpha=0.6)
    v = vmax if vmax else np.percentile(mag, 98) + 1e-6
    ax.imshow(mag, extent=[0, gw, gh, 0], cmap="inferno", vmin=0, vmax=v,
              aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=10, color="white")


def render(data, idx, play_fps, mode, out_prefix, src_fps, label=None):
    tmp = Path(out_prefix).parent / (Path(out_prefix).stem + "_frames")
    tmp.mkdir(exist_ok=True)
    vmax = np.percentile(data["mag"], 99) + 1e-6
    pngs = []
    for k, fi in enumerate(idx):
        fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        t = fi / src_fps
        title = f"{label or ''} 运动动画 · {mode} · t={t:05.1f}s"
        if mode == "quiver":
            render_quiver(data["mag"][fi], data["dirx"][fi], data["diry"][fi],
                          ax, title)
        else:
            render_heatmap(data["mag"][fi], ax, title, vmax)
        fig.tight_layout()
        p = tmp / f"f{k:05d}.png"
        fig.savefig(p, facecolor=fig.get_facecolor())
        plt.close(fig)
        pngs.append(p)
    # ffmpeg 合成 mp4 + gif
    import subprocess
    mp4 = f"{out_prefix}.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(play_fps), "-i",
                    str(tmp / "f%05d.png"), "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", mp4],
                   capture_output=True, check=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(play_fps), "-i",
                    str(tmp / "f%05d.png"), "-vf",
                    "fps=%d,scale=640:-1:flags=lanczos,split[s0][s1];"
                    "[s0]palettegen[p];[s1][p]paletteuse" % int(play_fps),
                    f"{out_prefix}.gif"],
                   capture_output=True, check=True)
    for p in pngs:
        p.unlink()
    tmp.rmdir()
    print(f"[movie] {mp4} / {out_prefix}.gif ({len(pngs)} frames @ {play_fps:.0f}fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="视频文件")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fps-sample", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--src-fps", type=float, default=30.0)
    args = ap.parse_args()

    npz = Path("results") / f"_tmp_mv_{args.name}.npz"
    if not npz.exists():
        data = extract(args.input, progress=500)
        np.savez_compressed(npz, **data)
    else:
        data = dict(np.load(npz, allow_pickle=True))

    idx, play_fps = sample_indices(len(data["mag"]), args.src_fps,
                                   args.fps_sample, args.max_seconds)
    for mode in ["quiver", "heatmap"]:
        render(data, idx, max(play_fps, 1), mode,
               f"results/movie_{args.name}_{mode}_{args.fps_sample:g}fps",
               args.src_fps, args.name)


if __name__ == "__main__":
    main()
