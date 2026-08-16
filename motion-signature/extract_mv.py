#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""extract_mv.py — 从视频码流直接导出编码器运动矢量（PyAV, export_mvs）。

原理: H.264/H.265 编码时每个宏块/块已算好运动矢量(MV)，解码端设置
flags2=+export_mvs 后，ffmpeg 把 MV 作为 side_data 挂在帧上导出——
成本接近零（只解码，不做任何运动估计计算）。

帧类型取舍:
    - I 帧(关键帧): 无 MV（帧内编码），本工具输出零运动场并在 meta 标记。
    - P 帧: 单向 MV（前向参考）。
    - B 帧: 双向 MV（source=-1 前向 / +1 后向 / 0 双向）。
      简单起见：同一宏块两向都有时取幅度较大者；所有 MV 幅度取
      |motion_x, motion_y| / motion_scale（编码器用亚像素精度，
      motion_scale 通常是 4 即 1/4 像素单位）。

网格化: 视频是 720p(1280x720)，宏块网格约 80x45。统一重采样到
GRID_W x GRID_H（默认 48x27，即每格 16x16 像素块约 2 个宏块），
每格取覆盖块 MV 的幅度加权平均。

用法:
    python extract_mv.py --input video.flv --out results/mv.npz [--target-fps 3]

输出 npz:
    mag:  (T, GRID_H, GRID_W) float32 块级运动幅度
    dirx, diry: (T, GRID_H, GRID_W) float32 块级运动方向分量（已归一到幅度）
    pts:  (T,) 每帧时间戳(秒)
    pict: (T,) 帧类型 ('I','P','B')
    frames: 另行保存的代表性视频帧 JPEG 由 visualize 脚本负责
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np

GRID_W, GRID_H = 48, 27


def frame_to_grid(mvs, width, height, gw=GRID_W, gh=GRID_H):
    """把该帧的 MV 列表网格化为 (mag, dirx, diry)。"""
    sum_x = np.zeros((gh, gw), np.float64)
    sum_y = np.zeros((gh, gw), np.float64)
    cnt = np.zeros((gh, gw), np.float64)
    for mv in mvs:
        scale = mv.motion_scale if mv.motion_scale else 4
        vx = mv.motion_x / scale
        vy = mv.motion_y / scale
        # 块中心像素坐标（dst 为当前帧块位置）
        cx = (mv.dst_x + mv.w / 2) / width * gw
        cy = (mv.dst_y + mv.h / 2) / height * gh
        gx, gy = int(cx), int(cy)
        if 0 <= gx < gw and 0 <= gy < gh:
            sum_x[gy, gx] += vx
            sum_y[gy, gx] += vy
            cnt[gy, gx] += 1
    cnt[cnt == 0] = 1
    vx = sum_x / cnt
    vy = sum_y / cnt
    mag = np.hypot(vx, vy)
    safe = np.where(mag > 1e-6, mag, 1.0)
    return mag.astype(np.float32), (vx / safe).astype(np.float32), \
           (vy / safe).astype(np.float32)


def _source_fps(stream):
    """从 PyAV stream 获取源帧率。"""
    rate = stream.average_rate
    if rate is not None and rate > 0:
        return float(rate)
    rate = stream.guessed_rate
    if rate is not None and rate > 0:
        return float(rate)
    return 30.0  # fallback


def extract(video_path: str, target_fps: float = 3.0,
            gw=GRID_W, gh=GRID_H, progress=200):
    """均匀采样提取运动矢量，覆盖整个视频时长。
    
    Args:
        target_fps: 目标采样帧率。根据源帧率计算采样间隔，
                   确保均匀覆盖整个视频，而非只取前 N 帧。
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    W, H = stream.codec_context.width, stream.codec_context.height
    src_fps = _source_fps(stream)
    frame_interval = max(1, round(src_fps / target_fps))

    mags, dirxs, dirys, pts_list, picts = [], [], [], [], []
    pict_names = {1: "I", 2: "P", 3: "B"}
    frame_idx = 0
    processed = 0
    
    for frame in container.decode(stream):
        if frame_idx % frame_interval == 0:
            sd = frame.side_data.get("MOTION_VECTORS")
            if sd is None or len(sd) == 0:  # I 帧或无 MV
                mag = np.zeros((gh, gw), np.float32)
                dx = dy = np.zeros((gh, gw), np.float32)
            else:
                mag, dx, dy = frame_to_grid(sd, W, H, gw, gh)
            mags.append(mag)
            dirxs.append(dx)
            dirys.append(dy)
            pts_list.append(float(frame.pts * stream.time_base) if frame.pts is not None else frame_idx / src_fps)
            picts.append(pict_names.get(int(frame.pict_type), "?"))
            processed += 1
            if progress and processed % progress == 0:
                print(f"[mv] {processed} frames (sampled)", flush=True)
        frame_idx += 1
    
    container.close()
    
    # 兼容性：保留 max_frames 语义但不再截断
    return {
        "mag": np.stack(mags), "dirx": np.stack(dirxs), "diry": np.stack(dirys),
        "pts": np.array(pts_list), "pict": np.array(picts),
        "meta": {"width": W, "height": H, "grid_w": gw, "grid_h": gh,
                 "source": video_path, "method": "pyav export_mvs (encoder MVs)",
                 "source_fps": src_fps, "target_fps": target_fps,
                 "frame_interval": frame_interval},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results/mv.npz")
    ap.add_argument("--target-fps", type=float, default=3.0,
                    help="目标采样帧率（均匀采样覆盖整个视频，默认 3fps）")
    ap.add_argument("--grid", default=f"{GRID_W}x{GRID_H}")
    args = ap.parse_args()
    gw, gh = map(int, args.grid.split("x"))

    data = extract(args.input, target_fps=args.target_fps, gw=gw, gh=gh)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(data["meta"], ensure_ascii=False, indent=2))
    n_p = int((data["pict"] == "P").sum())
    n_b = int((data["pict"] == "B").sum())
    n_i = int((data["pict"] == "I").sum())
    pts_range = float(data["pts"].max() - data["pts"].min())
    print(f"[mv] done: {len(data['pts'])} frames (I={n_i} P={n_p} B={n_b}) "
          f"pts=[0..{pts_range:.0f}s] grid={gw}x{gh} -> {out}")


if __name__ == "__main__":
    main()
