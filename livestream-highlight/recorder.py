#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""recorder.py — 直播流录制。优先 streamlink，退化到 yt-dlp 拉 HLS 存 mp4。

用法:
    python recorder.py --url https://live.bilibili.com/31654967 --duration 600 \
        --out results/live.mp4

注意: 录制的相对时间轴从进程启动算起，与 danmaku.py 的 rel_t 对齐需要
两者同时启动（见 README 的建议流程，或记录各自启动 epoch 做对齐）。
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def record_streamlink(url: str, out: Path, duration: float) -> bool:
    sl = shutil.which("streamlink")
    if not sl:
        return False
    cmd = [sl, url, "best", "-o", str(out), "-f"]
    print("[recorder] streamlink:", " ".join(cmd))
    p = subprocess.Popen(cmd)
    try:
        p.wait(timeout=duration)
    except subprocess.TimeoutExpired:
        p.terminate()
    return out.exists() and out.stat().st_size > 0


def record_ytdlp(url: str, out: Path, duration: float) -> bool:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        return False
    # B 站直播优先选 FLV over HTTP（fmp4/m3u8 会被 yt-dlp 安全策略拒绝）
    cmd = [ytdlp, "-f", "best[ext=flv]/best", "-o", str(out), "--no-part", url]
    print("[recorder] yt-dlp:", " ".join(cmd))
    p = subprocess.Popen(cmd)
    try:
        p.wait(timeout=duration)
    except subprocess.TimeoutExpired:
        p.terminate()
        p.wait(timeout=30)
    # yt-dlp 下载直播流时被 terminate 会保留已下载分片
    return out.exists() and out.stat().st_size > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="直播间 URL")
    ap.add_argument("--duration", type=float, default=600, help="录制秒数")
    ap.add_argument("--out", default="results/live.mp4")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = record_streamlink(args.url, out, args.duration)
    if not ok:
        print("[recorder] streamlink 不可用或失败，降级 yt-dlp")
        ok = record_ytdlp(args.url, out, args.duration)
    if ok:
        print(f"[recorder] done: {out} {out.stat().st_size/1e6:.1f}MB "
              f"in {time.time()-t0:.0f}s")
    else:
        print("[recorder] FAILED: 未能录制到文件")
        sys.exit(1)


if __name__ == "__main__":
    main()
