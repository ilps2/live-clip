#!/usr/bin/env python3
"""
avis-online.py — Stream any video URL to AVIS without keeping source file

Usage:
    python3 avis-online.py "https://b23.tv/xxx" -o ./output
    python3 avis-online.py "https://youtube.com/watch?v=xxx" --asr tiny

Architecture:
    yt-dlp → temp file → [ASR ∥ MV ∥ CLIP] → AVIS dir → delete temp
    All three encoders read from the same temp file in parallel.
    Temp file is deleted immediately after encoding completes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "motion-signature"))

# Prevent Hermes PYTHONPATH contamination
os.environ.pop('PYTHONPATH', None)


def find_ytdlp_python():
    """Find a Python with yt-dlp installed."""
    for candidate in ["/usr/local/bin/python3", "python3"]:
        r = subprocess.run(
            [candidate, "-c", "import yt_dlp; print('ok')"],
            capture_output=True, text=True, timeout=10
        )
        if "ok" in r.stdout:
            return candidate
    return None


def probe_video(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=30
    )
    info = json.loads(r.stdout)
    dur = float(info["format"]["duration"])
    vstreams = [s for s in info["streams"] if s["codec_type"] == "video"]
    if not vstreams:
        raise SystemExit("No video stream")
    vs = vstreams[0]
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "duration": dur,
        "width": vs["width"],
        "height": vs["height"],
        "fps": eval(vs.get("r_frame_rate", "30/1").replace("/", "*1.0/")),
        "codec": vs.get("codec_name", "unknown"),
        "orientation": "vertical" if vs["height"] > vs["width"] else "horizontal",
    }


def run_asr(video_path: Path, out_path: Path, model: str = "tiny"):
    """Run faster-whisper ASR."""
    r = subprocess.run(
        [sys.executable, str(BASE / "livestream-highlight" / "asr.py"),
         "--video", str(video_path), "--out", str(out_path),
         "--model", model],
        capture_output=True, text=True, timeout=1800
    )
    if r.returncode != 0:
        print(f"  ASR failed: {r.stderr[-200:]}")
        return 0
    segs = [json.loads(l) for l in open(out_path)] if out_path.exists() else []
    return len(segs)


def run_mv(video_path: Path, out_path: Path, fps_target: int = 3, duration: float = 0):
    """Extract motion vectors from H.264 bitstream."""
    max_frames = int(duration * fps_target * 1.5) if duration else 5000
    r = subprocess.run(
        [sys.executable, str(BASE / "motion-signature" / "extract_mv.py"),
         "--input", str(video_path), "--out", str(out_path),
         "--max-frames", str(max_frames)],
        capture_output=True, text=True, timeout=600
    )
    return r.returncode == 0 and out_path.exists()


def run_clip(video_path: Path, avis_dir: Path):
    """Extract CLIP embeddings from keyframes."""
    import numpy as np
    from transformers import CLIPModel, CLIPProcessor
    import torch
    from PIL import Image

    # Unset proxy
    for v in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
        os.environ.pop(v, None)

    clip_path = avis_dir / "clip.npz"
    meta_path = avis_dir / "clip_meta.json"

    if clip_path.exists():
        return True

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    model.eval()

    # Probe duration
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True, text=True)
    dur = float(json.loads(r.stdout)["format"]["duration"])

    clip_every = 5
    max_frames = 500
    timestamps = list(range(0, min(int(dur), max_frames * clip_every), clip_every))
    if len(timestamps) > max_frames:
        timestamps = timestamps[:max_frames]

    embeddings = []
    tmpdir = Path(tempfile.mkdtemp(prefix="avis_clip_"))

    try:
        for batch_start in range(0, len(timestamps), 16):
            batch_ts = timestamps[batch_start:batch_start + 16]
            frames = []
            for ts in batch_ts:
                frame_path = tmpdir / f"f_{ts:06d}.jpg"
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-ss", str(ts),
                     "-i", str(video_path), "-vframes", "1", "-q:v", "3",
                     str(frame_path)],
                    capture_output=True)
                if frame_path.exists():
                    frames.append(Image.open(frame_path).convert("RGB"))
            if frames:
                inputs = processor(images=frames, return_tensors="pt", padding=True)
                with torch.no_grad():
                    be = model.get_image_features(**inputs)
                    if hasattr(be, 'pooler_output'): be = be.pooler_output
                    be = be / be.norm(dim=-1, keepdim=True)
                embeddings.extend(be.numpy().tolist())
            print(f"  CLIP [{batch_start+1}-{min(batch_start+16, len(timestamps))}/{len(timestamps)}]",
                  flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    emb_arr = np.array(embeddings, dtype=np.float32)
    np.savez_compressed(clip_path, embeddings=emb_arr)
    with open(meta_path, "w") as f:
        json.dump({"timestamps": timestamps, "model": "ViT-B-32"}, f)

    return True


def run_scene_classifier(mv_path: Path, avis_dir: Path, meta: dict):
    """Classify scenes from motion vectors."""
    import numpy as np
    import pandas as pd

    from scene_classifier import SceneClassifier

    sc = SceneClassifier(str(mv_path))
    sc.classify(orientation=meta["orientation"])
    scene_w = sc.weights(format_type="launch" if meta["orientation"] == "horizontal" else "livestream")

    dur = int(meta["duration"])
    if len(scene_w) > dur + 1:
        scene_w = scene_w[:dur + 1]
    elif len(scene_w) < dur + 1:
        pad = np.ones(dur + 1 - len(scene_w)) * 0.8
        scene_w = np.concatenate([scene_w, pad])

    vbound = set(sc.boundaries(min_gap=1))
    ss = sc.per_second()
    ss.to_csv(avis_dir / "scenes.csv", header=["scene"])

    return len(vbound), scene_w


def write_manifest(avis_dir: Path, meta: dict, n_segs: int, asr_model: str,
                    has_mv: bool, fps_target: int, n_boundaries: int,
                    has_clip: bool, peaks: int):
    manifest = {
        "avis_version": "0.2.0-online",
        "video": meta,
        "signals": {
            "mv": {"path": "mv.npz", "available": has_mv, "fps_target": fps_target},
            "asr": {"path": "transcript.jsonl", "segments": n_segs, "model": asr_model},
            "scenes": {"path": "scenes.csv", "boundaries": n_boundaries},
            "clip": {"path": "clip.npz", "available": has_clip},
        },
        "timeline": {"path": "timeline.csv", "peaks": peaks,
                      "duration_sec": meta["duration"]},
        "clips": [],
    }
    with open(avis_dir / "avis.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def stream_encode(url: str, output_dir: Path, asr_model: str = "tiny",
                   fps_target: int = 3):
    """Stream a URL to AVIS — download to temp, encode, delete source."""

    yt_python = find_ytdlp_python()
    if not yt_python:
        print("ERROR: No Python with yt-dlp found")
        return None

    is_bilibili = "bilibili.com" in url or "b23.tv" in url

    print(f"📥 Streaming from {'B站' if is_bilibili else 'YouTube'}...")

    # Download to temp file
    tmpdir = Path(tempfile.mkdtemp(prefix="avis_stream_"))
    dl_dir = tmpdir / "dl"
    dl_dir.mkdir()

    yt_cmd = [yt_python, "-m", "yt_dlp"]
    if is_bilibili:
        yt_cmd += ["--cookies-from-browser", "chrome"]
    else:
        yt_cmd += ["--cookies-from-browser", "chrome", "--remote-components", "ejs:github"]

    yt_cmd += [
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--embed-metadata", "--merge-output-format", "mp4",
        "-o", str(dl_dir / "%(title)s.%(ext)s"),
        url
    ]

    t0 = time.time()
    r = subprocess.run(yt_cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  Download failed: {r.stderr[-200:]}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    mp4s = sorted(dl_dir.glob("*.mp4"))
    if not mp4s:
        print("  No video found")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    video_path = mp4s[0]
    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ {video_path.name} ({size_mb:.0f}MB) — {time.time() - t0:.0f}s")

    # Ensure H.264 for MV (PyAV doesn't support AV1/VP9)
    from avis import _ensure_h264
    video_path = _ensure_h264(video_path)

    # Probe
    meta = probe_video(video_path)
    dur = meta["duration"]
    print(f"  {dur:.0f}s ({dur/60:.1f}min) | {meta['width']}×{meta['height']} {meta['orientation']}")

    # Setup AVIS dir
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() or c in '-_' else '_' for c in video_path.stem)[:80]
    avis_dir = output_dir / f"{slug}_avis"
    avis_dir.mkdir(parents=True, exist_ok=True)

    # ── Parallel encoding ──
    print(f"\n⚡ Parallel encode: ASR ∥ MV ∥ CLIP")
    t_enc = time.time()

    mv_path = avis_dir / "mv.npz"
    transcript_path = avis_dir / "transcript.jsonl"

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        futures["asr"] = pool.submit(run_asr, video_path, transcript_path, asr_model)
        futures["mv"] = pool.submit(run_mv, video_path, mv_path, fps_target, dur)
        futures["clip"] = pool.submit(run_clip, video_path, avis_dir)

        results = {}
        for name, fut in futures.items():
            try:
                results[name] = fut.result(timeout=1800)
            except Exception as e:
                print(f"  {name.upper()} failed: {e}")
                results[name] = False if name != "asr" else 0

    n_segs = results.get("asr", 0)
    has_mv = results.get("mv", False)
    has_clip = results.get("clip", False)

    # Scene classifier (depends on MV)
    n_boundaries = 0
    scene_w = None
    if has_mv:
        print(f"\n🎬 Scene classification...")
        n_boundaries, scene_w = run_scene_classifier(mv_path, avis_dir, meta)

    # Timeline scoring
    print(f"📊 Timeline scoring...")
    import numpy as np
    import pandas as pd

    segs = [json.loads(l) for l in open(transcript_path)] if transcript_path.exists() else []

    # Simple voice activity + promo scoring
    va = np.zeros(int(dur) + 2)
    for s in segs:
        t0 = int(s["start"]); t1 = min(int(s["end"]) + 1, len(va))
        va[t0:t1] = 1

    secs = np.arange(0, int(dur) + 1)
    score = np.ones(len(secs)) * 0.8  # default uniform
    if has_mv and scene_w is not None and n_boundaries > 0:
        score = scene_w[:len(secs)] * (1.0 + 0.5 * va[:len(secs)])

    tl_df = pd.DataFrame({"second": secs, "score": score})
    tl_df.to_csv(avis_dir / "timeline.csv", index=False)

    # Find peaks
    threshold = 1.0
    pk = []
    for i in range(1, len(score) - 1):
        if score[i] > threshold and score[i] > score[i-1] and score[i] >= score[i+1]:
            if not pk or i - pk[-1] >= 60:
                pk.append(i)

    # Write manifest
    write_manifest(avis_dir, meta, n_segs, asr_model, has_mv, fps_target,
                    n_boundaries, has_clip, len(pk))

    # Delete temp source file
    shutil.rmtree(tmpdir, ignore_errors=True)

    enc_time = time.time() - t_enc
    total_time = time.time() - t0

    print(f"\n{'═' * 55}")
    print(f"✅ AVIS streamed: {avis_dir}/")
    print(f"   Download: {total_time - enc_time:.0f}s | Encode: {enc_time:.0f}s | Total: {total_time:.0f}s")
    print(f"   MV: {'✅' if has_mv else '❌'} | ASR: {n_segs} segs |"
          f" Scenes: {n_boundaries} | CLIP: {'✅' if has_clip else '❌'}")
    print(f"   Source deleted — only AVIS data kept")
    print(f"{'═' * 55}")

    return avis_dir


def main():
    p = argparse.ArgumentParser(description="Stream video URL → AVIS (no source file kept)")
    p.add_argument("url", help="Video URL (YouTube, B站, etc.)")
    p.add_argument("-o", "--output", default="./avis_streamed",
                    help="Output directory for AVIS data")
    p.add_argument("--asr-model", default="tiny", choices=["tiny", "base", "small"])
    p.add_argument("--fps", type=int, default=3, help="MV sampling FPS")
    args = p.parse_args()

    result = stream_encode(
        args.url,
        Path(args.output),
        asr_model=args.asr_model,
        fps_target=args.fps,
    )

    if result:
        print(f"\n📦 AVIS data: {result}/")
        print(f"   Ready for: avis info/search/clip/v6_triple_cut --avis")


if __name__ == "__main__":
    main()
