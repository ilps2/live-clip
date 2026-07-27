#!/usr/bin/env python3
"""
avis — AVIS (Audio-Visual Intelligent Sampling) CLI

一次编码，多方复用：把视频的廉价信号（运动矢量、ASR、场景分类）
提取一次存入标准化 AVIS 目录，后续批量剪辑/查询/导出都基于同一份数据。

Usage:
    avis encode <video.mp4>              # 将视频编码为 AVIS 格式
    avis encode <video.mp4> -o <dir>     # 指定输出目录
    avis batch <input_dir>               # 批量编码目录下所有视频
    avis clip <avis_dir>                 # 从 AVIS 数据生成切片
    avis clip <avis_dir> --no-sticker    # 纯切片不加贴纸
    avis info <avis_dir>                 # 查看 AVIS 数据摘要
    avis export <avis_dir>               # 导出切片到指定目录

AVIS 目录结构:
    <name>_avis/
    ├── avis.json          # 清单：元数据、信号、切片信息
    ├── mv.npz             # 运动矢量（L3 信号）
    ├── transcript.jsonl   # ASR 转录（L1 信号）
    ├── scenes.csv         # 逐帧场景分类
    ├── timeline.csv       # 逐秒融合评分
    └── clips/             # 生成的切片
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "motion-signature"))  # for fast_features import by scene_classifier

# ── Helpers ────────────────────────────────────────────────────────

def run(cmd, desc="", timeout=300, cwd=None, shell=False):
    """Run a command, print status, return CompletedProcess."""
    label = f"  [{desc}]" if desc else ""
    print(label, flush=True)
    kwargs = dict(timeout=timeout, cwd=cwd or str(BASE))
    if shell:
        kwargs["shell"] = True
        r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if r.returncode != 0 and r.stderr.strip():
        print(f"  STDERR: {r.stderr.strip()[-300:]}", flush=True)
    return r


def probe(video_path: Path) -> dict:
    """Probe video metadata via ffprobe."""
    r = run(["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(video_path)])
    if r.returncode != 0:
        raise SystemExit(f"ffprobe failed on {video_path}")
    info = json.loads(r.stdout)
    dur = float(info["format"]["duration"])
    vstreams = [s for s in info["streams"] if s["codec_type"] == "video"]
    if not vstreams:
        raise SystemExit(f"No video stream in {video_path}")
    vs = vstreams[0]
    w, h = vs["width"], vs["height"]
    fps_str = vs.get("r_frame_rate", "30/1")
    fps = eval(fps_str) if "/" in fps_str else float(fps_str)
    codec = vs.get("codec_name", "unknown")
    orient = "vertical" if h > w else "horizontal"
    fmt = "launch" if orient == "horizontal" else "livestream"
    return {
        "path": str(video_path.resolve()),
        "name": video_path.name,
        "stem": video_path.stem,
        "duration": dur,
        "width": w, "height": h,
        "fps": fps, "codec": codec,
        "orientation": orient,
        "format": fmt,
    }


def auto_fps(source_fps: float, duration_sec: float) -> int:
    """Auto-select optimal MV sampling FPS based on video length."""
    if duration_sec < 120:       return min(int(source_fps), 10)
    elif duration_sec < 480:     return min(int(source_fps), 8)
    elif duration_sec < 1200:    return 5
    else:                         return 3


# ── CLIP Semantic Indexing ─────────────────────────────────────────

def extract_clip_embeddings(video_path: Path, avis_dir: Path,
                             clip_every: int = 5, max_frames: int = 500):
    """Extract CLIP ViT-B-32 embeddings from keyframes for visual search.
    
    Returns (embeddings_np, frame_timestamps) or (None, None) on failure.
    """
    import numpy as np
    import tempfile
    
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        from PIL import Image
    except ImportError:
        print("  ⚠ CLIP requires: pip install transformers torch pillow")
        return None, None
    
    clip_path = avis_dir / "clip.npz"
    meta_path = avis_dir / "clip_meta.json"
    
    if clip_path.exists() and meta_path.exists():
        print("  Using existing CLIP embeddings")
        emb = np.load(clip_path)["embeddings"]
        with open(meta_path) as f:
            meta = json.load(f)
        return emb, meta["timestamps"]
    
    print(f"\n[CLIP] Extracting ViT-B-32 embeddings (every {clip_every}s)...")
    
    # Set HF mirror for China users, unset proxy
    for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
        os.environ.pop(var, None)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    
    # Load model (cpu, local cache first)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    model.eval()
    
    # Probe duration
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True, text=True)
    dur = float(json.loads(r.stdout)["format"]["duration"])
    
    # Extract keyframes
    timestamps = list(range(0, min(int(dur), max_frames * clip_every), clip_every))
    if len(timestamps) > max_frames:
        timestamps = timestamps[:max_frames]
    
    embeddings = []
    batch_size = 16
    tmpdir = Path(tempfile.mkdtemp(prefix="avis_clip_"))
    
    try:
        for batch_start in range(0, len(timestamps), batch_size):
            batch_ts = timestamps[batch_start:batch_start + batch_size]
            frames = []
            
            for ts in batch_ts:
                frame_path = tmpdir / f"frame_{ts:06d}.jpg"
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
                    batch_emb = model.get_image_features(**inputs)
                    if hasattr(batch_emb, 'pooler_output'):
                        batch_emb = batch_emb.pooler_output
                    elif hasattr(batch_emb, 'image_embeds'):
                        batch_emb = batch_emb.image_embeds
                    batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
                embeddings.extend(batch_emb.numpy().tolist())
            
            print(f"  [{batch_start + 1}-{min(batch_start + batch_size, len(timestamps))}/{len(timestamps)}]")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    emb_arr = np.array(embeddings, dtype=np.float32)
    np.savez_compressed(clip_path, embeddings=emb_arr)
    
    with open(meta_path, "w") as f:
        json.dump({"timestamps": timestamps, "model": "ViT-B-32"}, f)
    
    print(f"  ✅ {len(timestamps)} frames, {emb_arr.shape}")
    return emb_arr, timestamps


def search_avis(avis_dir: Path, query: str, top_k: int = 5):
    """Search AVIS directory for frames matching text query via CLIP embeddings."""
    import numpy as np
    
    clip_path = avis_dir / "clip.npz"
    meta_path = avis_dir / "clip_meta.json"
    
    if not clip_path.exists():
        raise SystemExit(f"No CLIP embeddings in {avis_dir}. Encode with --clip first.")
    
    emb = np.load(clip_path)["embeddings"]
    with open(meta_path) as f:
        meta = json.load(f)
    timestamps = meta["timestamps"]
    
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
    except ImportError:
        raise SystemExit("CLIP requires: pip install transformers torch pillow")
    
    print(f"🔍 CLIP search: \"{query}\"")
    
    # Unset proxy, use local cache
    for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
        os.environ.pop(var, None)
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    model.eval()
    
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
        if hasattr(text_emb, 'pooler_output'):
            text_emb = text_emb.pooler_output
        elif hasattr(text_emb, 'text_embeds'):
            text_emb = text_emb.text_embeds
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    
    # Cosine similarity
    similarities = np.dot(emb, text_emb.numpy().T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    for rank, idx in enumerate(top_indices):
        ts = timestamps[idx]
        m, s = divmod(int(ts), 60)
        bar = "█" * int(similarities[idx] * 50)
        print(f"  #{rank+1} [{m}:{s:02d}] score={similarities[idx]:.3f} {bar}")
    
    return [(timestamps[idx], float(similarities[idx])) for idx in top_indices]


# ── Curate: download + encode + package for distribution ──────────

def curate_video(url: str, output_dir: Path = None, asr_model: str = "tiny",
                  max_duration: int = 3600):
    """Download a video from YouTube/B站, encode to AVIS, package for sharing.
    
    Returns dict with paths and copy text, or None on failure.
    """
    from datetime import datetime
    import shutil as _shutil
    import glob as _glob
    import re as _re
    
    # Detect platform
    is_bilibili = "bilibili.com" in url or "b23.tv" in url
    is_youtube = "youtube.com" in url or "youtu.be" in url
    
    if not (is_bilibili or is_youtube):
        print("ERROR: only YouTube and B站 URLs are supported")
        return None
    
    platform = "bilibili" if is_bilibili else "youtube"
    print(f"📥 Downloading from {platform}...")
    
    # Download with yt-dlp
    if output_dir is None:
        output_dir = Path.cwd() / "avis_curated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct yt-dlp command
    dl_dir = output_dir / "_dl"
    dl_dir.mkdir(exist_ok=True)
    
    # Find yt-dlp: prefer /usr/local/bin/python3 on macOS
    yt_python = os.environ.get("YT_PYTHON", "")
    if not yt_python:
        for candidate in ["/usr/local/bin/python3", "python3"]:
            r = subprocess.run([candidate, "-c", "import yt_dlp; print('ok')"],
                              capture_output=True, text=True, timeout=10)
            if r.stdout.strip() == "ok":
                yt_python = candidate
                break
    
    yt_cmd = [yt_python, "-m", "yt_dlp"]
    if is_bilibili:
        yt_cmd += ["--cookies-from-browser", "chrome"]
    else:
        yt_cmd += ["--cookies-from-browser", "chrome", "--remote-components", "ejs:github"]
    
    yt_cmd += [
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--embed-metadata", "--merge-output-format", "mp4",
        "-o", str(dl_dir / "%(title)s.%(ext)s"),
        "--max-filesize", "3G",
        url
    ]
    
    print(f"  Running: {' '.join(yt_cmd[:5])}...")
    r = subprocess.run(yt_cmd, capture_output=True, text=True, timeout=600, cwd=str(dl_dir))
    
    if r.returncode != 0:
        print(f"  Download failed: {r.stderr[-300:]}")
        return None
    
    # Find downloaded file
    mp4s = sorted(dl_dir.glob("*.mp4"))
    if not mp4s:
        print("  No MP4 found after download")
        return None
    
    video_path = mp4s[0]
    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Downloaded: {video_path.name} ({size_mb:.0f}MB)")
    
    # Check duration
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True, text=True)
    dur = float(json.loads(r.stdout)["format"]["duration"])
    print(f"  Duration: {dur:.0f}s ({dur/60:.1f}min)")
    
    if dur > max_duration:
        print(f"  ⚠ Video too long ({dur/60:.0f}min > {max_duration/60:.0f}min max)")
        print(f"  Consider using a shorter clip or raising --max-duration")
        _shutil.rmtree(dl_dir, ignore_errors=True)
        return None
    
    # Encode to AVIS
    slug = _re.sub(r'[^\w\-]', '_', video_path.stem)[:80]
    avis_dir = output_dir / f"{slug}_avis"
    
    print(f"\n🔧 Encoding to AVIS...")
    result = encode_video(video_path, output_dir=output_dir, asr_model=asr_model,
                         fps_target="auto", use_clip=True)
    
    if result is None:
        _shutil.rmtree(dl_dir, ignore_errors=True)
        return None
    
    # Package
    pkg_dir = output_dir / slug
    pkg_dir.mkdir(exist_ok=True)
    
    # Copy AVIS data (excluding clips/)
    for item in result.iterdir():
        if item.name == "clips":
            continue
        dest = pkg_dir / item.name
        if item.is_dir():
            if not dest.exists():
                _shutil.copytree(item, dest)
        else:
            _shutil.copy2(item, dest)
    
    # Copy source video
    _shutil.copy2(video_path, pkg_dir / video_path.name)
    
    # Cleanup download dir
    _shutil.rmtree(dl_dir, ignore_errors=True)
    
    pkg_size = sum(f.stat().st_size for f in pkg_dir.rglob("*") if f.is_file()) / (1024 * 1024)
    
    # Calculate traditional token estimate
    # ~300 image tokens per frame at 1fps for 720p video
    trad_tokens = int(dur) * 300
    
    # Calculate AVIS token estimate from transcript
    avis_tokens = 0
    transcript_path = pkg_dir / "transcript.jsonl"
    if transcript_path.exists():
        with open(transcript_path) as f:
            avis_tokens = sum(len(l) for l in f) // 2  # rough: 2 chars ~ 1 token
    
    # Minimum avis_tokens for display (CLIP + manifest overhead)
    if avis_tokens < 5000:
        avis_tokens = 5000 + avis_tokens
    
    savings = trad_tokens // max(avis_tokens, 1)
    
    # Generate copy
    title = video_path.stem
    mins = int(dur / 60)
    secs = int(dur % 60)
    
    copy_text = f"""🎬 AVIS Encoded: {title}

📊 {mins}m{secs}s | AVIS package: {pkg_size:.0f}MB

Token comparison (per AI query):
  传统逐帧理解: {trad_tokens:,} tokens (~¥{trad_tokens/1000*0.01:.1f})
  AVIS 编码后:   {avis_tokens:,} tokens (~¥{avis_tokens/1000*0.002:.4f})
  节省:          {savings}×

一次编码，所有模型复用。直接下载 AVIS 数据丢给 LLM，不重新解码像素。

📦 AVIS Package: github.com/ilps2/avis
🔗 Source: {url}
"""
    
    # Write README
    readme_path = pkg_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# AVIS: {title}\n\n")
        f.write(f"- **Duration:** {mins}m{secs}s\n")
        f.write(f"- **Package:** {pkg_size:.0f}MB\n")
        f.write(f"- **Source:** {url}\n\n")
        f.write(f"## Usage\n\n")
        f.write(f"```bash\n")
        f.write(f"# Query the video with any LLM\n")
        f.write(f"cat transcript.jsonl | head -100  # preview transcript\n")
        f.write(f"avis info .                        # data summary\n")
        f.write(f"avis search . \"your query\"         # CLIP visual search\n")
        f.write(f"```\n\n")
        f.write(f"## Token Comparison\n\n")
        f.write(f"| Method | Tokens | Cost (DeepSeek) |\n")
        f.write(f"|--------|--------|-----------------|\n")
        f.write(f"| Traditional (GPT-4o, 1fps) | {trad_tokens:,} | ~¥{trad_tokens/1000*0.01:.1f} |\n")
        f.write(f"| AVIS (features only) | {avis_tokens:,} | ~¥{avis_tokens/1000*0.002:.4f} |\n")
        f.write(f"| **Savings** | **{savings}×** | — |\n\n")
        f.write(f"## x.com Post\n\n```\n{copy_text}```\n")
    
    print(f"\n{'═' * 55}")
    print(f"✅ AVIS Package: {pkg_dir}/")
    print(f"   Size: {pkg_size:.0f}MB")
    print(f"   README: {readme_path.name}")
    print(f"{'═' * 55}")
    print(f"\n📱 X.com copy:\n")
    print(copy_text)
    
    return {
        "dir": pkg_dir,
        "title": title,
        "token_savings": trad_tokens // max(avis_tokens, 1),
        "copy": copy_text,
    }


# ── Encode: extract all signals into AVIS format ──────────────────

def encode_video(video_path: Path, output_dir: Path = None,
                 asr_model: str = "base", fps_target: str = "auto",
                 skip_mv: bool = False, skip_asr: bool = False,
                 use_clip: bool = False,
                 keep_intermediate: bool = True):
    """
    Encode a single video into the AVIS format.
    Steps: probe → MV extraction → ASR → scene classification → scoring → manifest.
    """
    video = video_path.resolve()
    if not video.exists():
        print(f"ERROR: {video} not found")
        return None

    # Determine output directory
    if output_dir is None:
        avis_dir = video.parent / f"{video.stem}_avis"
    else:
        avis_dir = Path(output_dir) / f"{video.stem}_avis"
    avis_dir.mkdir(parents=True, exist_ok=True)

    print(f"╔{'═' * 55}")
    print(f"║ AVIS encode: {video.name}")
    print(f"║ Output:      {avis_dir}")
    print(f"╚{'═' * 55}")

    # --- Step 1: Probe ---
    print("\n[1/6] Probing video...")
    meta = probe(video)
    dur = meta["duration"]
    w, h, fps = meta["width"], meta["height"], meta["fps"]
    orient, fmt = meta["orientation"], meta["format"]
    print(f"  {dur:.0f}s ({dur/60:.1f}min) | {w}×{h} {orient} | {meta['codec']} {fps:.0f}fps | {fmt}")

    if fps_target == "auto":
        ft = auto_fps(fps, dur)
        print(f"  MV target: {ft}fps (auto-selected)")
    else:
        ft = int(fps_target)
        print(f"  MV target: {ft}fps (manual)")

    # --- Step 2: MV extraction ---
    mv_path = avis_dir / "mv.npz"
    has_mv = False
    print("\n[2/6] Extracting motion vectors...")
    if skip_mv and mv_path.exists():
        print(f"  Using existing {mv_path}")
        has_mv = True
    else:
        max_frames = int(dur * ft * 1.5)
        r = run([sys.executable, str(BASE / "motion-signature" / "extract_mv.py"),
                 "--input", str(video), "--out", str(mv_path),
                 "--max-frames", str(max_frames)],
                desc="MV extraction", timeout=600)
        has_mv = (r.returncode == 0 and mv_path.exists())
        if not has_mv:
            print("  ⚠ MV extraction failed — continuing in audio-only mode")

    # --- Step 3: ASR ---
    transcript_path = avis_dir / "transcript.jsonl"
    print("\n[3/6] Transcribing audio...")
    if skip_asr and transcript_path.exists():
        print(f"  Using existing {transcript_path}")
    else:
        r = run([sys.executable, str(BASE / "livestream-highlight" / "asr.py"),
                 "--video", str(video), "--out", str(transcript_path),
                 "--model", asr_model], desc="ASR", timeout=900)

    segs = []
    n_segs = 0
    if transcript_path.exists():
        segs = [json.loads(l) for l in open(transcript_path)]
        n_segs = len(segs)
    print(f"  {n_segs} segments")

    # --- Step 4: Scene classification ---
    scenes_path = avis_dir / "scenes.csv"
    n_boundaries = 0
    print("\n[4/6] Classifying scenes...")
    if has_mv:
        from scene_classifier import SceneClassifier
        sc = SceneClassifier(str(mv_path))
        sc.classify(orientation=orient)
        scene_w = sc.weights(format_type=fmt)
        # Trim/pad
        if len(scene_w) > int(dur) + 1:
            scene_w = scene_w[:int(dur) + 1]
        elif len(scene_w) < int(dur) + 1:
            import numpy as np
            pad = np.ones(int(dur) + 1 - len(scene_w)) * 0.8
            scene_w = np.concatenate([scene_w, pad])
        vbound = set(sc.boundaries(min_gap=1))
        n_boundaries = len(vbound)
        # Save scenes CSV
        ss = sc.per_second()
        ss.to_csv(scenes_path, header=["scene"])
        print(f"  Scenes: {dict(ss.value_counts())} | {n_boundaries} boundaries")
    else:
        import numpy as np
        scene_w = np.ones(int(dur) + 1) * 0.8
        vbound = set()
        print("  No MV — using uniform weights")

    # --- Step 5: Scoring ---
    timeline_path = avis_dir / "timeline.csv"
    print("\n[5/6] Computing timeline scores...")
    import numpy as np
    import pandas as pd

    # Soft promo patterns
    SOFT = [
        ("拍下", 0.7), ("下单", 0.8), ("价格", 0.5), ("优惠", 0.6),
        ("福利", 0.6), ("券", 0.5), ("放心拍", 0.7), ("去下单", 0.8),
        ("链接", 0.8), ("尺码", 0.3), ("颜色", 0.3), ("首发", 0.6),
        ("上市", 0.5), ("限时", 0.6), ("元", 0.3),
    ]
    promo_events = []
    for s in segs:
        w_text = sum(wt for pat, wt in SOFT if pat in s["text"])
        if w_text > 0:
            promo_events.append({"t": (s["start"] + s["end"]) / 2, "weight": min(w_text, 2.5)})

    # Voice activity
    va = np.zeros(int(dur) + 2)
    for s in segs:
        t0 = int(s["start"]); t1 = min(int(s["end"]) + 1, len(va))
        va[t0:t1] = 1

    secs = np.arange(0, int(dur) + 1)
    promo_s = np.zeros(len(secs))
    for e in promo_events:
        promo_s[int(e["t"])] += e["weight"]

    def zscore(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

    promo_z = zscore(pd.Series(promo_s).rolling(15, center=True, min_periods=1).sum().to_numpy())
    va_s = va[:len(secs)]

    if fmt == "launch":
        sp_dense = pd.Series(va_s).rolling(20, center=True, min_periods=1).mean().to_numpy()
        score = scene_w[:len(secs)] * (1.0 + sp_dense)
        threshold = float(np.percentile(score[score > 0.3], 40)) if (score > 0.3).any() else 0.5
    else:
        score = scene_w[:len(secs)] * (1.0 + 1.5 * np.maximum(promo_z, 0) + 0.5 * va_s)
        threshold = 1.0

    # Save timeline
    tl_df = pd.DataFrame({
        "second": secs,
        "score": score,
        "promo_z": promo_z[:len(secs)],
    })
    tl_df.to_csv(timeline_path, index=False)

    # Find peaks
    def find_peaks(sig, dist=60, min_h=None):
        if min_h is None: min_h = threshold
        pk = []
        for i in range(1, len(sig) - 1):
            if sig[i] > min_h and sig[i] > sig[i-1] and sig[i] >= sig[i+1]:
                pk.append(i)
        out = []
        for p in pk:
            if not out or p - out[-1] >= dist:
                out.append(p)
        return out

    peaks = find_peaks(score, dist=max(60, int(dur / 10)))
    peaks_sorted = sorted([(score[p], p) for p in peaks], reverse=True)
    print(f"  Peaks: {len(peaks_sorted)} | Threshold: {threshold:.2f}")

    # --- Step 6: Write manifest ---
    print("\n[6/6] Writing AVIS manifest...")
    
    # --- CLIP semantic indexing (optional) ---
    has_clip = False
    if use_clip:
        emb, clip_ts = extract_clip_embeddings(video, avis_dir)
        has_clip = emb is not None
    
    manifest = {
        "avis_version": "0.1.0",
        "video": meta,
        "signals": {
            "mv": {
                "path": "mv.npz",
                "available": has_mv,
                "fps_target": ft,
            },
            "asr": {
                "path": "transcript.jsonl",
                "segments": n_segs,
                "model": asr_model,
            },
            "scenes": {
                "path": "scenes.csv",
                "boundaries": n_boundaries,
            },
            "clip": {
                "path": "clip.npz",
                "available": has_clip,
            },
        },
        "timeline": {
            "path": "timeline.csv",
            "peaks": len(peaks_sorted),
            "duration_sec": dur,
        },
        "clips": [],  # populated by 'avis clip'
    }

    manifest_path = avis_dir / "avis.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ AVIS encoded: {avis_dir}/")
    bits = [f"MV: {'✅' if has_mv else '❌'}", f"ASR: {n_segs} segs", f"Scenes: {n_boundaries} boundaries"]
    if has_clip: bits.append("CLIP: ✅")
    print(f"     {' | '.join(bits)}")
    return avis_dir


# ── Clip: generate clips from AVIS data ────────────────────────────

def clip_from_avis(avis_dir: Path, max_clips: int = 3,
                   min_dur: int = 30, max_dur: int = 60,
                   sticker_overlay: bool = True, sticker_lines: str = ""):
    """Generate clips from pre-computed AVIS data."""
    avis_dir = avis_dir.resolve()
    manifest_path = avis_dir / "avis.json"
    if not manifest_path.exists():
        raise SystemExit(f"No avis.json found in {avis_dir}")

    with open(manifest_path) as f:
        m = json.load(f)

    video_path = Path(m["video"]["path"])
    if not video_path.exists():
        # Try to find video next to avis dir
        alt = avis_dir.parent / m["video"]["name"]
        if alt.exists():
            video_path = alt
        else:
            raise SystemExit(f"Video not found: {m['video']['path']}")

    meta = m["video"]
    dur, w, h = meta["duration"], meta["width"], meta["height"]
    orient, fmt = meta["orientation"], meta["format"]

    clips_dir = avis_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    print(f"╔{'═' * 55}")
    print(f"║ AVIS clip: {avis_dir.name}")
    print(f"║ Video:     {video_path.name}")
    print(f"║ Format:    {fmt} | {w}×{h} {orient}")
    print(f"╚{'═' * 55}")

    # Load signals
    import numpy as np
    import pandas as pd

    has_mv = m["signals"]["mv"]["available"]
    mv_path = avis_dir / "mv.npz"
    transcript_path = avis_dir / "transcript.jsonl"
    timeline_path = avis_dir / "timeline.csv"

    segs = [json.loads(l) for l in open(transcript_path)] if transcript_path.exists() else []
    tl = pd.read_csv(timeline_path)
    score = tl["score"].to_numpy()

    # Visual boundaries
    vbound = set()
    if has_mv:
        from scene_classifier import SceneClassifier
        sc = SceneClassifier(str(mv_path))
        sc.classify(orientation=orient)
        vbound = set(sc.boundaries(min_gap=1))

    # --- Scoring ---
    threshold = 1.0
    if fmt == "launch":
        threshold = float(np.percentile(score[score > 0.3], 40)) if (score > 0.3).any() else 0.5

    def find_peaks(sig, dist=60, min_h=None):
        if min_h is None: min_h = threshold
        pk = []
        for i in range(1, len(sig) - 1):
            if sig[i] > min_h and sig[i] > sig[i-1] and sig[i] >= sig[i+1]:
                pk.append(i)
        out = []
        for p in pk:
            if not out or p - out[-1] >= dist:
                out.append(p)
        return out

    peaks = find_peaks(score, dist=max(60, int(dur / 10)))
    peaks_sorted = sorted([(score[p], p) for p in peaks], reverse=True)

    # --- Clip generation ---
    def snap_visual(t, pref="nearest", max_d=8):
        if not vbound: return t
        best, best_d = t, max_d + 1
        for b in vbound:
            d = abs(b - t)
            if d < best_d and d <= max_d:
                if pref == "start" and b <= t: d *= 0.7
                elif pref == "end" and b >= t: d *= 0.7
                if d < best_d: best, best_d = b, d
        return best

    def sentence_end(t, max_d=4):
        best, best_d = t, max_d + 1
        for s in segs:
            for cand in [s["start"], s["end"]]:
                d = abs(cand - t)
                if d < best_d and d <= max_d:
                    endings = ["。」","！","？","了","的","吧","呢","吗","行","带","看","拍","单"]
                    if any(s["text"].strip().endswith(p) for p in endings):
                        d *= 0.5
                    if d < best_d: best, best_d = cand, d
        return best

    def has_price(t0, t1):
        return any(
            any(kw in s["text"] for kw in ["价格","元","多少钱","优惠","下单","拍","链接","定价","售价","券","福利","拍下","放心拍","去下单","去拍"])
            for s in segs if t0 <= s["start"] <= t1
        )

    def has_name(t0, t1):
        intro_kw = ["就是","介绍","这款","新品","新成员","上市","首发",
                    "K90","Max","Pro","我们家","今天给","给大家","笼重"]
        item_kw = ["颜色","衣服","裤子","款","面料","尺码","材质","版型",
                   "这个","这款","我们家","成分","效果","看一下","展示"]
        return any(
            any(kw in s["text"] for kw in intro_kw + item_kw)
            for s in segs if t0 <= s["start"] <= t1
        )

    selected = []
    for sc_val, pk in peaks_sorted:
        if len(selected) >= max_clips: break
        half = max_dur // 2
        t0 = max(0, pk - half)
        t1 = min(int(dur), pk + half)
        t0 = snap_visual(t0, "start")
        t1 = snap_visual(t1, "end")
        t0 = sentence_end(t0, 4)
        t1 = sentence_end(t1, 4)
        if t1 - t0 < min_dur:
            c = (t0 + t1) / 2
            t0 = max(0, c - min_dur / 2)
            t1 = min(int(dur), c + min_dur / 2)
        if t1 - t0 > max_dur + 10:
            t1 = t0 + max_dur
        overlap = any(not (t1 <= s["t0"] or t0 >= s["t1"]) for s in selected)
        if overlap: continue
        selected.append({"t0": int(t0), "t1": int(t1), "score": round(sc_val, 2)})

    # Price coverage
    has_any_price = any(has_price(s["t0"], s["t1"]) for s in selected)
    if not has_any_price:
        for seg in segs:
            if any(kw in seg["text"] for kw in ["价格","元","多少钱","定价","售价"]):
                pt = seg["start"]
                if not any(s["t0"] <= pt <= s["t1"] for s in selected):
                    t0 = max(0, pt - 15)
                    t1 = min(int(dur), pt + 15)
                    t0 = snap_visual(t0, "start")
                    t1 = snap_visual(t1, "end")
                    selected.append({"t0": int(t0), "t1": int(t1), "score": 0, "price_forced": True})
                    break

    selected.sort(key=lambda x: x["t0"])

    # Name in first clip
    if selected and not has_name(selected[0]["t0"], selected[0]["t0"] + 15):
        for s in reversed(segs):
            if s["start"] < selected[0]["t0"] and any(
                kw in s["text"] for kw in ["就是","介绍","这款","新品","新成员"]
            ):
                selected[0]["t0"] = max(s["start"] - 1, selected[0]["t0"] - 10)
                break

    # --- Sticker overlay setup ---
    sticker_top = sticker_bot = None
    bar_h_top = bar_h_bot = 0
    all_sticker_variants = {}

    if sticker_overlay and has_mv and orient == "vertical":
        from content_region import content_region
        tfrac, bfrac, _ = content_region(str(mv_path))
        bar_h_top = int(h * tfrac)
        bar_h_bot = int(h * (1 - bfrac))
        print(f"  Sticker bars: top {tfrac*100:.0f}% ({bar_h_top}px), bot {(1-bfrac)*100:.0f}% ({bar_h_bot}px)")

        from collections import Counter
        if sticker_lines:
            parts = sticker_lines.split("|")
            top_lines = parts[:2] if len(parts) >= 2 else [parts[0], ""]
            bot_lines = parts[2:] if len(parts) > 2 else []
            all_sticker_variants["manual"] = (top_lines, bot_lines)
        else:
            # AI extraction: audience segments
            AUDIENCE_SEGMENTS = {
                "price": {
                    "name": "价格敏感",
                    "top_kw": ["划算","便宜","性价比","优惠","福利","限时","秒杀","补贴","赠品","送","平均才","不到"],
                    "bot_kw": ["试用","包退","运费险","放心拍","售后","正品","保质期"],
                },
                "quality": {
                    "name": "品质导向",
                    "top_kw": ["高级","质感","专柜","明星","贵妇","成分","精华","科技","专利","进口","限量"],
                    "bot_kw": ["持妆","服帖","轻薄","养肤","不卡粉","细腻","光泽","哑光"],
                },
                "effect": {
                    "name": "效果焦虑",
                    "top_kw": ["遮瑕","显白","毛孔","痘印","斑点","暗沉","细纹","松弛","出油","脱妆"],
                    "bot_kw": ["前后对比","肉眼可见","立竿见影","回购","复购","空瓶","自用"],
                },
            }
            segment_scores = {}
            for seg_id, seg_data in AUDIENCE_SEGMENTS.items():
                score_s = 0
                for kw in seg_data["top_kw"] + seg_data["bot_kw"]:
                    for s in segs:
                        if kw in s["text"]: score_s += 1
                segment_scores[seg_id] = score_s
            active_segments = sorted(segment_scores, key=segment_scores.get, reverse=True)[:3]

            for seg_id in active_segments:
                seg_d = AUDIENCE_SEGMENTS[seg_id]
                top_hits = [kw for kw in seg_d["top_kw"] if any(kw in s["text"] for s in segs)]
                bot_hits = [kw for kw in seg_d["bot_kw"] if any(kw in s["text"] for s in segs)]
                top_lines = [
                    " · ".join(top_hits[:3]) if len(top_hits) >= 2 else (top_hits[0] if top_hits else "好物推荐"),
                    f"「{seg_d['name']}」看这里",
                ]
                bot_lines = bot_hits[:4] if bot_hits else ["限时优惠"]
                all_sticker_variants[seg_id] = (top_lines, bot_lines)

            if not active_segments:
                kw_counter = Counter()
                for seg in segs:
                    for kw in ["元","优惠","福利","限时","便宜","送","赠","试用","包退","遮瑕","持妆","保湿"]:
                        if kw in seg["text"]: kw_counter[kw] += 1
                top_kw = [w for w, _ in kw_counter.most_common(3)]
                all_sticker_variants["default"] = (
                    [" · ".join(top_kw) if top_kw else "好物推荐", "到手好价"],
                    ["限时优惠"]
                )

        # Render sticker PNGs
        t2p = str(BASE / "text2png_bin")
        if not Path(t2p).exists():
            subprocess.run(["swiftc", str(BASE / "text2png.swift"), "-o", t2p], check=True)

        all_sticker_pngs = {}
        for seg_id, (top_lines, bot_lines) in all_sticker_variants.items():
            top_pngs = []
            for i, line in enumerate(top_lines):
                p = str(clips_dir / f"_sticker_{seg_id}_top_{i}.png")
                subprocess.run([t2p, line, "32" if i == 0 else "36", p,
                                "gold" if i == 0 else "white"], check=True)
                top_pngs.append(p)
            bot_pngs = []
            for i, line in enumerate(bot_lines):
                p = str(clips_dir / f"_sticker_{seg_id}_bot_{i}.png")
                subprocess.run([t2p, line, "30", p,
                                "gold" if i == len(bot_lines) - 1 else "white"], check=True)
                bot_pngs.append(p)

            # Composite top bar
            bar_path = str(clips_dir / "_tmp_bar.png")
            subprocess.check_call(
                f"ffmpeg -y -v error -f lavfi -i color=c=black:s={w}x{bar_h_top}:d=0.1 "
                f"-frames:v 1 -c:v png '{bar_path}'", shell=True)
            prev = "0:v"; y = 40
            tmp_prev = ""
            top_final = str(clips_dir / f"_sticker_{seg_id}_top_final.png")
            for idx, p in enumerate(top_pngs):
                tmp = str(clips_dir / f"_tmp_{idx}.png")
                out_label = top_final if idx == len(top_pngs) - 1 else tmp
                subprocess.check_call(
                    f"ffmpeg -y -v error -i '{bar_path if idx == 0 else tmp_prev}' -i '{p}' "
                    f'-filter_complex "[{prev}][1:v]overlay=(W-w)/2:{y}" '
                    f"-c:v png '{out_label}'", shell=True)
                if idx < len(top_pngs) - 1: tmp_prev = tmp
                prev = "0:v"; y += 100

            # Composite bottom bar
            bar_bot_path = str(clips_dir / "_tmp_bar_bot.png")
            subprocess.check_call(
                f"ffmpeg -y -v error -f lavfi -i color=c=black:s={w}x{bar_h_bot}:d=0.1 "
                f"-frames:v 1 -c:v png '{bar_bot_path}'", shell=True)
            prev = "0:v"; y = 30
            tmp_prev = ""
            bot_final = str(clips_dir / f"_sticker_{seg_id}_bot_final.png")
            for idx, p in enumerate(bot_pngs):
                tmp = str(clips_dir / f"_tmp_{idx}_b.png")
                out_label = bot_final if idx == len(bot_pngs) - 1 else tmp
                subprocess.check_call(
                    f"ffmpeg -y -v error -i '{bar_bot_path if idx == 0 else tmp_prev}' -i '{p}' "
                    f'-filter_complex "[{prev}][1:v]overlay=(W-w)/2:{y}" '
                    f"-c:v png '{out_label}'", shell=True)
                if idx < len(bot_pngs) - 1: tmp_prev = tmp
                prev = "0:v"; y += 80

            all_sticker_pngs[seg_id] = (top_final, bot_final)

    # --- Print preview ---
    print(f"\n  {'#':>3s} {'Start':>6s} {'End':>6s} {'Dur':>5s} {'Score':>6s} {'Price':>6s} {'Name':>5s}")
    for i, s in enumerate(selected):
        ph = "✅" if has_price(s["t0"], s["t1"]) else "❌"
        nh = "✅" if has_name(s["t0"], s["t1"]) else "❌"
        print(f"  {i+1:3d} {s['t0']:5d}s {s['t1']:5d}s {s['t1']-s['t0']:4d}s {s['score']:6.2f} {ph:>6s} {nh:>5s}")

    # --- Cut clips ---
    for i, s in enumerate(selected):
        base_name = f"clip_{i+1:02d}_{s['t0']}-{s['t1']}s"

        if all_sticker_pngs:
            for seg_id, (stop, sbot) in all_sticker_pngs.items():
                out = clips_dir / f"{base_name}_{seg_id}.mp4"
                subprocess.check_call(
                    f"ffmpeg -y -v error -ss {s['t0']} -to {s['t1']} "
                    f"-i '{video_path}' -i '{stop}' -i '{sbot}' "
                    f'-filter_complex "[0:v][1:v]overlay=0:0[tmp];[tmp][2:v]overlay=0:{h - bar_h_bot}[outv]" '
                    f'-map "[outv]" -map 0:a:0 '
                    f"-c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k '{out}'",
                    shell=True)
        else:
            out = clips_dir / f"{base_name}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-ss", str(s["t0"]), "-to", str(s["t1"]),
                "-i", str(video_path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                str(out)
            ], check=True)

    # --- Update manifest ---
    m["clips"] = selected
    with open(manifest_path, "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    # --- Cleanup temp PNGs ---
    for p in clips_dir.glob("_tmp*"): p.unlink()
    for p in clips_dir.glob("_sticker_*"): p.unlink()

    # --- Summary ---
    total = sum(s["t1"] - s["t0"] for s in selected)
    print(f"\n{'═' * 55}")
    print(f"  ✅ {len(selected)} clips, {total:.0f}s total")
    print(f"  Output: {clips_dir}/")
    for f in sorted(clips_dir.glob("*.mp4")):
        size = f.stat().st_size / 1024
        print(f"    {f.name} ({size:.0f}KB)")
    print(f"{'═' * 55}")

    return selected


# ── Batch: encode all videos in a directory ────────────────────────

def batch_encode(input_dir: str, output_dir: str = None,
                 asr_model: str = "base", fps_target: str = "auto",
                 skip_existing: bool = True):
    """Encode all videos in a directory into AVIS format."""
    in_path = Path(input_dir)
    if not in_path.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    out_path = Path(output_dir) if output_dir else in_path

    # Find all video files
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ts", ".flv"}
    videos = sorted([f for f in in_path.iterdir()
                     if f.suffix.lower() in video_exts and f.is_file()])

    if not videos:
        print(f"No video files found in {input_dir}")
        return

    print(f"╔{'═' * 55}")
    print(f"║ AVIS batch: {len(videos)} videos in {input_dir}")
    print(f"║ Output:     {out_path}")
    print(f"╚{'═' * 55}")

    results = []
    for i, video in enumerate(videos):
        avis_dir = out_path / f"{video.stem}_avis"
        if skip_existing and avis_dir.exists() and (avis_dir / "avis.json").exists():
            print(f"\n[{i+1}/{len(videos)}] SKIP: {video.name} (already encoded)")
            results.append({"video": video.name, "status": "skipped"})
            continue

        print(f"\n{'─' * 55}")
        print(f"[{i+1}/{len(videos)}] {video.name}")
        try:
            result = encode_video(video, output_dir=out_path,
                                  asr_model=asr_model, fps_target=fps_target,
                                  keep_intermediate=True)
            results.append({"video": video.name, "status": "ok", "dir": str(result)})
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({"video": video.name, "status": "error", "error": str(e)})

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skipped")
    err = sum(1 for r in results if r["status"] == "error")
    print(f"\n{'═' * 55}")
    print(f"  Batch complete: {ok} encoded, {skip} skipped, {err} errors")
    print(f"{'═' * 55}")


# ── Info ───────────────────────────────────────────────────────────

def show_info(avis_dir: str):
    """Display summary of AVIS data."""
    avis_path = Path(avis_dir)
    manifest_path = avis_path / "avis.json"
    if not manifest_path.exists():
        raise SystemExit(f"No avis.json found in {avis_dir}")

    with open(manifest_path) as f:
        m = json.load(f)

    v = m["video"]
    sig = m["signals"]
    tl = m["timeline"]
    clips = m.get("clips", [])

    print(f"╔{'═' * 55}")
    print(f"║ AVIS: {avis_path.name}")
    print(f"╠{'═' * 55}")
    print(f"║ Video:    {v['name']}")
    print(f"║ Duration: {v['duration']:.0f}s ({v['duration']/60:.1f}min)")
    print(f"║ Size:     {v['width']}×{v['height']} @ {v['fps']:.0f}fps")
    print(f"║ Codec:    {v['codec']} | {v['orientation']} | {v['format']}")
    print(f"╠{'═' * 55}")
    print(f"║ Signals:")
    print(f"║   MV:     {'✅' if sig['mv']['available'] else '❌'}  {sig['mv'].get('fps_target','?')}fps target")
    print(f"║   ASR:    {sig['asr']['segments']} segments ({sig['asr']['model']})")
    print(f"║   Scenes: {sig['scenes']['boundaries']} boundaries")
    print(f"╠{'═' * 55}")
    print(f"║ Timeline: {tl['peaks']} peaks over {tl['duration_sec']:.0f}s")
    print(f"║ Clips:    {len(clips)} generated")

    if clips:
        print(f"╠{'═' * 55}")
        print(f"║ {'#':>3s} {'Start':>6s} {'End':>6s} {'Dur':>5s} {'Score':>6s}")
        for i, c in enumerate(clips):
            print(f"║ {i+1:3d} {c['t0']:5d}s {c['t1']:5d}s {c['t1']-c['t0']:4d}s {c['score']:6.2f}")

    # Check clip files
    clips_dir = avis_path / "clips"
    if clips_dir.exists():
        mp4s = sorted(clips_dir.glob("*.mp4"))
        if mp4s:
            total_size = sum(f.stat().st_size for f in mp4s)
            print(f"║ Files:    {len(mp4s)} clips, {total_size/1024/1024:.1f}MB total")

    print(f"╚{'═' * 55}")


# ── Export ─────────────────────────────────────────────────────────

def export_clips(avis_dir: str, output_dir: str):
    """Copy clips from AVIS directory to output directory."""
    avis_path = Path(avis_dir)
    clips_dir = avis_path / "clips"
    if not clips_dir.exists():
        raise SystemExit(f"No clips/ directory in {avis_dir}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(clips_dir.glob("*.mp4"))
    if not mp4s:
        print("No clips to export.")
        return

    print(f"Exporting {len(mp4s)} clips to {out_path}/")
    for f in mp4s:
        dest = out_path / f.name
        shutil.copy2(f, dest)
        print(f"  → {dest.name}")

    print(f"Done: {len(mp4s)} files exported.")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AVIS — Audio-Visual Intelligent Sampling CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  avis encode video.mp4                       # Encode single video
  avis encode video.mp4 -o ./avis_data        # Custom output dir
  avis batch ./videos/                        # Batch encode directory
  avis clip video_avis/                       # Generate clips from AVIS data
  avis clip video_avis/ --no-sticker          # Clips without sticker overlay
  avis info video_avis/                       # Show AVIS summary
  avis export video_avis/ -o ./final_clips/   # Export clips
        """)

    sub = parser.add_subparsers(dest="command", help="Commands")

    # --- encode ---
    enc = sub.add_parser("encode", help="Encode video into AVIS format")
    enc.add_argument("video", help="Input video path")
    enc.add_argument("-o", "--output", help="Output directory (default: next to video)")
    enc.add_argument("--asr-model", default="base",
                     choices=["tiny", "base", "small"], help="ASR model size")
    enc.add_argument("--fps-target", default="auto", help="MV sampling FPS or 'auto'")
    enc.add_argument("--skip-mv", action="store_true", help="Skip MV if already exists")
    enc.add_argument("--skip-asr", action="store_true", help="Skip ASR if already exists")
    enc.add_argument("--clip", action="store_true", help="Extract CLIP semantic embeddings for visual search")

    # --- batch ---
    bat = sub.add_parser("batch", help="Batch encode all videos in directory")
    bat.add_argument("input_dir", help="Directory containing video files")
    bat.add_argument("-o", "--output", help="Output directory (default: same as input)")
    bat.add_argument("--asr-model", default="base", choices=["tiny", "base", "small"])
    bat.add_argument("--fps-target", default="auto")
    bat.add_argument("--force", action="store_true", help="Re-encode even if AVIS dir exists")

    # --- clip ---
    clp = sub.add_parser("clip", help="Generate clips from AVIS data")
    clp.add_argument("avis_dir", help="AVIS directory (contains avis.json)")
    clp.add_argument("--max-clips", type=int, default=3, help="Max clips to generate")
    clp.add_argument("--min-dur", type=int, default=30, help="Min clip duration (seconds)")
    clp.add_argument("--max-dur", type=int, default=60, help="Max clip duration (seconds)")
    clp.add_argument("--no-sticker", action="store_true", help="Disable sticker overlay")
    clp.add_argument("--sticker-lines", default="",
                     help="Manual sticker text: brand|price|line1|line2|...")

    # --- info ---
    inf = sub.add_parser("info", help="Show AVIS data summary")
    inf.add_argument("avis_dir", help="AVIS directory")

    # --- export ---
    exp = sub.add_parser("export", help="Export clips from AVIS data")
    exp.add_argument("avis_dir", help="AVIS directory")
    exp.add_argument("-o", "--output", required=True, help="Output directory for clips")

    # --- search ---
    srch = sub.add_parser("search", help="Search video content via CLIP semantic embeddings")
    srch.add_argument("avis_dir", help="AVIS directory (must have CLIP embeddings)")
    srch.add_argument("query", help="Search query (e.g. 'red sports car', 'person on stage')")
    srch.add_argument("-n", "--top", type=int, default=5, help="Number of results")

    # --- curate ---
    cur = sub.add_parser("curate", help="Download + encode + package a video for sharing")
    cur.add_argument("url", help="YouTube or B站 video URL")
    cur.add_argument("-o", "--output", help="Output directory (default: ./avis_curated)")
    cur.add_argument("--asr-model", default="tiny", choices=["tiny", "base", "small"])
    cur.add_argument("--max-duration", type=int, default=3600,
                     help="Max video duration in seconds (default: 3600 = 1h)")

    args = parser.parse_args()

    if args.command == "encode":
        encode_video(
            Path(args.video),
            output_dir=Path(args.output) if args.output else None,
            asr_model=args.asr_model,
            fps_target=args.fps_target,
            skip_mv=args.skip_mv,
            skip_asr=args.skip_asr,
            use_clip=getattr(args, 'clip', False),
        )

    elif args.command == "batch":
        batch_encode(
            args.input_dir,
            output_dir=args.output,
            asr_model=args.asr_model,
            fps_target=args.fps_target,
            skip_existing=not args.force,
        )

    elif args.command == "clip":
        clip_from_avis(
            Path(args.avis_dir),
            max_clips=args.max_clips,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            sticker_overlay=not args.no_sticker,
            sticker_lines=args.sticker_lines,
        )

    elif args.command == "info":
        show_info(args.avis_dir)

    elif args.command == "export":
        export_clips(args.avis_dir, args.output)

    elif args.command == "search":
        search_avis(Path(args.avis_dir), args.query, top_k=args.top)

    elif args.command == "curate":
        curate_video(
            args.url,
            output_dir=Path(args.output) if args.output else None,
            asr_model=args.asr_model,
            max_duration=args.max_duration,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
