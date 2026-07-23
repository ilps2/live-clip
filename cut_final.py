#!/usr/bin/env python3
"""
cut_final.py — One-click livestream clipping pipeline.

Usage:
    python cut_final.py <video.mp4> [--format livestream|launch] [--max-clips 5] [--min-dur 30] [--max-dur 60]

Output:
    <video_dir>/clips_out/     Final clips ready to post
    <video_dir>/pipeline/      Intermediate data (MV, ASR, scores)
"""
import argparse, json, subprocess, sys, os, shutil
from pathlib import Path
import numpy as np, pandas as pd

# ---- Paths ----
BASE = Path(__file__).resolve().parent  # motion-signature/
sys.path.insert(0, str(BASE))
from scene_classifier import SceneClassifier

def run(cmd, desc="", timeout=300, cwd=None):
    print(f"  [{desc}]", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd or str(BASE))
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-300:]}", flush=True)
    return r

def auto_fps(source_fps, duration_sec):
    """根据视频时长和源帧率自动选择最优采样帧率。
    策略：短视频保持高采样精度，长视频降低采样率以控制总帧数。
    经测试：10fps 对短视频已足够（与 15fps 场景分类结果一致），
           长视频 5fps 能在精度和性能间取得平衡。
    """
    if duration_sec < 120:       # < 2min: 短视频
        return min(source_fps, 10)
    elif duration_sec < 480:     # 2-8min: 中视频
        return min(source_fps, 8)
    elif duration_sec < 1200:    # 8-20min: 长视频
        return 5
    else:                        # > 20min: 超长视频
        return 3

def main():
    p = argparse.ArgumentParser(description="One-click livestream clip generator")
    p.add_argument("video", help="Input video path")
    p.add_argument("--format", default="auto", choices=["auto","livestream","launch"])
    p.add_argument("--max-clips", type=int, default=3, help="Max clips to output")
    p.add_argument("--min-dur", type=int, default=30, help="Min clip duration (s)")
    p.add_argument("--max-dur", type=int, default=60, help="Max clip duration (s)")
    p.add_argument("--asr-model", default="base", choices=["tiny","base","small"])
    p.add_argument("--skip-mv", action="store_true", help="Skip MV extraction if already done")
    p.add_argument("--skip-asr", action="store_true", help="Skip ASR if already done")
    p.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    p.add_argument("--fps-target", default="auto",
                   help="Target FPS for MV extraction. 'auto' selects based on duration, or specify a number")
    p.add_argument("--auto-crop", action="store_true", help="Auto-crop to content region (screen-recorded livestreams)")
    p.add_argument("--sticker-overlay", action="store_true", help="Overlay gradient stickers instead of cropping")
    p.add_argument("--sticker-lines", default="", help="Custom sticker text, comma-separated (top:brand|price, bot:line1|line2|line3)")
    args = p.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        print(f"ERROR: {video} not found"); sys.exit(1)

    # Output directories
    out_dir = video.parent / f"{video.stem}_clips"
    pipe_dir = video.parent / f"{video.stem}_pipeline"
    out_dir.mkdir(exist_ok=True)
    pipe_dir.mkdir(exist_ok=True)

    print(f"═" * 50)
    print(f"cut_final: {video.name}")
    print(f"Output: {out_dir}/")
    print(f"═" * 50)

    # ============================================================
    # Step 1: Probe video
    # ============================================================
    print("\n[1/6] Probing video...")
    r = run(["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(video)])
    info = json.loads(r.stdout)
    dur = float(info["format"]["duration"])
    vstream = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    w, h = vstream["width"], vstream["height"]
    orient = "vertical" if h > w else "horizontal"
    codec = vstream["codec_name"]
    fps_str = vstream.get("r_frame_rate", "30/1")
    fps = eval(fps_str) if "/" in fps_str else float(fps_str)

    if args.format == "auto":
        fmt = "launch" if orient == "horizontal" else "livestream"
    else:
        fmt = args.format

    print(f"  Duration: {dur:.0f}s ({dur/60:.1f}min) | {w}x{h} {orient} | {codec} {fps:.0f}fps | Format: {fmt}")

    # Resolve fps-target
    if args.fps_target == "auto":
        fps_target = auto_fps(fps, dur)
        print(f"  Auto FPS: {fps_target}fps (source {fps:.0f}fps, {dur:.0f}s)")
    else:
        fps_target = int(args.fps_target)
        print(f"  Manual FPS: {fps_target}fps")

    # ============================================================
    # Step 2: MV extraction
    # ============================================================
    print("\n[2/6] Extracting motion vectors...")
    mv_path = pipe_dir / "mv.npz"
    if args.skip_mv and mv_path.exists():
        print(f"  Using existing {mv_path}")
    else:
        max_frames = int(dur * fps_target * 1.5)
        print(f"  MV target={fps_target}fps (source {fps:.0f}fps)")
        r = run([sys.executable, str(BASE / "extract_mv.py"),
                 "--input", str(video), "--out", str(mv_path),
                 "--fps-target", str(fps_target), "--max-frames", str(max_frames)], desc="MV extract", timeout=600)
        if r.returncode != 0:
            print("  MV extraction failed — falling back to audio-only mode")

    # ============================================================
    # Step 3: ASR
    # ============================================================
    print("\n[3/6] Transcribing audio...")
    transcript_path = pipe_dir / "transcript.jsonl"
    if args.skip_asr and transcript_path.exists():
        print(f"  Using existing {transcript_path}")
    else:
        r = run([sys.executable, str(BASE / "real_fusion/asr.py"),
                 "--video", str(video), "--out", str(transcript_path),
                 "--model", args.asr_model], desc="ASR", timeout=900)

    segs = [json.loads(l) for l in open(transcript_path)] if transcript_path.exists() else []
    print(f"  {len(segs)} segments")

    # ============================================================
    # Step 4: Scene classification
    # ============================================================
    print("\n[4/6] Classifying scenes...")
    has_mv = mv_path.exists()
    if has_mv:
        sc = SceneClassifier(str(mv_path))
        sc.classify(orientation=orient)
        scene_w = sc.weights(format_type=fmt)
        # Trim/pad to full video duration
        if len(scene_w) > int(dur) + 1:
            scene_w = scene_w[:int(dur) + 1]
        elif len(scene_w) < int(dur) + 1:
            pad = np.ones(int(dur) + 1 - len(scene_w)) * 0.8
            scene_w = np.concatenate([scene_w, pad])
        vbound = set(sc.boundaries(min_gap=1))
        print(f"  Scenes: {dict(sc.per_second().value_counts())} | {len(vbound)} boundaries")
    else:
        scene_w = np.ones(int(dur) + 1) * 0.8
        vbound = set()
        print("  No MV — using uniform weights")

    # ---- Sticker overlay (AI extraction or manual input) ----
    sticker_top = sticker_bot = None
    sticker_filter = ""
    all_sticker_variants = {}
    crop_filter = ""
    if args.sticker_overlay and has_mv and orient == "vertical":
        # 1. Detect overlay region from MV motion heatmap
        from content_region import content_region
        tfrac, bfrac, _ = content_region(str(mv_path))
        bar_h_top = int(h * tfrac)
        bar_h_bot = int(h * (1 - bfrac))
        print(f"  Sticker bars: top {tfrac*100:.0f}% ({bar_h_top}px), bot {(1-bfrac)*100:.0f}% ({bar_h_bot}px)")

        # 2. Determine sticker text: manual input or AI extraction
        all_sticker_sets = {}
        if args.sticker_lines:
            # Manual mode: comma-separated "top_brand|top_price|bot_line1|bot_line2|..."
            parts = args.sticker_lines.split("|")
            top_lines = parts[:2] if len(parts) >= 2 else [parts[0], ""]
            bot_lines = parts[2:] if len(parts) > 2 else []
            all_sticker_sets["manual"] = (top_lines, bot_lines)
            mode = "manual"
        else:
            # AI extraction mode: audience-segmented keyword matching
            from collections import Counter
            
            # Audience segments — each has keyword sets for different sticker layers
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

            # Score each segment against ASR in this video
            segment_scores = {}
            for seg_id, seg_data in AUDIENCE_SEGMENTS.items():
                score = 0
                for kw in seg_data["top_kw"] + seg_data["bot_kw"]:
                    for s in segs:
                        if kw in s["text"]:
                            score += 1
                segment_scores[seg_id] = score

            # Pick top 3 matching segments (all three: price, quality, effect)
            active_segments = sorted(segment_scores, key=segment_scores.get, reverse=True)[:3]
            
            # Generate sticker text per segment
            for seg_id in active_segments:
                seg = AUDIENCE_SEGMENTS[seg_id]
                # Top lines: brand intro + segment-specific hook
                top_hits = [kw for kw in seg["top_kw"] if any(kw in s["text"] for s in segs)]
                bot_hits = [kw for kw in seg["bot_kw"] if any(kw in s["text"] for s in segs)]
                
                top_lines = [
                    " · ".join(top_hits[:3]) if len(top_hits) >= 2 else (top_hits[0] if top_hits else "好物推荐"),
                    f"「{seg['name']}」看这里",
                ]
                bot_lines = bot_hits[:4] if bot_hits else ["限时优惠"]
                all_sticker_sets[seg_id] = (top_lines, bot_lines)
            
            # Fallback: if no segment matched, use top keywords
            if not active_segments:
                kw_counter = Counter()
                for seg in segs:
                    for kw in ["元","优惠","福利","限时","便宜","送","赠","试用","包退","遮瑕","持妆","保湿"]:
                        if kw in seg["text"]:
                            kw_counter[kw] += 1
                top_kw = [w for w, _ in kw_counter.most_common(3)]
                all_sticker_sets["default"] = (
                    [" · ".join(top_kw) if top_kw else "好物推荐", "到手好价"],
                    ["限时优惠"]
                )

            mode = f"audience-split ({len(active_segments)} segments: {', '.join(active_segments)})"

        print(f"  Sticker text ({mode})")

        # 3. Render all sticker variants (one per audience segment)
        t2p = str(BASE / "text2png_bin")
        if not Path(t2p).exists():
            subprocess.run(["swiftc", str(BASE / "text2png.swift"), "-o", t2p], check=True)

        def composite_side(bar_h, pngs, out_path, y_start=30, y_step=80):
            bar_path = str(out_dir / "_tmp_bar.png")
            subprocess.check_call(
                f"ffmpeg -y -v error -f lavfi -i color=c=black:s={w}x{bar_h}:d=0.1 "
                f"-frames:v 1 -c:v png '{bar_path}'", shell=True)
            prev = "0:v"; y = y_start
            for idx, p in enumerate(pngs):
                tmp = str(out_dir / f"_tmp_{idx}.png")
                out_label = out_path if idx == len(pngs) - 1 else tmp
                subprocess.check_call(
                    f"ffmpeg -y -v error -i '{bar_path if idx == 0 else tmp_prev}' -i '{p}' "
                    f'-filter_complex "[{prev}][1:v]overlay=(W-w)/2:{y}" '
                    f"-c:v png '{out_label}'", shell=True)
                if idx < len(pngs) - 1: tmp_prev = tmp
                prev = "0:v"; y += y_step

        # Generate sticker sets for each audience segment
        all_sticker_variants = {}  # {seg_id: (top_png_path, bot_png_path)}
        for seg_id, (top_lines, bot_lines) in all_sticker_sets.items():
            top_pngs = []
            for i, line in enumerate(top_lines):
                p = str(out_dir / f"_sticker_{seg_id}_top_{i}.png")
                subprocess.run([t2p, line, "32" if i == 0 else "36", p, "gold" if i == 0 else "white"], check=True)
                top_pngs.append(p)
            bot_pngs = []
            for i, line in enumerate(bot_lines):
                p = str(out_dir / f"_sticker_{seg_id}_bot_{i}.png")
                subprocess.run([t2p, line, "30", p, "gold" if i == len(bot_lines) - 1 else "white"], check=True)
                bot_pngs.append(p)

            sticker_top = str(out_dir / f"_sticker_{seg_id}_top_final.png")
            sticker_bot = str(out_dir / f"_sticker_{seg_id}_bot_final.png")
            composite_side(bar_h_top, top_pngs, sticker_top, y_start=40, y_step=100)
            composite_side(bar_h_bot, bot_pngs, sticker_bot, y_start=30, y_step=80)
            all_sticker_variants[seg_id] = (sticker_top, sticker_bot)
            print(f"    {seg_id}: top={top_lines}, bot={bot_lines[:3]}")

    elif args.auto_crop and has_mv:
        from content_region import content_region
        top_frac, bot_frac, _ = content_region(str(mv_path))
        crop_h = int(h * (bot_frac - top_frac))
        crop_y = int(h * top_frac)
        crop_filter = f"crop={w}:{crop_h}:0:{crop_y}"
        print(f"  Auto-crop: {top_frac*100:.0f}%–{bot_frac*100:.0f}% → {w}x{crop_h}")

    # ============================================================
    # Step 5: Promo detection + scoring
    # ============================================================
    print("\n[5/6] Detecting highlights...")

    # Soft promo patterns
    SOFT = [
        ("拍下", 0.7), ("下单", 0.8), ("价格", 0.5), ("优惠", 0.6),
        ("福利", 0.6), ("券", 0.5), ("放心拍", 0.7), ("去下单", 0.8),
        ("链接", 0.8), ("尺码", 0.3), ("颜色", 0.3), ("首发", 0.6),
        ("上市", 0.5), ("限时", 0.6), ("元", 0.3),
    ]

    promo = []
    for s in segs:
        w_text = sum(wt for pat, wt in SOFT if pat in s["text"])
        if w_text > 0:
            promo.append({"t": (s["start"] + s["end"]) / 2, "weight": min(w_text, 2.5)})
    print(f"  Promo events: {len(promo)}")

    # Voice activity
    va = np.zeros(int(dur) + 2)
    for s in segs:
        for sec in range(int(s["start"]), min(int(s["end"]) + 1, len(va))):
            va[sec] = 1

    # Per-second scoring
    secs = np.arange(0, int(dur) + 1)
    promo_s = np.zeros(len(secs))
    for e in promo:
        promo_s[int(e["t"])] += e["weight"]

    def zscore(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

    promo_z = zscore(pd.Series(promo_s).rolling(15, center=True, min_periods=1).sum().to_numpy())
    va_s = va[:len(secs)]

    if fmt == "launch":
        # Launch: scene_weight × speech density (no promo)
        sp_dense = pd.Series(va_s).rolling(20, center=True, min_periods=1).mean().to_numpy()
        score = scene_w[:len(secs)] * (1.0 + sp_dense)
        threshold = np.percentile(score[score > 0.3], 40) if (score > 0.3).any() else 0.5
    else:
        # Livestream: scene_weight × (1 + promo + voice)
        score = scene_w[:len(secs)] * (1.0 + 1.5 * np.maximum(promo_z, 0) + 0.5 * va_s)
        threshold = 1.0

    # ---- Find peaks ----
    def find_peaks(sig, dist=60, min_h=None):
        if min_h is None:
            min_h = threshold
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

    # ============================================================
    # Step 6: Clip generation with visual boundaries
    # ============================================================
    print(f"\n[6/6] Generating clips...")

    def snap_visual(t, pref="nearest", max_d=8):
        if not vbound:
            return t
        best, best_d = t, max_d + 1
        for b in vbound:
            d = abs(b - t)
            if d < best_d and d <= max_d:
                if pref == "start" and b <= t: d *= 0.7
                elif pref == "end" and b >= t: d *= 0.7
                if d < best_d:
                    best, best_d = b, d
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
                    if d < best_d:
                        best, best_d = cand, d
        return best

    def has_price(t0, t1):
        return any(
            any(kw in s["text"] for kw in ["价格","元","多少钱","优惠","下单","拍","链接","定价","售价","券","福利","拍下","放心拍","去下单","去拍"])
            for s in segs if t0 <= s["start"] <= t1
        )

    def has_name(t0, t1):
        # Product intro keywords + fallback: any item-describing word in first clip
        intro_kw = ["就是","介绍","这款","新品","新成员","上市","首发",
                    "K90","Max","Pro","我们家","今天给","给大家","笼重"]
        item_kw = ["颜色","衣服","裤子","款","面料","尺码","材质","版型",
                   "这个","这款","我们家","成分","效果","看一下","展示"]
        return any(
            any(kw in s["text"] for kw in intro_kw + item_kw)
            for s in segs if t0 <= s["start"] <= t1
        )

    selected = []
    total_dur = 0
    max_total = args.max_clips * args.max_dur

    for sc_val, pk in peaks_sorted:
        if len(selected) >= args.max_clips:
            break
        if total_dur >= max_total:
            break

        half = args.max_dur // 2
        t0 = max(0, pk - half)
        t1 = min(int(dur), pk + half)

        # Snap to visual boundaries
        t0 = snap_visual(t0, "start")
        t1 = snap_visual(t1, "end")

        # Refine with speech
        t0 = sentence_end(t0, 4)
        t1 = sentence_end(t1, 4)

        # Enforce duration
        if t1 - t0 < args.min_dur:
            c = (t0 + t1) / 2
            t0 = max(0, c - args.min_dur / 2)
            t1 = min(int(dur), c + args.min_dur / 2)
        if t1 - t0 > args.max_dur + 10:
            t1 = t0 + args.max_dur

        # Avoid overlap
        overlap = any(
            not (t1 <= s["t0"] or t0 >= s["t1"]) for s in selected
        )
        if overlap:
            continue

        selected.append({"t0": int(t0), "t1": int(t1), "score": round(sc_val, 2)})
        total_dur += t1 - t0

    # ---- Post-processing: ensure price coverage ----
    has_any_price = any(has_price(s["t0"], s["t1"]) for s in selected)
    if not has_any_price:
        # Add a price segment if found elsewhere
        for seg in segs:
            if any(kw in seg["text"] for kw in ["价格","元","多少钱","定价","售价"]):
                pt = seg["start"]
                # Check no overlap with existing
                if not any(s["t0"] <= pt <= s["t1"] for s in selected):
                    # Add a short price clip
                    t0 = max(0, pt - 15)
                    t1 = min(int(dur), pt + 15)
                    t0 = snap_visual(t0, "start")
                    t1 = snap_visual(t1, "end")
                    selected.append({"t0": int(t0), "t1": int(t1), "score": 0, "price_forced": True})
                    break

    selected.sort(key=lambda x: x["t0"])

    # ---- Ensure name in first clip ----
    if selected and not has_name(selected[0]["t0"], selected[0]["t0"] + 15):
        # Search backwards for product name intro
        for s in reversed(segs):
            if s["start"] < selected[0]["t0"] and any(
                kw in s["text"] for kw in ["就是","介绍","这款","新品","新成员"]
            ):
                selected[0]["t0"] = max(s["start"] - 1, selected[0]["t0"] - 10)
                break

    # ---- Print preview ----
    print(f"\n  {'#':>3s} {'Start':>6s} {'End':>6s} {'Dur':>5s} {'Score':>6s} {'Price':>6s} {'Name':>5s}")
    for i, s in enumerate(selected):
        ph = "✅" if has_price(s["t0"], s["t1"]) else "❌"
        nh = "✅" if has_name(s["t0"], s["t1"]) else "❌"
        print(f"  {i+1:3d} {s['t0']:5d}s {s['t1']:5d}s {s['t1']-s['t0']:4d}s {s['score']:6.2f} {ph:>6s} {nh:>5s}")

    # ---- Cut + optional sticker overlay (audience variants) ----
    for i, s in enumerate(selected):
        base_name = f"clip_{i+1:02d}_{s['t0']}-{s['t1']}s"

        if args.sticker_overlay and all_sticker_variants:
            # Output one clip per audience segment
            for seg_id, (stop, sbot) in all_sticker_variants.items():
                out = out_dir / f"{base_name}_{seg_id}.mp4"
                subprocess.check_call(
                    f"ffmpeg -y -v error -ss {s['t0']} -to {s['t1']} "
                    f"-i '{video}' -i '{stop}' -i '{sbot}' "
                    f'-filter_complex "[0:v][1:v]overlay=0:0[tmp];[tmp][2:v]overlay=0:{h - bar_h_bot}[outv]" '
                    f'-map "[outv]" -map 0:a:0 '
                    f"-c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k '{out}'",
                    shell=True)
        elif args.sticker_overlay and sticker_top and sticker_bot:
            # Simple: overlay 2 pre-composited sticker PNGs onto video
            subprocess.check_call(
                f"ffmpeg -y -v error -ss {s['t0']} -to {s['t1']} "
                f"-i '{video}' -i '{sticker_top}' -i '{sticker_bot}' "
                f'-filter_complex "[0:v][1:v]overlay=0:0[tmp];[tmp][2:v]overlay=0:{h - bar_h_bot}[outv]" '
                f'-map "[outv]" -map 0:a:0 '
                f"-c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k '{out}'",
                shell=True)
        elif args.sticker_overlay:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", str(s["t0"]), "-to", str(s["t1"]),
                "-i", str(video),
            ]
            if crop_filter:
                cmd.extend(["-vf", crop_filter])
            cmd.extend([
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                str(out)
            ])
            subprocess.run(cmd, check=True)

    # ---- Cleanup ----
    if not args.keep_intermediate:
        shutil.rmtree(pipe_dir, ignore_errors=True)

    # ---- Summary ----
    total = sum(s["t1"] - s["t0"] for s in selected)
    print(f"\n{'═' * 50}")
    print(f"Done: {len(selected)} clips, {total:.0f}s total")
    print(f"Output: {out_dir}/")
    for f in sorted(out_dir.glob("*.mp4")):
        size = f.stat().st_size / 1024
        print(f"  {f.name} ({size:.0f}KB)")
    print(f"{'═' * 50}")

if __name__ == "__main__":
    main()
