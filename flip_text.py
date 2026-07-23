#!/usr/bin/env python3
"""
9:16 flip-text: typewriter + pop-in effects.
Typewriter mode: each line reveals left-to-right character-by-character.
Pop mode: each line fades in with pop effect.

Usage:
    python flip_text.py -i script.txt -o output/ [--typewriter]
"""
import subprocess, sys, os, tempfile, json
from pathlib import Path

W, H = 720, 1280
FPS = 30
SCRIPT_DIR = Path(__file__).parent
T2P = str(SCRIPT_DIR / "text2png_bin")
OUT_DIR = None  # set at runtime for persistent temp files

def ensure_t2p():
    if not os.path.exists(T2P):
        subprocess.run(["swiftc", str(SCRIPT_DIR / "text2png.swift"), "-o", T2P], check=True)

def render_line(text, font_size, color, name):
    p = str(OUT_DIR / f"{name}.png")
    subprocess.run([T2P, text, str(font_size), p, color], check=True, capture_output=True)
    return p

def png_size(path):
    r = subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_streams",path],
                       capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    return s["width"], s["height"]


def make_typewriter_scene(lines_data, output_path, char_speed=0.08):
    """
    True typewriter: characters appear one at a time.
    Optimized: render full line once, ffmpeg crops individual chars.
    """
    # Render each line once
    full_pngs = [render_line(t, s, c, f"tw_full_{i}") for i, (t, s, c) in enumerate(lines_data)]
    full_sizes = [png_size(p) for p in full_pngs]
    
    # Per-line: split into characters based on width
    line_chars = []
    for (text, _, _), (line_w, line_h) in zip(lines_data, full_sizes):
        chars = list(text)
        char_w = line_w // len(chars) if chars else 32
        # Crop each char from the full line PNG
        char_crops = []
        for j in range(len(chars)):
            char_crops.append((j * char_w, char_w))  # (x_offset, width)
        line_chars.append((chars, char_crops, char_w, line_h))

    # Layout
    gap = 24
    total_h = sum(lh for _, _, _, lh in line_chars) + gap * (len(line_chars) - 1)
    y_start = (H - total_h) // 2

    # Timing
    total_chars = sum(len(cs) for cs, _, _, _ in line_chars)
    total_dur = total_chars * char_speed + 1.5

    # Build filter: for each char, crop from full-line PNG, overlay at position
    filters = [f"color=c=black:s={W}x{H}:d={total_dur}:r={FPS}[bg]"]
    last_out = "bg"
    char_idx = 0
    y = y_start

    for line_idx, (full_png, (chars, char_crops, char_w, line_h)) in enumerate(zip(full_pngs, line_chars)):
        line_w_px = char_w * len(chars)
        x_start = (W - line_w_px) // 2
        
        for c_idx, (ch, (cx, cw)) in enumerate(zip(chars, char_crops)):
            t0 = char_idx * char_speed
            x = x_start + cx
            next_out = f"final_out" if (line_idx == len(line_chars) - 1 and c_idx == len(chars) - 1) else f"t{char_idx}"
            
            # Crop single char from full line PNG, then overlay
            filters.append(
                f"movie='{full_png}':loop=1[c{char_idx}];"
                f"[c{char_idx}]crop={cw}:{line_h}:{cx}:0,"
                f"setpts=PTS-STARTPTS+{t0}/TB[ch{char_idx}];"
                f"[{last_out}][ch{char_idx}]overlay={x}:{y}:"
                f"enable='between(t,{t0},{total_dur})'[{next_out}]"
            )
            last_out = next_out
            char_idx += 1
        
        y += line_h + gap

    filter_str = ";".join(filters)
    subprocess.check_call(
        f"ffmpeg -y -v error -filter_complex \"{filter_str}\" "
        f"-map '[{last_out}]' -c:v libx264 -preset fast -crf 23 "
        f"-pix_fmt yuv420p -t {total_dur} '{output_path}'",
        shell=True)
    return output_path


def make_pop_scene(lines_data, output_path):
    """Fade-in pop: lines appear with fade, sequentially."""
    pngs = [render_line(t, s, c, f"pop_{i}") for i, (t, s, c) in enumerate(lines_data)]
    sizes = [png_size(p) for p in pngs]
    
    gap = 24
    total_h = sum(h for _, h in sizes) + gap * (len(pngs) - 1)
    y = (H - total_h) // 2

    per_line = 0.75
    total_dur = len(pngs) * per_line + 1.0

    filters = [f"color=c=black:s={W}x{H}:d={total_dur}:r={FPS}[bg]"]
    last_out = "bg"

    for i, (png, (_, ph)) in enumerate(zip(pngs, sizes)):
        t0 = i * per_line
        next_out = f"out{i}" if i == len(pngs) - 1 else f"tmp{i}"
        filters.append(
            f"movie='{png}':loop=1,setpts=N/{FPS}/TB[f{i}];"
            f"[f{i}]fade=in:st=0:d=0.25:alpha=1,"
            f"setpts=PTS-STARTPTS+{t0}/TB[r{i}];"
            f"[{last_out}][r{i}]overlay=(W-w)/2:{y}:enable='between(t,{t0},{total_dur})'[{next_out}]"
        )
        last_out = next_out
        y += ph + gap

    filter_str = ";".join(filters)
    subprocess.check_call(
        f"ffmpeg -y -v error -filter_complex \"{filter_str}\" "
        f"-map '[{last_out}]' -c:v libx264 -preset fast -crf 23 "
        f"-pix_fmt yuv420p -t {total_dur} '{output_path}'",
        shell=True)
    return output_path


def parse_script(text):
    scenes, current = [], []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line == "---":
            if current: scenes.append(current); current = []
        elif line and not line.startswith("#"):
            color, size, content = "white", 48, line
            for tag, c, s in [("【黄字】","gold",52),("【红字】","red",56),
                               ("【大字】","white",60),("【巨字】","white",80),
                               ("【数字】","gold",72),("【小字】","white",36)]:
                if line.startswith(tag):
                    color, size, content = c, s, line[len(tag):]; break
            current.append((content, size, color))
    if current: scenes.append(current)
    return scenes


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input","-i",help="Script file")
    p.add_argument("--output","-o",default="flip_videos")
    p.add_argument("--typewriter","-t",action="store_true",help="Typewriter reveal effect")
    args = p.parse_args()

    text = open(args.input).read() if args.input else sys.stdin.read()
    ensure_t2p()
    scenes = parse_script(text)
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    OUT_DIR = out
    for f in out.glob("scene_*.mp4"): f.unlink()
    for f in out.glob("*.png"): f.unlink()

    mode = "typewriter" if args.typewriter else "fade-in"
    make_fn = make_typewriter_scene if args.typewriter else make_pop_scene
    
    print(f"Generating {len(scenes)} scenes ({mode})...")
    for i, scene in enumerate(scenes):
        path = str(out / f"scene_{i+1:02d}.mp4")
        make_fn(scene, path)
        preview = " | ".join(t[:15] for t, _, _ in scene)
        print(f"  scene_{i+1:02d}.mp4: {preview}")

    print(f"\nDone. Drag {out}/ into CapCut, add BGM.")
