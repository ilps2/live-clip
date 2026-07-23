#!/usr/bin/env python3
"""
9:16 vertical text card generator for CapCut flip-text videos.
Input: text script with scenes separated by ---
Output: numbered PNG cards at 720x1280.

Usage:
    python capcut_cards.py -i script.txt -o cards/
"""
import subprocess, sys, os, tempfile, json, shutil
from pathlib import Path

W, H = 720, 1280
OUT_DIR = "cards"
SCRIPT_DIR = Path(__file__).parent
T2P = str(SCRIPT_DIR / "text2png_bin")

def ensure_t2p():
    if not os.path.exists(T2P):
        subprocess.run(["swiftc", str(SCRIPT_DIR / "text2png.swift"), "-o", T2P], check=True)

def render_line(text, font_size, color="white"):
    """Render single line → PNG path."""
    p = tempfile.mktemp(suffix=".png")
    subprocess.run([T2P, text, str(font_size), p, color], check=True, capture_output=True)
    return p

def png_size(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
                       capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    return s["width"], s["height"]

def make_card(lines, output_path):
    """
    lines: [(text, font_size, color), ...]
    Renders all lines vertically centered on a 720x1280 black card.
    """
    # Render each line
    pngs = [render_line(t, s, c) for t, s, c in lines]
    heights = [png_size(p)[1] for p in pngs]
    gap = max(16, min(40, (H - sum(heights)) // (len(lines) + 2)))
    y = (H - sum(heights) - gap * (len(lines) - 1)) // 2

    # Build incrementally
    current = None
    for i, (p, h_val) in enumerate(zip(pngs, heights)):
        if i == 0:
            # Black background + first line
            current = tempfile.mktemp(suffix=".png") if len(lines) > 1 else output_path
            subprocess.check_call(
                f"ffmpeg -y -v error -f lavfi -i color=c=black:s={W}x{H}:d=0.1 "
                f"-i '{p}' -frames:v 1 "
                f'-filter_complex "[0:v][1:v]overlay=(W-w)/2:{y}" '
                f"-c:v png '{current}'", shell=True)
        else:
            prev = current
            current = output_path if i == len(lines) - 1 else tempfile.mktemp(suffix=".png")
            subprocess.check_call(
                f"ffmpeg -y -v error -i '{prev}' -i '{p}' "
                f'-filter_complex "[0:v][1:v]overlay=(W-w)/2:{y}" '
                f"-c:v png '{current}'", shell=True)
        y += h_val + gap

    return output_path


def parse_script(text):
    """Scenes separated by ---. Lines may have color/size tags."""
    scenes = []
    current = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line == "---":
            if current:
                scenes.append(current)
                current = []
        elif line and not line.startswith("#"):
            color, size, content = "white", 48, line
            for tag, c, s in [("【黄字】","gold",52),("【红字】","red",56),
                               ("【大字】","white",60),("【巨字】","white",80),
                               ("【数字】","gold",72),("【小字】","white",36)]:
                if line.startswith(tag):
                    color, size, content = c, s, line[len(tag):]
                    break
            current.append((content, size, color))
    if current:
        scenes.append(current)
    return scenes


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", help="Script file (or stdin)")
    p.add_argument("--output", "-o", default=OUT_DIR)
    args = p.parse_args()

    text = open(args.input).read() if args.input else sys.stdin.read()
    ensure_t2p()

    scenes = parse_script(text)
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    # Clean old cards
    for f in out.glob("card_*.png"):
        f.unlink()

    print(f"Generating {len(scenes)} cards → {out}/")
    for i, scene in enumerate(scenes):
        path = str(out / f"card_{i+1:02d}.png")
        make_card(scene, path)
        preview = " | ".join(t[:18] for t, _, _ in scene)
        print(f"  card_{i+1:02d}.png: {preview}")

    print(f"\nDone. Drag {out}/ into CapCut as image sequence.")
