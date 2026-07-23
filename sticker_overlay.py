"""Sticker overlay generator - gradient bars with text, zero Python deps."""
import struct, zlib, os, subprocess, shutil


def _make_png(w, h, pixels_rgba):
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    header = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
    raw = b''
    for y in range(h):
        raw += b'\x00'
        for x in range(w):
            r, g, b, a = pixels_rgba[y * w + x]
            raw += struct.pack('BBBB', r, g, b, a)
    idat = chunk(b'IDAT', zlib.compress(raw))
    iend = chunk(b'IEND', b'')
    return header + ihdr + idat + iend


def gradient_bar(width, height, color=(10,10,10), top_alpha=160, bot_alpha=0):
    pixels = []
    for y in range(height):
        a = int(top_alpha + (bot_alpha - top_alpha) * y / max(height-1, 1))
        for x in range(width):
            pixels.append((*color, a))
    return pixels


def sticker_overlay(frame_w, frame_h, top_pct=0.21, bot_pct=0.21):
    top_h = int(frame_h * top_pct)
    bot_h = int(frame_h * bot_pct)
    top_pixels = gradient_bar(frame_w, top_h, color=(10,10,10), top_alpha=160, bot_alpha=0)
    bot_pixels = gradient_bar(frame_w, bot_h, color=(10,10,10), top_alpha=0, bot_alpha=160)
    top_png = _make_png(frame_w, top_h, top_pixels)
    bot_png = _make_png(frame_w, bot_h, bot_pixels)
    import tempfile
    td = tempfile.gettempdir()
    pid = os.getpid()
    top_path = os.path.join(td, f"sticker_top_{pid}.png")
    bot_path = os.path.join(td, f"sticker_bot_{pid}.png")
    with open(top_path, 'wb') as f: f.write(top_png)
    with open(bot_path, 'wb') as f: f.write(bot_png)
    return top_path, bot_path, top_h, bot_h
