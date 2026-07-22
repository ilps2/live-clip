#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""promo_detect.py — 促单话术模式检测。

用法:
    python promo_detect.py --transcript results/transcript.jsonl \
        --out results/promo_events.jsonl

输出 JSONL 每行:
    {"start": 12.3, "end": 15.1, "pattern": "上链接", "weight": 3.0,
     "text": "命中的句子"}

也可作为模块被 highlight_score.py import（detect_promo / score_events）。
"""
import argparse
import json
import re
from pathlib import Path

# 中文促单话术模式库: (正则, 权重)。权重反映"临近下单/高光"的信号强度。
PATTERNS = [
    (r"上\s*(个)?链\s*接", 3.0),
    (r"上\s*车", 3.0),
    (r"3\s*[,，、]?\s*2\s*[,，、]?\s*1", 2.5),
    (r"三\s*二\s*一", 2.5),
    (r"倒计时", 2.0),
    (r"秒\s*杀", 2.5),
    (r"最\s*后\s*[一二两三四五六七八九十\d]+\s*[单件个份只]", 3.0),
    (r"最\s*后\s*[一二两三四五六七八九十\d]+\s*(?![单件个份只])", 2.0),
    (r"拍\s*下", 1.5),
    (r"(去|去)?拍\s*(一|1)\s*号", 2.5),
    (r"库\s*存\s*(不足|紧张|没|少|还有)", 2.0),
    (r"宝\s*宝\s*们", 0.8),
    (r"手\s*快\s*有", 2.0),
    (r"错过\s*(就)?没(有)?", 2.0),
    (r"不\s*要\s*(错\s*过|犹\s*豫)", 1.5),
    (r"(现|马)在\s*(就|立刻|马上)(去)?(拍|买|抢)", 2.0),
    (r"下\s*单", 1.2),
    (r"抢\s*(到|完)\s*(就)?(没|完)", 1.8),
    (r"(半|1|一)分?钟\s*(内|之?后)?(结|恢)束?", 1.5),
]
COMPILED = [(re.compile(p), p, w) for p, w in PATTERNS]


def detect_promo(transcript_path: str) -> list:
    """对转录 JSONL 逐句匹配，返回促单事件列表。"""
    events = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            seg = json.loads(line)
            text = seg["text"]
            for rx, pat, w in COMPILED:
                if rx.search(text):
                    events.append({
                        "start": seg["start"], "end": seg["end"],
                        "pattern": pat, "weight": w, "text": text,
                    })
    # 同句多模式命中合并为强度叠加（封顶）
    merged = {}
    for e in events:
        key = (e["start"], e["end"])
        if key in merged:
            merged[key]["weight"] = min(merged[key]["weight"] + e["weight"], 5.0)
            merged[key]["pattern"] += "|" + e["pattern"]
        else:
            merged[key] = dict(e)
    return sorted(merged.values(), key=lambda x: x["start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", default="results/promo_events.jsonl")
    args = ap.parse_args()
    events = detect_promo(args.transcript)
    with open(args.out, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[promo] {len(events)} events -> {args.out}")


if __name__ == "__main__":
    main()
