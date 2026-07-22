#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""synthetic_demo.py — 无网络时生成合成 demo 数据，跑通整条管线。

模拟一场 20 分钟带货直播：基础弹幕流 + 3 个高光时刻（弹幕爆发 +
促单话术密集），然后依次跑 promo_detect -> highlight_score -> analyze，
并用已知 GT 验证 F1。

用法:
    python synthetic_demo.py
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path("results")
R.mkdir(exist_ok=True)
rng = random.Random(42)
np.random.seed(42)

DUR = 1200.0  # 20 分钟
HIGHLIGHTS = [(240, 300), (620, 680), (980, 1050)]  # GT 高光区间

PROMO_TEXTS = [
    "宝宝们，3、2、1，上链接！",
    "最后 5 单，拍下就没有了",
    "手快有手慢无，赶紧上车",
    "库存不多了，错过今天再等一年",
    "秒杀价，倒计时开始",
    "最后 10 件，宝宝们快拍",
]
FILLER = ["这个颜色好看", "主播好美", "多少钱", "已拍", "求链接", "讲一讲材质"]


def gen_danmaku(path):
    rows = []
    t = 0.0
    while t < DUR:
        in_hl = any(s <= t <= e for s, e in HIGHLIGHTS)
        lam = 2.5 if in_hl else 0.4  # 弹幕强度（泊松）
        t += rng.expovariate(lam)
        if t >= DUR:
            break
        rows.append({"ts": t, "rel_t": round(t, 2), "type": "DANMU_MSG",
                     "user": f"user{rng.randint(1, 500)}",
                     "text": rng.choice(FILLER)})
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def gen_transcript(path):
    segs = []
    t = 0.0
    while t < DUR:
        in_hl = any(s - 20 <= t <= e + 10 for s, e in HIGHLIGHTS)
        dur = rng.uniform(2, 5)
        text = (rng.choice(PROMO_TEXTS) if in_hl and rng.random() < 0.55
                else rng.choice(FILLER))
        segs.append({"start": round(t, 2), "end": round(t + dur, 2),
                     "text": text, "words": []})
        t += dur + rng.uniform(0.3, 2.5 if in_hl else 5.0)
    with open(path, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return len(segs)


def gen_gt(path):
    with open(path, "w") as f:
        f.write("start,end\n")
        for s, e in HIGHLIGHTS:
            f.write(f"{s},{e}\n")


def run(cmd):
    print(f"[demo] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    n_dm = gen_danmaku(R / "demo_danmaku.jsonl")
    n_tr = gen_transcript(R / "demo_transcript.jsonl")
    gen_gt(R / "demo_ground_truth.csv")
    print(f"[demo] generated: {n_dm} danmaku, {n_tr} transcript segs, 3 GT highlights")

    py = sys.executable
    run([py, "promo_detect.py", "--transcript", str(R / "demo_transcript.jsonl"),
         "--out", str(R / "demo_promo_events.jsonl")])
    run([py, "highlight_score.py",
         "--danmaku", str(R / "demo_danmaku.jsonl"),
         "--promo", str(R / "demo_promo_events.jsonl"),
         "--scores", str(R / "demo_scores.csv"),
         "--segments", str(R / "demo_segments.json"),
         "--ground-truth", str(R / "demo_ground_truth.csv")])
    run([py, "analyze.py", "--scores", str(R / "demo_scores.csv"),
         "--promo", str(R / "demo_promo_events.jsonl"),
         "--ground-truth", str(R / "demo_ground_truth.csv"),
         "--out", str(R / "demo_timeline.png")])


if __name__ == "__main__":
    main()
