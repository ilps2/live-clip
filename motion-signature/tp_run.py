#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tp_run.py — 单格增量运行时间对分类实验（防超时）。

用法:
    python tp_run.py --cache pair1          # 预计算特征缓存(每对一次)
    python tp_run.py --pair pair1 --fps 1   # 跑一格并追加 results/tp_results.csv
    python tp_run.py --pair pair1 --singleframe   # 单帧不可分性验证
"""
import argparse
import csv
import os
import numpy as np
import seq_classify as S
from classify_temporal import frame_rf_singleframe

OUT = "results/tp_results.csv"


def cache(pair):
    feats, seg_ids, events = S.load_features(
        f"results/tp_{pair}_mv.npz", f"results/tp_{pair}_labels.csv")
    np.savez_compressed(f"results/tp_{pair}_feats.npz",
                        feats=feats, seg_ids=seg_ids, events=events)
    print(f"[cache] {pair}: {feats.shape}")


def load(pair):
    d = np.load(f"results/tp_{pair}_feats.npz", allow_pickle=True)
    return d["feats"], d["seg_ids"], d["events"]


def append_row(row):
    exists = os.path.exists(OUT) and os.path.getsize(OUT) > 0
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["pair", "fps", "single_frame_rf", "gru", "vote"])
        w.writerow(row)
        f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    ap.add_argument("--pair")
    ap.add_argument("--fps", type=int)
    ap.add_argument("--singleframe", action="store_true")
    args = ap.parse_args()

    if args.cache:
        cache(args.cache)
        return
    feats, seg_ids, events = load(args.pair)
    mu, sd = feats.mean(0), feats.std(0) + 1e-9
    F = (feats - mu) / sd
    classes = sorted(set(events))
    y = np.array([classes.index(e) for e in events])
    if args.singleframe:
        sf = frame_rf_singleframe(F, y, seg_ids)
        print(f"[{args.pair}] single-frame RF: {sf:.3f}", flush=True)
        return
    seq_ok, vote_ok = [], []
    classes = sorted(set(events))
    for s, so, fo in S.run_loso(feats, seg_ids, events, args.fps,
                                epochs=25, n_folds=4, aug_reps=2,
                                classes=classes):
        seq_ok.append(so)
        vote_ok.append(fo)
    row = [args.pair, args.fps, "", round(np.mean(seq_ok), 3),
           round(np.mean(vote_ok), 3)]
    append_row(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
