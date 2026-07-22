#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""classify_temporal.py — 时间维决战：帧级 vs 序列级分类。

每对事件:
    1) 单帧不可分性验证: 帧级 RF（段级划分）对单帧二分类 —— 期望 ~50%
    2) 帧级 RF + 段内投票（基线）
    3) GRU 序列分类 @ 1/5/15 fps
验证: 5 折分组 CV（每折 40 段，200 段/对）。

用法: python classify_temporal.py
输出: results/tp_results.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

import seq_classify as S

PAIRS = ["pair1", "pair2", "pair3", "pair4"]
N_FOLDS = 5


def frame_rf_singleframe(F, y, groups):
    """单帧二分类（段级划分），验证单帧不可分性。"""
    accs = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr, te in gkf.split(F, y, groups):
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(F[tr], y[tr])
        accs.append(accuracy_score(y[te], rf.predict(F[te])))
    return float(np.mean(accs))


def main():
    rows = []
    for pair in PAIRS:
        feats, seg_ids, events = S.load_features(
            f"results/tp_{pair}_mv.npz", f"results/tp_{pair}_labels.csv")
        mu, sd = feats.mean(0), feats.std(0) + 1e-9
        F = (feats - mu) / sd
        classes = sorted(set(events))
        y = np.array([classes.index(e) for e in events])

        sf_acc = frame_rf_singleframe(F, y, seg_ids)
        print(f"[{pair}] 单帧RF(段级划分): {sf_acc:.3f} "
              f"(classes={classes})", flush=True)

        rec = {"pair": pair, "single_frame_rf": round(sf_acc, 3)}
        for fps in [1, 5, 15]:
            seq_ok, vote_ok = [], []
            for s, so, fo in S.run_loso(feats, seg_ids, events, fps,
                                        epochs=50, n_folds=N_FOLDS, aug_reps=3):
                seq_ok.append(so); vote_ok.append(fo)
            rec[f"gru_{fps}fps"] = round(float(np.mean(seq_ok)), 3)
            rec[f"vote_{fps}fps"] = round(float(np.mean(vote_ok)), 3)
            print(f"  fps={fps:>2}: GRU={rec[f'gru_{fps}fps']:.3f} "
                  f"RF+投票={rec[f'vote_{fps}fps']:.3f}", flush=True)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv("results/tp_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
