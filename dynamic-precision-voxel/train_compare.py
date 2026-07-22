# train_compare.py - 主实验脚本
#
# 验证"动态精度体素"理论：在 80x60 低分辨率灰度帧上，
#   实验 A: 单帧分类 HOG+SVM（每帧独立预测，序列多数投票）
#   实验 B: 时序分类 特征+LSTM（沿用 ../train_model.py 的 Keras 架构风格）
#   附加实验: 分辨率梯度 80x60 -> 40x30 -> 20x15，看 LSTM 准确率变化
#
# 运行: python train_compare.py   （合成数据，无需下载任何数据集）

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")   # Keras 3 + torch 后端，CPU 可跑

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import features as F
import synthetic_data as S

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE = HERE / "data_cache.npz"
RESOLUTIONS = [(80, 60), (40, 30), (20, 15)]


# ---------------------------------------------------------------- 数据

def get_splits():
    """生成/加载合成数据（带磁盘缓存）。"""
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: (z[f"{k}_frames"], z[f"{k}_labels"]) for k in ("train", "val", "test")}
    print("  生成合成数据（首次运行，之后会缓存到 data_cache.npz）...")
    splits = S.make_splits(n_train=100, n_val=25, n_test=25)
    np.savez_compressed(CACHE,
                        **{f"{k}_frames": v[0] for k, v in splits.items()},
                        **{f"{k}_labels": v[1] for k, v in splits.items()})
    return splits


def _extract_one(args):
    frames, w, h = args
    return F.extract_from_frames(frames, target_size=(w, h))[0]


def extract_split_features(splits, size):
    """对三个 split 批量提取特征（多进程 + 磁盘缓存）。

    返回 dict[split] -> (X [N,T,D], y [N])。"""
    w, h = size
    cache = HERE / f"features_{w}x{h}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: (z[f"{k}_X"], z[f"{k}_y"]) for k in ("train", "val", "test")}

    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for name, (frames, labels) in splits.items():
        args = [(seq, w, h) for seq in frames]
        try:
            with ProcessPoolExecutor() as pool:
                feats = list(pool.map(_extract_one, args, chunksize=8))
        except Exception:
            feats = [_extract_one(a) for a in args]   # 无法多进程时退回串行
        out[name] = (np.stack(feats), labels)
    np.savez_compressed(cache,
                        **{f"{k}_X": v[0] for k, v in out.items()},
                        **{f"{k}_y": v[1] for k, v in out.items()})
    return out


# ---------------------------------------------------------------- 实验 A: 单帧 HOG+SVM

def run_svm(data):
    """单帧分类：只用 HOG 外观特征（严格单帧信息），每帧独立预测，序列多数投票。

    注意：运动特征（帧差能量等）本身已编码两帧间的时序信息，
    因此不放进"单帧"实验，留给 LSTM 时序实验使用。
    """
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]
    N_tr, T, D = X_train.shape
    N_te = X_test.shape[0]
    hog_dim = D - F.N_MOTION_FEATS
    X_train = X_train[:, :, :hog_dim]
    X_test = X_test[:, :, :hog_dim]
    D = hog_dim

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.reshape(-1, D))
    Xte = scaler.transform(X_test.reshape(-1, D))

    svm = LinearSVC(C=1.0, max_iter=5000)
    svm.fit(Xtr, np.repeat(y_train, T))

    frame_pred = svm.predict(Xte).reshape(N_te, T)
    frame_acc = accuracy_score(np.repeat(y_test, T), frame_pred.reshape(-1))

    vote_pred = np.array([np.bincount(p, minlength=len(S.ACTIONS)).argmax()
                          for p in frame_pred])
    vote_acc = accuracy_score(y_test, vote_pred)
    cm = confusion_matrix(y_test, vote_pred)
    return frame_acc, vote_acc, cm


# ---------------------------------------------------------------- 实验 B: LSTM 序列分类

def build_model(input_shape, num_classes):
    """轻量 LSTM，沿用 ../train_model.py 的 Keras 写法（LSTM+Dropout+Dense）。"""
    model = Sequential([
        LSTM(64, return_sequences=False, activation="tanh", input_shape=input_shape),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def run_lstm(data, epochs=40, verbose=0, pca_dim=8):
    """训练并评估 LSTM。返回 (test_acc, cm, history, train_seconds)。

    HOG 部分维度过高（1944 维，对 400 个训练样本是噪声海洋），
    先用 PCA（在训练集上拟合）压到 pca_dim 维，再与运动特征拼接送入 LSTM。
    """
    from sklearn.decomposition import PCA

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    num_classes = len(S.ACTIONS)
    hog_dim = X_train.shape[2] - F.N_MOTION_FEATS

    pca = PCA(n_components=pca_dim)
    if hog_dim > 0:
        pca.fit(X_train[:, :, :hog_dim].reshape(-1, hog_dim))

    def transform(X):
        if hog_dim == 0:
            return X
        hog_low = pca.transform(X[:, :, :hog_dim].reshape(-1, hog_dim))
        hog_low = hog_low.reshape(X.shape[0], X.shape[1], pca_dim)
        return np.concatenate([hog_low, X[:, :, hog_dim:]], axis=2)

    X_train, X_val, X_test = transform(X_train), transform(X_val), transform(X_test)

    # 标准化（用训练集统计量）
    D = X_train.shape[2]
    mean = X_train.reshape(-1, D).mean(0)
    std = X_train.reshape(-1, D).std(0) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    model = build_model((X_train.shape[1], D), num_classes)
    early_stop = EarlyStopping(monitor="val_loss", patience=5,
                               restore_best_weights=True)

    t0 = time.time()
    history = model.fit(X_train, to_categorical(y_train, num_classes),
                        epochs=epochs, batch_size=16,
                        validation_data=(X_val, to_categorical(y_val, num_classes)),
                        callbacks=[early_stop], verbose=verbose)
    train_seconds = time.time() - t0

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm, history, train_seconds


# ---------------------------------------------------------------- 图表

def plot_comparison(svm_acc, lstm_acc, res_results):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = ["单帧 SVM\n(多数投票)"] + [f"LSTM\n{w}x{h}" for w, h in RESOLUTIONS]
    accs = [svm_acc] + [a for a, _ in res_results]
    colors = ["#94a3b8"] + ["#2563eb", "#3b82f6", "#60a5fa"]
    bars = ax.bar(names, accs, color=colors)
    ax.axhline(0.25, ls="--", c="red", lw=1, label="随机猜测 (25%)")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.1%}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("测试集准确率")
    ax.set_title("低分辨率动作识别：单帧 SVM vs 时序 LSTM")
    ax.legend()
    fig.savefig(RESULTS / "comparison_bar.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(S.ACTIONS)), S.ACTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(S.ACTIONS)), S.ACTIONS)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_resolution_curve(res_results):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{w}x{h}" for w, h in RESOLUTIONS]
    accs = [a for a, _ in res_results]
    ax.plot(labels, accs, "o-", color="#2563eb", lw=2)
    for x, a in enumerate(accs):
        ax.text(x, a + 0.01, f"{a:.1%}", ha="center", fontsize=10)
    ax.axhline(0.25, ls="--", c="red", lw=1, label="随机猜测 (25%)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("分辨率"); ax.set_ylabel("LSTM 测试准确率")
    ax.set_title("分辨率梯度实验：多低的分辨率还能认出动作？")
    ax.legend()
    fig.savefig(RESULTS / "resolution_curve.png", bbox_inches="tight")
    plt.close(fig)


def print_cm(cm, header):
    print(f"\n  {header}（行=真实, 列=预测）:")
    print("        " + "".join(f"{a:>6}" for a in S.ACTIONS))
    for i, a in enumerate(S.ACTIONS):
        print(f"  {a:>6}" + "".join(f"{cm[i][j]:>6}" for j in range(cm.shape[1])))


# ---------------------------------------------------------------- 主流程

def main():
    RESULTS.mkdir(exist_ok=True)
    keras.utils.set_random_seed(42)

    print("[1/5] 准备合成数据...")
    splits = get_splits()
    for name, (frames, labels) in splits.items():
        print(f"  {name}: {frames.shape[0]} 段, 每段 {frames.shape[1]} 帧 "
              f"{frames.shape[3]}x{frames.shape[2]}")

    print("\n[2/5] 提取特征 (80x60)...")
    data80 = extract_split_features(splits, (80, 60))
    D = data80["train"][0].shape[2]
    print(f"  每帧特征维度: {D} (HOG {D - F.N_MOTION_FEATS} + 运动 {F.N_MOTION_FEATS})")

    print("\n[3/5] 实验 A: 单帧 HOG + SVM ...")
    svm_frame_acc, svm_vote_acc, svm_cm = run_svm(data80)
    print(f"  单帧准确率: {svm_frame_acc:.2%}")
    print(f"  多数投票后序列准确率: {svm_vote_acc:.2%}")
    print_cm(svm_cm, "SVM 混淆矩阵")

    print("\n[4/5] 实验 B: 特征 + LSTM（含分辨率梯度 80x60 / 40x30 / 20x15）...")
    res_results = []
    lstm_cm = None
    lstm_seconds = 0.0
    for w, h in RESOLUTIONS:
        print(f"  -- 分辨率 {w}x{h} --")
        data = extract_split_features(splits, (w, h))
        acc, cm, history, secs = run_lstm(data)
        res_results.append((acc, cm))
        print(f"     测试准确率: {acc:.2%}  (训练 {len(history.history['loss'])} epochs, "
              f"{secs:.1f}s)")
        if (w, h) == (80, 60):
            lstm_cm = cm
            lstm_seconds = secs
    print_cm(lstm_cm, "LSTM (80x60) 混淆矩阵")

    print("\n[5/5] 保存图表到 results/ ...")
    plot_comparison(svm_vote_acc, res_results[0][0], res_results)
    plot_confusion(svm_cm, f"SVM 多数投票 (准确率 {svm_vote_acc:.1%})",
                   RESULTS / "confusion_svm.png")
    plot_confusion(lstm_cm, f"LSTM 80x60 (准确率 {res_results[0][0]:.1%})",
                   RESULTS / "confusion_lstm.png")
    plot_resolution_curve(res_results)

    print("\n================ 实验结果汇总 ================")
    print(f"  单帧 SVM（多数投票）:  {svm_vote_acc:.2%}")
    for (w, h), (acc, _) in zip(RESOLUTIONS, res_results):
        print(f"  LSTM {w}x{h}:            {acc:.2%}")
    print(f"\n  LSTM 训练耗时 (80x60): {lstm_seconds:.1f} 秒")
    print(f"  图表已保存: {RESULTS}/")
    print("    - comparison_bar.png      (SVM vs LSTM 对比柱状图)")
    print("    - confusion_svm.png / confusion_lstm.png  (混淆矩阵)")
    print("    - resolution_curve.png    (分辨率-准确率曲线)")


if __name__ == "__main__":
    main()
