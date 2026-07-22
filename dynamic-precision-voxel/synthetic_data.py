# synthetic_data.py - 合成动作数据生成器
#
# 关键实验设计：四类动作（walk/run/wave/idle）的每一帧外观几乎相同——
# 都是灰噪背景上的一个亮块。类与类之间只有【时间运动模式】不同。
# 因此单帧分类器（HOG+SVM）应当接近随机猜测（25%），
# 而时序模型（LSTM）应能轻松区分——这正是"动态精度体素"理论的验证点。
#
# 为避免单帧信息泄漏：
#   - 所有类（含 idle）都含亮块，块大小/亮度/初始位置都随机
#   - idle 的块有随机亮度闪烁，使其帧差能量分布与其他类重叠

import numpy as np

ACTIONS = ["idle", "walk", "run", "wave"]
SEQ_LEN = 30
FRAME_W, FRAME_H = 80, 60


def _draw_block(frame, x, y, size, value):
    H, W = frame.shape
    x0 = int(np.clip(round(x), 0, W - 1))
    y0 = int(np.clip(round(y), 0, H - 1))
    x1 = min(x0 + size, W)
    y1 = min(y0 + size, H)
    if x1 > x0 and y1 > y0:
        frame[y0:y1, x0:x1] = value


def generate_sequence(action, rng, seq_len=SEQ_LEN, size=(FRAME_W, FRAME_H)):
    """生成一段动作帧序列。返回 uint8 [T, H, W]。"""
    W, H = size
    T = seq_len

    block_size = rng.integers(8, 15)              # 8~14 px
    base_value = rng.uniform(170, 235)            # 块亮度
    noise_sigma = rng.uniform(6, 12)
    bg_level = rng.uniform(20, 40)

    # 初始位置（各类都全图随机，防止位置泄漏类别）
    x0 = rng.uniform(5, W - block_size - 5)
    y0 = rng.uniform(5, H - block_size - 5)

    if action == "walk":
        v = rng.uniform(1.5, 2.5) * rng.choice([-1, 1])   # 中速水平匀速
    elif action == "run":
        v = rng.uniform(3.5, 5.0) * rng.choice([-1, 1])   # 高速 + 上下抖动
        bob_amp = rng.uniform(1.5, 3.0)
    elif action == "wave":
        osc_amp = rng.uniform(3, 6)               # 原地左右快速振荡
        osc_period = rng.uniform(4, 7)            # 周期 4~7 帧
        osc_phase = rng.uniform(0, 2 * np.pi)
    elif action == "idle":
        flicker = rng.uniform(10, 30)             # 亮度随机闪烁（掩盖"零帧差"线索）
    else:
        raise ValueError(f"未知动作: {action}")

    frames = np.zeros((T, H, W), dtype=np.uint8)
    for t in range(T):
        frame = rng.normal(bg_level, noise_sigma, (H, W))

        if action == "walk":
            x = (x0 + v * t) % (W - block_size)
            y = y0
            val = base_value
        elif action == "run":
            x = (x0 + v * t) % (W - block_size)
            y = y0 + bob_amp * np.sin(2 * np.pi * t / 3.0)
            val = base_value
        elif action == "wave":
            x = x0 + osc_amp * np.sin(2 * np.pi * t / osc_period + osc_phase)
            y = y0
            val = base_value
        else:  # idle: 静止 + 亮度闪烁
            x, y = x0, y0
            val = base_value + rng.uniform(-flicker, flicker)

        _draw_block(frame, x, y, block_size, val)
        frames[t] = np.clip(frame, 0, 255).astype(np.uint8)

    return frames


def make_dataset(n_per_class, seed=0, seq_len=SEQ_LEN, size=(FRAME_W, FRAME_H)):
    """生成平衡数据集。返回 (frames [N,T,H,W] uint8, labels [N] int)。"""
    rng = np.random.default_rng(seed)
    all_frames, all_labels = [], []
    for label, action in enumerate(ACTIONS):
        for _ in range(n_per_class):
            all_frames.append(generate_sequence(action, rng, seq_len, size))
            all_labels.append(label)
    order = rng.permutation(len(all_labels))
    return (np.stack(all_frames)[order],
            np.array(all_labels, dtype=np.int64)[order])


def make_splits(n_train=100, n_val=25, n_test=25, seq_len=SEQ_LEN):
    """生成 train/val/test 集（每类各 n 段）。"""
    return {
        "train": make_dataset(n_train, seed=42, seq_len=seq_len),
        "val": make_dataset(n_val, seed=43, seq_len=seq_len),
        "test": make_dataset(n_test, seed=44, seq_len=seq_len),
    }
