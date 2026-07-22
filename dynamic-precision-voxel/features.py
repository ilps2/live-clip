# features.py - 低分辨率视频特征提取
#
# 核心流程（对应"动态精度体素"理论）:
#   视频帧 -> 灰度 -> resize 到 80x60（人脸不可辨识的隐私安全分辨率）
#   -> 帧差运动检测（absdiff + median filter + threshold）
#   -> 每帧特征向量 = HOG 外观特征 + 运动特征（面积/质心/能量）
#
# HOG 单元格大小随分辨率自适应，保证任意输入分辨率下特征维度一致。

import numpy as np
from scipy.ndimage import median_filter
from skimage.feature import hog

BASE_W, BASE_H = 80, 60          # 目标低分辨率
HOG_ORIENTATIONS = 9
HOG_CELLS_PER_BLOCK = (2, 2)
GRID_CELLS_W, GRID_CELLS_H = 10, 7   # 期望的 HOG 网格数（用 cell 大小自适应凑齐）
N_MOTION_FEATS = 8  # 运动特征: 面积, 质心x, 质心y, 能量, 质心速度dx, 质心速度dy, 亮度质心x, 亮度质心y


def _cell_size(width, height):
    """按目标网格数反推 cell 像素大小，使不同分辨率下 HOG 维度一致。"""
    return (max(1, width // GRID_CELLS_W), max(1, height // GRID_CELLS_H))


def motion_detection(frames):
    """帧差运动检测。

    frames: float32 [T, H, W]，取值 0~1
    返回: (bboxes [T, 4] (x,y,w,h，无运动时全0), energies [T], areas [T], centroids [T, 2])
    """
    T, H, W = frames.shape
    bboxes = np.zeros((T, 4), dtype=np.float32)
    energies = np.zeros(T, dtype=np.float32)
    areas = np.zeros(T, dtype=np.float32)
    centroids = np.zeros((T, 2), dtype=np.float32)

    prev = frames[0]
    for t in range(T):
        if t == 0:
            diff = np.zeros((H, W), dtype=np.float32)
        else:
            diff = np.abs(frames[t] - prev).astype(np.float32)
        prev = frames[t]

        diff = median_filter(diff, size=3)
        mask = diff > 0.10

        energies[t] = float(diff.mean())
        ys, xs = np.nonzero(mask)
        if len(xs) > 0:
            areas[t] = len(xs) / (H * W)                       # 归一化面积
            centroids[t] = [xs.mean() / W, ys.mean() / H]      # 归一化质心
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            bboxes[t] = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

    return bboxes, energies, areas, centroids


def frame_features(gray_frame):
    """单帧 HOG 特征。gray_frame: float32 [H, W] 0~1。"""
    H, W = gray_frame.shape
    cell = _cell_size(W, H)
    return hog(gray_frame,
               orientations=HOG_ORIENTATIONS,
               pixels_per_cell=cell,
               cells_per_block=HOG_CELLS_PER_BLOCK,
               feature_vector=True).astype(np.float32)


def extract_from_frames(frames, target_size=(BASE_W, BASE_H)):
    """从帧序列提取特征。

    frames: uint8/float [T, H, W] 灰度帧（或 [T, H, W, 3] BGR/RGB）
    返回: (features [T, D], meta dict)
    """
    import cv2

    tw, th = target_size
    gray = []
    for f in frames:
        if f.ndim == 3:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if f.shape[:2] != (th, tw):
            f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        gray.append(f.astype(np.float32) / 255.0)
    gray = np.stack(gray)

    bboxes, energies, areas, centroids = motion_detection(gray)

    # 亮度质心：直接跟踪亮块位置（对亮于均值的像素取强度加权质心）
    T0, H, W = gray.shape
    yy, xx = np.mgrid[0:H, 0:W]
    bright_c = np.full((T0, 2), 0.5, dtype=np.float32)
    for t in range(T0):
        wgt = gray[t] - gray[t].mean()
        wgt = np.clip(wgt, 0, None)
        s = wgt.sum()
        if s > 1e-6:
            bright_c[t] = [(wgt * xx).sum() / s / W, (wgt * yy).sum() / s / H]

    # 质心速度（相邻帧质心位移，无运动时保持上一帧质心避免跳变）
    T0 = gray.shape[0]
    filled = centroids.copy()
    last = np.array([0.5, 0.5])
    for t in range(T0):
        if areas[t] > 0:
            last = filled[t]
        else:
            filled[t] = last
    velocity = np.zeros((T0, 2), dtype=np.float32)
    velocity[1:] = filled[1:] - filled[:-1]

    T = T0
    hog_dim = len(frame_features(gray[0]))
    feats = np.zeros((T, hog_dim + N_MOTION_FEATS), dtype=np.float32)
    for t in range(T):
        hog_f = frame_features(gray[t])
        motion_f = np.array([areas[t], centroids[t, 0], centroids[t, 1],
                             energies[t], velocity[t, 0], velocity[t, 1],
                             bright_c[t, 0], bright_c[t, 1]],
                            dtype=np.float32)
        feats[t] = np.concatenate([hog_f, motion_f])

    meta = {
        "width": tw, "height": th, "num_frames": T,
        "hog_dim": hog_dim, "feature_dim": feats.shape[1],
        "bboxes": bboxes, "energies": energies,
    }
    return feats, meta


def extract_from_video(path, target_size=(BASE_W, BASE_H)):
    """从视频文件提取特征。返回 (features [T, D], meta dict)。"""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"视频中没有可读帧: {path}")
    feats, meta = extract_from_frames(frames, target_size)
    meta["source"] = str(path)
    return feats, meta
