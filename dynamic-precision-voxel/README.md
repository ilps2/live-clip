# dynamic-precision-voxel · 低分辨率视频动作识别对比实验

验证"动态精度体素"理论的关键实验：在 **80×60 灰度低分辨率**（隐私安全、人脸不可辨识）上，
**时间维度建模是否带来增益**？

- **实验 A（单帧）**：HOG + SVM，每帧独立预测，序列结果多数投票
- **实验 B（时序）**：HOG(PCA 降维) + 运动特征 + LSTM（沿用 `../train_model.py` 的 Keras 写法）

## 实验设计

合成数据生成器（`synthetic_data.py`）程序化生成 4 类动作，每段 30 帧：

| 类别 | 运动模式 |
|---|---|
| `walk` | 亮块中速水平匀速移动 |
| `run`  | 亮块高速移动 + 上下抖动 |
| `wave` | 亮块原地左右快速振荡 |
| `idle` | 亮块静止（带亮度闪烁，掩盖"零帧差"线索） |

关键：**四类动作的单帧外观几乎相同**（都是灰噪背景上一个随机大小/亮度/位置的亮块），
只有时间运动模式可区分。因此单帧 SVM 应当接近随机（25%），LSTM 应显著更高。

## 如何运行

```bash
python train_compare.py
```

无需下载任何数据集。首次运行自动生成合成数据（缓存到 `data_cache.npz`）并提取特征
（缓存到 `features_*.npz`，多进程加速），之后重跑秒级完成。删除缓存文件即可重新生成。

依赖：numpy / scikit-learn / scikit-image / opencv-python-headless / torch / keras（Keras 3 + torch 后端，纯 CPU）。

## 本次运行结果（test 集 100 段）

| 模型 | 测试准确率 |
|---|---|
| 单帧 SVM（多数投票） | **27.0%**（单帧准确率 24.5%，≈ 随机） |
| LSTM 80×60 | **87.0%** |
| LSTM 40×30 | **94.0%** |
| LSTM 20×15 | **95.0%** |

LSTM 训练耗时约 7 秒（80×60，CPU）。

**分辨率梯度结果反而升高的原因**：运动特征（亮度质心轨迹、帧差能量等）本身对分辨率极其鲁棒，
而 HOG 外观特征在本实验中不含类别信息、只贡献过拟合噪声——分辨率越低 HOG 噪声越少。
这恰好佐证了理论：识别动作靠的是"动态"而非"精度"。

## 结果文件（results/）

- `comparison_bar.png` — SVM vs 各分辨率 LSTM 对比柱状图
- `confusion_svm.png` / `confusion_lstm.png` — 混淆矩阵
- `resolution_curve.png` — 分辨率-准确率曲线（真实跑出的数据）

## 换成真实视频

```python
import features

# 从视频文件提取（自动转灰度 + resize 到 80x60）
feats, meta = features.extract_from_video("your_video.mp4")
# feats: [T, 1952]  (HOG 1944 + 运动 8)
# meta["bboxes"]: 每帧运动区域 (x, y, w, h)；meta["energies"]: 每帧运动能量

# 或从已有帧序列（uint8 [T, H, W] 灰度 或 [T, H, W, 3]）
feats, meta = features.extract_from_frames(frames)
```

将 `feats` 按 30 帧滑窗切成序列即可送入 LSTM；单帧基线则逐帧送入 SVM。

## 文件说明

- `features.py` — 特征提取：灰度+resize、帧差运动检测、HOG + 运动特征（面积/质心/能量/质心速度/亮度质心）
- `synthetic_data.py` — 合成数据生成器（train 400 / val 100 / test 100 段）
- `train_compare.py` — 主实验脚本
- `results/` — 图表输出
