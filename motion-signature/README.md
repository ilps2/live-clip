# 运动签名验证实验（motion-signature）

验证假设：**从视频码流直接导出的编码器运动矢量（MV），无需任何运动估计计算，
其块级运动场的空间/时间统计签名足以区分直播中的典型视觉事件**。

## 为什么 MV 是"免费"的信号

H.264/H.265 编码器的核心工作就是块匹配运动估计：每个宏块/块的运动矢量
在编码时已经算好并写进码流。解码端只需打开 `flags2=+export_mvs`（ffmpeg），
MV 会作为 side_data 随帧导出。**成本 = 一次解码，运动估计开销为 0**。
对照：自己跑光流/帧差需要逐像素计算，成本高 1–2 个数量级。

本实验用 **PyAV**（`av` 包）实现，关键代码：

```python
stream.codec_context.options = {"flags2": "+export_mvs"}
for frame in container.decode(stream):
    mvs = frame.side_data.get("MOTION_VECTORS")  # None 表示 I 帧
```

每个 MV 含 `src_x/y, dst_x/y, motion_x/y, motion_scale, w, h, source`。
`motion_x/y` 是亚像素单位，需除以 `motion_scale`（通常 4，即 1/4 像素）。

## 帧类型取舍

- **I 帧**：帧内编码，无 MV → 输出零运动场并跳过（占 0.4–1.5%，影响可忽略；
  也可用前后 P 帧插值，未做）。
- **P 帧**：单向前向 MV。
- **B 帧**：双向 MV（source 字段区分前/后向），取幅度较大者。

网格化：MV 按块中心坐标归入统一 **48×27 网格**（720p 下每格约 2 个宏块），
格内取平均；输出 `(T,27,48)` 的幅度矩阵 + 方向矩阵（npz）。

## 实验设计

1. `extract_mv.py` — 从任意视频文件导出块级 MV 场（真实编码器 MV，无降级）。
2. `synth_events.py` — 程序化生成 60 秒合成视频（numpy 渲染 → ffmpeg libx264
   真实编码），含 4 类事件各 3 段，每段 5 秒：
   - `product` 展示商品：中心小方块往复运动
   - `gesture` 主播手势：大块不规则游走
   - `camera` 镜头移动：整幅画面平移
   - `static` 静止：固定机位
3. `classify.py` — 每帧 8 维签名特征 → 段内 ±2 帧时间平滑 → RandomForest，
   **段级划分**（每类 2 段训练、1 段测试，杜绝相邻帧泄漏）。
4. `visualize.py` — quiver 叠加图、时间轴、签名对比。
5. 真实数据：`../livestream-highlight/results/live_clip.flv`（B 站 LPL 解说
   直播 90 秒）同样导出 MV 并出图。

### 8 维签名特征

`global_mag` 全局运动量 · `move_ratio` 运动块占比 · `centroid_x/y` 运动质心 ·
`spread` 空间集中度 · `center_edge` 中心/边缘运动比 · `dir_consist` 方向一致性 ·
`max_mag` 最大块运动。

## 结果

### 分类（合成 4 类，段级测试 598 帧）

**准确率 100%**，混淆矩阵对角。但需诚实说明合成数据的演进：

| 版本 | 准确率 | 说明 |
|---|---|---|
| 单帧特征，无平滑 | 78.3% | camera↔gesture 混淆（大块运动 vs 全局运动边界） |
| + 段内时间平滑 | 96.0% | 平滑掉单帧 MV 噪声后事件级区分度凸显 |
| + 静态背景修正 | 100% | 初版背景噪声逐帧随机重生，等于给全画面加了"假运动"；
固定噪声纹理后静止背景 MV≈0，信号变干净 |

合成数据 100% 只说明"信号上限够"，真实直播会更难（见下）。

### 四类事件的运动签名（见 results/signature_bars.png）

- **camera**：`global_mag`、`move_ratio`、`dir_consist` 三高 —— 全画面同方向运动，最独特
- **gesture**：`global_mag` 中等、`spread` 大、`max_mag` 最高 —— 大而弥散的运动
- **product**：`center_edge` 显著 >1、`spread` 小 —— 运动集中在画面中心小区域
- **static**：所有幅度特征≈0

### 可视化（results/）

- `quiver_product/gesture/camera/static.png` — 4 类事件 MV 网格箭头叠加图。
  camera 帧全画面一致向右箭头；product 帧箭头精确聚集在中心运动物体上；
  静止帧全场近零。
- `timeline_synth.png` — 全局运动量 + 中心/边缘比时间轴，事件区间着色，
  肉眼可分。
- `quiver_real_peak/calm.png`、`timeline_real.png` — 真实直播录像：
  峰值帧清晰可见舞台大屏幕、选手席、前景主持人的整体左移（摄像机横摇），
  平静帧只有零星小箭头。**真实码流的 MV 信号质量与合成一致，可用**。

## 结论

**支持"运动签名可区分事件"**：

1. MV 导出完全成功（PyAV export_mvs，真实录像 2819 帧、合成 1800 帧，
   无需降级方案）。
2. 8 维签名特征 + 时间平滑 + RF，合成 4 类帧级分类 96–100%。
3. 真实直播录像的 MV 场肉眼可辨镜头运动/人物运动，信号干净。

局限与下一步：

- 合成事件是理想化的；真实带货直播事件边界模糊（主播动+镜头动叠加），
  需要对真实录像人工标注事件后重测分类器。
- I 帧空洞、极低码率（块被跳过，无 MV）场景未压测。
- 与实验 A 的弹幕/促单信号互补：MV 签名回答"画面发生了什么"，
  弹幕回答"观众何时兴奋"，融合后高光检测应更准。

## 如何运行

```bash
pip install av scikit-learn   # numpy/pandas/matplotlib 已有

python synth_events.py                                    # 生成合成事件视频
python extract_mv.py --input results/synth.mp4 --out results/synth_mv.npz
python classify.py   --mv results/synth_mv.npz --labels results/synth_labels.csv
python visualize.py                                       # 全部图 -> results/

python extract_mv.py --input ../livestream-highlight/results/live_clip.flv \
    --out results/real_mv.npz                             # 真实录像
```

---

# 实验二：运动动画猜想（render_motion_movie / seq_classify）

**猜想**：把每秒 1 帧的块级运动场像逐格动画一样连起来播放，人眼就能从运动的
动态发展中理解视频内容——MV 场本身就是一部低分辨率动画。

## 渲染方案

`render_motion_movie.py`：视频 → extract_mv → 按指定 fps 采样 → 每帧渲染为
960×540 画面（深色底 + 网格 + 时间戳）→ ffmpeg 合成 MP4/GIF。两种动画语言：

- **quiver 箭头版**：方向=MV 方向，长度+颜色=幅度，静止块（<0.8px）不画
- **heatmap 热度版**：块亮度=运动幅度（inferno 色图）

产物（results/）：
`movie_synth_{quiver,heatmap}_1fps.{mp4,gif}`（60 帧）、
`movie_real_{quiver,heatmap}_1fps.{mp4,gif}`（94 帧，真实直播录像）、
`contact_*.png`（全部帧的拼图总览，便于一眼看完整部"动画"）。

## 可读性观察（人眼验证，诚实记录）

**合成动画**（对照 contact_synth_quiver/heatmap.png）：

- ✅ 事件起止清晰可读：静止段全空 → 运动突然出现的帧就是事件开始
- ✅ 事件类型可辨：camera=全网格同向箭头/整屏亮起；product=中心小簇箭头
  往复漂移；gesture=大团散乱箭头；static=空白
- ✅ 发展过程可读：product 段能看到运动簇从左到右再折返；camera 段方向稳定
- ⚠️ gesture 与"product+轻微镜头抖动"的边界只靠箭头版不易区分，
  heatmap 版看运动范围更直观
- 结论：quiver 适合读**方向**，heatmap 适合读**范围/强度**，两者互补

**真实录像动画**（contact_real_quiver/heatmap.png，90 秒电竞解说）：

- ✅ 0–20s 大场面镜头运动（整屏方向一致的箭头/大范围高亮）与后期
  解说席固定机位形成鲜明对比，场景切换点（约 33s、70s、86s 的全屏突变）
  一眼可辨
- ✅ 固定机位段能看到**人形轮廓的运动热区**（解说员手势/起身），
  位置随时间漂移——"运动的动画"确实承载了内容信息
- ❌ 读不出"内容是什么"：不知道在解说哪场比赛、看不出表情/道具细节。
  运动动画回答"哪在动、怎么动"，不回答"是什么"

## 定量验证（seq_classify.py）

三组数据：easy（12 段，干净背景）、hard（12 段，噪声背景+慢镜头+中途停顿）、
big（48 段，hard 的 4 倍重复）。GRU(32) 序列分类 vs 帧级 RF+段内投票，
段级交叉验证：

| 数据集 | 方法 | 1fps | 5fps | 15fps | 30fps |
|---|---|---|---|---|---|
| easy 12段 | GRU / RF | 1.00/1.00 | 0.92/1.00 | 1.00/1.00 | 0.92/1.00 |
| hard 12段 | GRU / RF | 0.92/0.92 | 0.83/0.92 | 0.92/1.00 | 0.92/1.00 |
| **hard 48段** | GRU / RF | **0.98/1.00** | 0.96/1.00 | 0.98/1.00 | 0.90/1.00 |

（帧率-准确率曲线见 results/fps_curve.png）

**两个诚实的否定/肯定**：

1. **1fps 足够**：帧级 RF 在各帧率下准确率差 ≤8pp（小数据集），48 段集上
   全帧率 100%。每秒 1 帧的运动场保留了这 4 类事件的几乎全部判别信息。
   猜想的核心参数成立。
2. **"连起来看"没有超过"单帧看+投票"**：GRU 序列分类全面略低于帧级 RF+投票，
   且 30fps 长序列反而更差（训练更难）。对这 4 类粗粒度事件，单帧签名已经
   几乎饱和，时间发展带来的增量信息有限。序列模型的优势要等更细粒度事件
   （如"往复晃动 vs 单向移动"这类只能靠时间维区分的类别）才会显现——
   这本身是下一步的好实验。

## 总结论

**猜想部分成立**：

- ✅ 运动动画人眼可读：事件起止、类型、发展过程都能从 1fps MV 动画中读出
- ✅ 1fps 采样在定量上几乎无损
- ❌ "连起来看"对粗粒度事件分类没有超过单帧签名+投票（信息已饱和）
- ❌ 运动动画无法理解语义内容（是什么商品、什么人），必须与 ASR/弹幕信号互补

## 运行

```bash
python render_motion_movie.py --input results/synth.mp4 --name synth --fps-sample 1
python render_motion_movie.py --input ../livestream-highlight/results/live_clip.flv --name real --fps-sample 1
python synth_events.py --hard --repeat 4 --out results/synth_big.mp4 --labels results/synth_big_labels.csv
python extract_mv.py --input results/synth_big.mp4 --out results/synth_big_mv.npz
python seq_classify.py   # easy 集对比 + fps_curve.png
```

---

# 实验三：时间维决战（temporal_pairs / classify_temporal）

**逻辑**：实验二中帧级 RF 没输，是因为 4 类事件的判别信息在空间分布。本实验构造
**原理上只能靠时间维区分**的事件对——两类在所有单帧统计上尽量一致，只有时间
结构不同。这是"连起来看才能理解"猜想的真正战场。

## 4 对时间定义型事件（temporal_pairs.py，每对 200 段×3 秒）

| 对 | A vs B | 时间结构差异 | 单帧等同化设计 |
|---|---|---|---|
| 对1 | osc 往复(三角波) vs ramp 单向(锯齿波) | 轨迹往返 vs 单向跳回 | 同幅度/同速率(4/T)/同范围，随机相位→位置边缘分布相同 |
| 对2 | cresc 渐强 vs dim 渐弱 | 幅度包络上升 vs 下降 | 三角波位置×线性包络，|v| 多重集合相同(顺序相反)，随机相位 |
| 对3 | rhythm 节奏(动5帧停5帧) vs random 随机开关 | 周期 vs 随机 | 相同占空比 50%、相同速度 |
| 对4 | ce 先中心后边缘 vs ec 先边缘后中心 | 事件内部时间顺序相反 | 两块的总量/幅度/位置集合相同，仅顺序交换 |

## 单帧不可分性验证（实验成立的前提，实测非假设）

对每对做**帧级** RF 二分类（段级划分，GroupKFold=5）：

| 对1 | 对2 | 对3 | 对4 |
|---|---|---|---|
| 0.616 | 0.630 | 0.621 | 0.614 |

接近随机 0.5。诚实说明：残余 ~11pp 泄漏来自编码器伪影（ramp 的跳回帧、
rhythm 的开关切换帧 MV 噪声），设计迭代了 3 版才从 0.90 压到 0.62——
初版失败原因：正弦 vs 锯齿的瞬时速度分布不同、段内相位对齐导致"第几帧"
成为泄漏通道。修正：速率/幅度/范围严格等同 + 每段随机相位。

## 分类结果（200 段/对，4 折段级 CV）

| 对 | 单帧RF | RF+投票 1/5/15fps | **GRU 1/5/15fps** |
|---|---|---|---|
| 对1 往复vs单向 | 0.616 | 0.685/0.870/0.915 | **0.635/0.935/0.990** |
| 对2 渐强vs渐弱 | 0.630 | 0.730/0.910/0.960 | **0.900/0.965/0.980** |
| 对3 节奏vs随机 | 0.621 | 0.910/0.985/0.990 | **0.980/0.980/0.950** |
| 对4 中心边缘顺序 | 0.614 | 0.885/0.995/0.995 | **1.000/1.000/1.000** |

图表：results/tp_bars.png（每对三方对比）、results/tp_fps_curve.png（帧率扫描）、
results/tp_pair{1-4}_sidebyside.gif（A/B 并排运动动画，左 A 右 B）。

## 结论

1. **时间维增量被证实**：在单帧特征近随机的战场上，GRU 序列分类 4 对全部
   ≥0.93（@5fps），其中对 4 达到 100%。帧级+投票在低采样率下明显更弱
   （1fps 时落后 GRU 7–12pp；对2 落后 17pp）。**"连起来看"在时间定义型
   事件上不可替代**——它利用的是轨迹形状/趋势/节奏/顺序，这些信息在单帧里
   根本不存在。
2. **帧级投票的"逆袭"机制**：高帧率（15fps）时投票追到 0.92–0.995——
   不是单帧信息变多，而是微弱泄漏（~11pp）× 大量投票帧累积。这提示真实场景
   中区分"真时间特征"和"伪时间相关泄漏"需要不可分性验证这样的前置检验。
3. **1fps 是否足够——取决于事件的时间尺度**：
   - 对2/3/4（模式周期 ≥1.5s 或持续 1.5s 的顺序结构）：1fps 下 GRU 仍
     0.90–1.00，**足够**。
   - 对1（3 秒一个完整往返）：1fps 每段只有 3 个采样点，轨迹形状采样不足，
     GRU 跌到 0.635，**不够**。5fps 恢复到 0.935。
   - 实践含义：1fps 适合"秒级以上发展模式"（渐强/顺序/节奏），快速运动
     细节（亚秒级轨迹）需要 5fps 档。混合采样（常态 1fps + 运动高峰 5fps）
     是自然的工程方案。
4. **最终回答"运动动画能不能当理解器"**：能，但有边界——它能理解"运动的
   句法"（哪在动、怎么动、往哪发展、什么节奏顺序），不能理解"语义"
   （是什么物体）。作为理解器的时间维能力已被本实验证明；与 ASR/弹幕
   信号融合才是完整的内容理解方案。

## 运行

```bash
python temporal_pairs.py --segs-per-class 100          # 生成 4 对视频+标签+演示
python extract_mv.py --input results/tp_pair1.mp4 --out results/tp_pair1_mv.npz
python fast_features.py results/tp_pair1               # 向量化特征(秒级)
python tp_run.py --pair pair1 --singleframe            # 单帧不可分性验证
python tp_run.py --pair pair1 --fps 5                  # GRU vs 投票, 增量落盘
python render_motion_movie.py --input results/tp_pair1_demo.mp4 --name tp1demo --fps-sample 10
```
