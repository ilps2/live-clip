# live-clip v2.0 — AI 粗剪 · 人工精修

直播带货回放 → LLM 语义选段 → 三条人群定向切片。

## 定位

**AI 做加法（多裁），人做减法（修剪）。**

传统剪辑：看 4h 回放 → 找高光 → 精剪 → 4h/条
本工具：扔视频 → 等 3min → 拿到 3 条粗剪 → 各删 5s 废话 → 3min/条

## 三条人群切片

```
16.7min 直播 → LLM 分析 ASR → 3 条切片

💰 price 人群  | 补贴价 375，比599便宜  | 43s
🔬 quality 人群 | 三张专利五张检测报告    | 42s  
✨ effect 人群  | 抗老紧致补水全第一      | 43s
```

## 快速开始

```bash
# 1. 安装依赖
pip install faster-whisper av numpy pandas

# 2. ASR 转录（首次需下载模型）
python livestream-highlight/asr.py \
  --video 直播回放.mp4 \
  --out transcript.jsonl \
  --model base

# 3. AI 粗剪（需要 DeepSeek API key）
python v6_triple_cut.py \
  --video 直播回放.mp4 \
  --transcript transcript.jsonl \
  --api-key sk-xxx \
  --model deepseek-v4-pro

# 输出: 三条 MP4，拖进剪映各修 5s 即发
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--min-dur` | 25 | 切片最短(秒) |
| `--max-dur` | 55 | 切片最长(秒) |
| `--padding` | 3 | 裁切边距(秒) |
| `--model` | deepseek-v4-pro | LLM 模型 |

## 成本

| 环节 | 单视频 | 50主播/天 |
|------|--------|-----------|
| ASR (faster-whisper CPU) | ¥0 | ¥0 |
| LLM (DeepSeek Pro) | ¥0.03 | ¥1.5 |
| FFmpeg 裁切 | ¥0 | ¥0 |
| **合计** | **¥0.03** | **¥1.5/天** |

## 工作原理

```
ASR 全文
  ↓
LLM 语义分析: 找到品名/价格/三类人群卖点的时间段
  ↓
FFmpeg 连续裁切: 语音画面同源，不拆不拼
  ↓
3 条独立切片: price / quality / effect 各一条
  ↓
人工精修: 各删 3-5s 开头结尾废话 → 发布
```

## 历史

- v1.x: `cut_final.py` — MV 运动矢量 + 关键词规则匹配 + 贴纸
- v2.0: `v6_triple_cut.py` — LLM 语义选段 + 人群定向

v1.x 完整管线保留在 `cut_final.py`、`scene_classifier.py`、`motion-signature/` 中。
