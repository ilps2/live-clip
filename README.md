# live-clip v2.0 — AI 粗剪助手

直播带货回放 → 定位精华片段 + 提取卖点文案。

**定位：帮人工省掉「看 4 小时找内容」这一步。**
产出三条人群定向粗剪 + 卖点文案，人工只需删多余部分、配字幕。

## 做了什么

```
4h 直播回放扔进去
        ↓
  ASR 转录（本地，免费）
        ↓
  LLM 语义分析：找到品名/价格/三类人群卖点的时间段
        ↓
  输出三样东西:
    📍 时间戳 — 人工知道从哪开始看
    📝 卖点文案 — 直接拿去配字幕
    🎬 粗剪 MP4 — 人工删 5s 废话就发布
```

## 三条人群定向切片

```
16.7min 直播 → LLM 分析 → 3 条

💰 price 人群  | 补贴价 375，比599便宜    | 43s
🔬 quality 人群 | 三张专利五张检测报告      | 42s  
✨ effect 人群  | 抗老紧致补水全第一        | 43s
```

每条 = 品名 + 价格 + 一个定向卖点，三段不重叠，语音画面同源。

## 快速开始

```bash
# 1. 装依赖
pip install faster-whisper av numpy pandas

# 2. ASR 转录
python livestream-highlight/asr.py \
  --video 直播回放.mp4 \
  --out transcript.jsonl \
  --model base

# 3. AI 粗剪
python v6_triple_cut.py \
  --video 直播回放.mp4 \
  --transcript transcript.jsonl \
  --api-key sk-xxx

# 输出: 3 条 MP4 + 卖点文案，人工修 5s 就发
```

## 成本

| 环节 | 单视频 | 50 主播/天 |
|------|--------|-----------|
| ASR | ¥0 | ¥0 |
| LLM | ¥0.03 | ¥1.5 |
| **合计** | **¥0.03** | **¥1.5/天** |

## 人工做什么

1. 看 3 条粗剪（3 × 40s = 2min）— 不需要看原片
2. 删开头结尾多余寒暄（每条 3-5s）
3. 配卖点文案当字幕 → 发布

## 历史版本

- v1.x: `cut_final.py` — MV 运动矢量 + 规则匹配 + 人群贴纸
- v2.0: `v6_triple_cut.py` — LLM 语义粗剪 + 人群定向
