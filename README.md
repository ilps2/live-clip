# live-clip · AVIS 视频理解信息层

> **给 LLM 一张「视频信息层」：token 压缩 99.95%+（模型无关的核心指标），一条命令出摘要+问答。**

让 LLM 读视频的**信息层**（ASR 转写 + 场景结构 + 运动对象轨迹 + YOLO 语义，约 1,000 token），而不是逐帧像素（约 540 万 token）。先低成本判断视频类型，自动选择最便宜且够用的理解策略。

## 核心数字（10 个真实 B站视频实测，2026-08）

| 指标 | 数值 |
|---|---|
| token 压缩率 | **99.95%~99.98%**（信息层 vs 逐帧估算） |
| 理解链路成功率 | **10/10**（8 类别 × 1~28 分钟 × 360p~1080p） |
| 单视频 LLM 成本（示例） | ~0.006 元（deepseek-v4-flash 空闲价，21min 视频） |
| 重复理解（上下文缓存） | 93% 命中 → 每次成本约 **1/20** |

> **口径**：token 压缩率 = 信息层实际 token ÷ 原始逐帧估算 token（30fps × 1000 token/帧），是模型无关的结构性指标；价格为按 deepseek-v4-flash 2026-08-17 调价空闲价折算的示例，随模型/时段/缓存变化。

## 快速开始（国内网络 5 分钟）

```bash
# 1. 克隆 + 一键安装（模型走国内镜像 hf-mirror + gh-proxy，无需翻墙）
git clone git@github.com:ilps2/live-clip.git && cd live-clip
bash install_models.sh

# 2. 一条命令理解任何 B站视频
python3 understand_video.py "B站链接或BV号"
# → 下载 → AVIS 分析 → 摘要 + 3 个问答 → 报告（token 压缩 99.98%）

# 3. 或理解本地视频
python3 understand_video.py /path/video.mp4 --no-download --ask "讲了什么" --ask "亮点"
```

## 核心能力

| 命令 | 功能 |
|---|---|
| `avis.py classify <video> --asr` | 类型路由（帧差+色彩+ASR+MV 码流，零模型成本） |
| `avis.py encode --obj-tracks` | 一次编码提取全部信号（MV/ASR/场景/YOLO 轨迹） |
| `avis.py prompt <avis_dir>` | 融合提示词（视频 RAG，模型无关） |
| `understand_video.py` | L0 一键理解：B站/本地 → 摘要 + 问答 + 成本报告 |

## 验证数据

**10 视频端到端**（详情见 [docs/视频理解成本降99.9%.md](docs/视频理解成本降99.9%.md)）：

| 视频 | 类型 | 信息层 tok | 原始逐帧 tok | token 压缩 |
|------|------|-----------|-------------|-----------|
| 舞蹈 (360p) | motion | 926 | 135 万 | 99.98% |
| 数码 EDC | speech | 1,166 | 540 万 | 99.98% |
| 恐怖动画 | motion | 643 | 234 万 | 99.97% |
| 美食吃播 | motion | 899 | 540 万 | 99.98% |
| 生活 Vlog | speech | 1,305 | 540 万 | 99.98% |
| 户外烧烤 | speech | 1,091 | 540 万 | 99.98% |
| 蒙眼摸车 | speech | 1,202 | 540 万 | 99.98% |
| 游戏实况 | speech | 649 | 357 万 | 99.98% |
| 科普解说 | speech | 1,193 | 540 万 | 99.98% |
| 手机评测 | motion | 870 | 186 万 | 99.95% |

**消融实验**（同一视频 × 4 信息层 × 8 事实问题）：融合 8/8 > 纯 ASR 6/8 > 逐帧描述 3/8 > 纯轨迹 1/8。

**重复理解**：信息层作为稳定前缀命中上下文缓存（实测 93%），换问题换模型都命中 → 重复成本约为首次的 1/20。

## 架构

```
视频 → ① 免费信号(MV码流/帧差/色彩/ASR) → ② 规则分类
     → 语音密集→纯ASR ｜ 运动密集→轨迹+YOLO ｜ 静态→稀疏帧 ｜ 混合→全套
     → 信息层(≈1k token) → LLM（任何模型）
```

## 安装依赖说明（为什么模型走镜像）

- **软件**（torch/opencv/ultralytics/faster-whisper）：pip 清华镜像
- **模型**（whisper tiny/base + yolov8n）：hf-mirror.com + gh-proxy 国内镜像自动下载（实测 4~6 秒）
- 离线分发可选：模型离线包（夸克网盘，199M）——需要时联系维护者

## 已知边界（诚实说明）

- MOG2 轨迹在 360p 低清/镜头跟随下碎片化（person 召回 5/12），1080p 源效果更好
- YOLO 仅 COCO 80 类，专业对象（食物/型号）仍 unknown
- tiny ASR 错别字影响精确型号提取（可用 base/small）
- 表情/纹理/OCR 等像素级细节需要多模态兜底（分级 L1/L2）

## dsh 插件（DeepSeek Harness）

[dsh-video-understand](https://github.com/ilps2/dsh-video-understand) 是 dsh 插件（独立 repo，topics: dsh/plugin/avis）：给 agent 注册 `video_understand` 工具（B站链接 → 摘要+问答，同上 token 压缩 99.95%+）。

```bash
# 安装（本地目录 file: 引用）
# profile package.json dependencies 加:
#   "dsh-video-understand": "file:/path/to/live-clip/plugin/dsh-video-understand"
# bundles 加 "dsh-video-understand"，然后 pnpm install && dsh web
```

工具调用即跑本仓库的 `understand_video.py`（`VIDEO_UNDERSTAND_SCRIPT` 可配）。已在 web profile 实测：21min 视频 228s 出结果，token 压缩 99.99%，成本 0.006 元。

---

# 原有功能：直播带货 AI 粗剪助手（v2.0）

> 直播回放 → 定位精华片段 + 提取卖点文案。
> **定位：帮人工省掉「看 4 小时找内容」这一步。**

```
4h 直播回放扔进去 → ASR 转录（本地免费）→ LLM 语义分析
→ 输出: 📍时间戳 + 📝卖点文案 + 🎬粗剪 MP4（3 条人群定向切片）
```

## 快速开始（旧版粗剪）

```bash
pip install faster-whisper av numpy pandas
python livestream-highlight/asr.py --video 直播回放.mp4 --out transcript.jsonl --model base
python v6_triple_cut.py --video 直播回放.mp4 --transcript transcript.jsonl --api-key sk-xxx
```

## 成本（旧版）

| 环节 | 单视频 | 50 主播/天 |
|------|--------|-----------|
| ASR | ¥0 | ¥0 |
| LLM | ¥0.03 | ¥1.5 |
| **合计** | **¥0.03** | **¥1.5/天** |

## 历史版本

- v1.x: `cut_final.py` — MV 运动矢量 + 规则匹配 + 人群贴纸
- v2.0: `v6_triple_cut.py` — LLM 语义粗剪 + 人群定向

## License

MIT（待定）· 作者：斯文的鱼在学习
