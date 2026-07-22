# 廉价信号预测直播高光时刻 · 实验 A

验证假设：**弹幕活跃度 + 促单话术模式**两个廉价信号（不需要视觉理解、不需要大模型）
足以预测带货直播的高光/切片时刻。目标：片段级 **F1 ≥ 0.7**（IoU > 0.5 计命中）。

## 管线架构

```
直播间 (B 站)
   │
   ├── danmaku.py ──► danmaku.jsonl ──────────────┐
   │   (弹幕 websocket, 时间戳)                    │
   ├── recorder.py ──► live.flv/mp4               │
   │   (yt-dlp/streamlink 拉流)                    ▼
   └── asr.py ──► transcript.jsonl ──► promo_detect.py ──► promo_events.jsonl
        (faster-whisper, 词级时间戳)   (中文促单正则模式库)
                                                  │
                highlight_score.py ◄──────────────┘
                (30s 滑窗速率 → z-score → 加权融合 → 阈值 → 候选片段)
                     │                        │
                     ▼                        ▼
              scores.csv + segments.json   evaluate() vs ground_truth.csv
                     │                        (P / R / F1, 片段级 IoU>0.5)
                     ▼
                analyze.py ──► results/timeline.png
```

## 实验 A 协议

1. 选 2–3 场带货直播，**同时**启动 `danmaku.py` 与 `recorder.py`（记录各自启动 epoch 以对齐 rel_t 时间轴；建议在同一脚本里先后启动并记下 `t0` 差值）。
2. 录制全程（1–2 小时/场），抓全程弹幕。
3. `asr.py` 转录 → `promo_detect.py` 提取促单事件。
4. **人工标注 ground truth**：回放视频，把"值得切片"的高光区间写入 `ground_truth.csv`（`start,end` 秒）。
5. `highlight_score.py --ground-truth ...` 输出 P/R/F1；网格搜索 `w1/w2/threshold`，报告 F1 是否 ≥ 0.7。
6. `analyze.py` 出时间线大图，人工复核 bad case。

## 如何运行

```bash
# 0. 依赖（托管 python）
pip install websocket-client brotli faster-whisper   # ffmpeg、yt-dlp 需已安装

# 1. 找在线房间（可选，电商/生活/时尚分区）
curl -s "https://api.live.bilibili.com/room/v1/Area/getRoomList?parent_area_id=10&page=1&page_size=20&sort_type=online" -H "User-Agent: Mozilla/5.0"

# 2. 弹幕 + 录制（同时开两个终端）
python danmaku.py  --room <房间号> --duration 3600 --out results/danmaku.jsonl
python recorder.py --url https://live.bilibili.com/<房间号> --duration 3600 --out results/live.flv

# 3. ASR（模型首次需下载；失败可跳过，融合层自动降级为纯弹幕信号）
python asr.py --video results/live.flv --out results/transcript.jsonl --model small

# 4. 促单检测 → 融合打分 → 评估 → 可视化
python promo_detect.py --transcript results/transcript.jsonl --out results/promo_events.jsonl
python highlight_score.py --danmaku results/danmaku.jsonl --promo results/promo_events.jsonl \
    --scores results/scores.csv --segments results/segments.json \
    --ground-truth results/ground_truth.csv
python analyze.py --scores results/scores.csv --promo results/promo_events.jsonl \
    --ground-truth results/ground_truth.csv --out results/timeline.png

# 无网络演示（合成 20 分钟带货直播，端到端跑通 + F1）
python synthetic_demo.py
```

## 为什么选 B 站（平台选型）

- **直播弹幕 websocket 协议公开文档化**且匿名可用（进房 + WBI 签名 token 即可），
  无需登录、无需逆向 App。抖音/淘宝直播没有公开的弹幕接口，需逆向私有协议或
  抓包，脆弱且有合规风险。
- 录制侧 yt-dlp/streamlink 原生支持 B 站直播。
- 未来扩展：抽象 `ChatSource` 接口（`iter_events(room) -> {ts, type, text}`），
  抖音可走第三方 danmaku 聚合服务或无障碍/OCR 方案，信号融合层不用改。

## 数据格式规范

| 文件 | 格式 | 字段 |
|---|---|---|
| `danmaku.jsonl` | JSONL | `ts`(epoch), `rel_t`(秒,相对抓取开始), `type`, `user`, `text` |
| `transcript.jsonl` | JSONL | `start`, `end`(秒), `text`, `words:[[词,起,止]]` |
| `promo_events.jsonl` | JSONL | `start`, `end`, `pattern`(命中正则), `weight`(强度,封顶5), `text` |
| `scores.csv` | CSV | `t_start,t_end,dm_rate,promo_density,z_dm,z_promo,score` |
| `segments.json` | JSON | `[{start,end,peak_score}]` |
| `ground_truth.csv` | CSV | `start,end`（秒，与 rel_t 同轴） |

促单模式库（`promo_detect.py` 内，正则+权重）：上链接/上车(3.0)、3,2,1/三二一(2.5)、
最后X单(3.0)、秒杀(2.5)、拍下(1.5)、库存紧张(2.0)、倒计时(2.0)、宝宝们(0.8)、
手快有(2.0)、错过没有(2.0)、下单(1.2) 等 18 条，可继续扩充。

## 已知坑（冒烟测试实录）

- **B 站弹幕风控（2024+）**：裸连 `broadcastlv.chat.bilibili.com` 会被秒断
  （join 后 connection lost）。必须先 ① `x/frontend/finger/spi` 领匿名 buvid3，
  ② 用 **WBI 签名**（nav 接口拿 img_key/sub_key + mixin 表 md5）调
  `getDanmuInfo` 拿 token 和 host_list，③ ws 握手 **Cookie 带 buvid3**、
  join payload 带 `key=token`。`danmaku.py` 已内置此流程。
- yt-dlp 对 B 站直播的 fmp4/m3u8 格式会因安全策略拒绝，需 `-f best[ext=flv]`。
- faster-whisper base 模型首次下载可能超时（HF 直连慢）；`asr.py` 失败时退出码 2，
  `highlight_score.py` 自动降级 w2=0 纯弹幕信号。
- 评估口径注意：预测片段 pad 后与 GT 的 IoU 可能恰好 = 0.5（严格 > 才算命中），
  pad 已调小至 5s；调参时留意边界。

## 冒烟测试状态（见 results/）

- 合成 demo：端到端 F1 = 1.0（3/3 高光全中），`results/demo_timeline.png`
- 真实数据：B 站房间 545068（LPL 解说，92 万人气）抓取 270s / **718 条弹幕**，
  yt-dlp 录制 90s / 11.5MB FLV，faster-whisper(tiny) 转录 59 句带词级时间戳。
  真实弹幕跑融合打分产出候选片段，`results/real_timeline.png`。
  （注：真实弹幕与录像为两次分别采集，时间轴未对齐，仅验证各环节可用；
  正式实验 A 需同场同时采集。）

## 未来扩展

- 信号：礼物/SC 事件（ws 协议同一通道，cmd=SEND_GIFT/SUPER_CHAT_MESSAGE）、
  弹幕情感/关键词密度、音频能量（RMS）峰值。
- 模型：阈值/权重用标注数据做逻辑回归或 GBDT 学习，替代手工加权。
- 平台：抖音/淘宝直播（无公开接口，需第三方聚合或 OCR 兜底）。
- 工程：`danmaku.py + recorder.py` 合一的采集守护进程，自动对时、断线重录。
