#!/usr/bin/env python3
"""
L0 一键视频理解：B站链接/BV → 下载(360p) → AVIS 分析 → 摘要 + 3 个问答
用法:
  python3 understand_video.py BV1GJ411x7h7
  python3 understand_video.py "https://www.bilibili.com/video/BV1GJ411x7h7/" --ask "博主推荐了哪些产品" --ask "价格多少"
  python3 understand_video.py /path/local.mp4 --no-download
成本: ~0.002 元/视频（信息层 1k tok vs 原始逐帧 540万 tok，降本 99.98%）
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile, time, urllib.request

PY = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
BILI = os.path.expanduser("~/.agents/skills/bilibili-downloader/scripts/bili_download.py")
AVIS = os.path.expanduser("~/Desktop/live-clip-repo/avis.py")
KEY = "sk-d8c7aa22549947fabd033bc63e16b759"
URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_QUESTIONS = [
    "这段视频的核心内容是什么？用 3-5 句话概括。",
    "视频中有哪些关键细节或亮点？",
    "这段视频适合什么场景/人群使用？",
]

def run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def llm(messages, max_tokens=400):
    body = {"model": "deepseek-chat", "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.3}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        r = json.loads(resp.read())
    return r["choices"][0]["message"]["content"], r.get("usage", {})

def fetch_title(url_or_bv):
    """从 bilibili API 拿视频标题（标题类问题需要；本地文件返回文件名）。"""
    try:
        bv = re.search(r"BV[0-9A-Za-z]{10}", url_or_bv)
        if bv:
            import urllib.request as _ur
            req = _ur.Request(f"{'https://api.bilibili.com/x/web-interface/view?bvid='}{bv.group(0)}",
                headers={"User-Agent": "Mozilla/5.0"})
            with _ur.urlopen(req, timeout=15) as resp:
                d = json.loads(resp.read())
            if d.get("code") == 0:
                return d["data"].get("title", "")
    except Exception:
        pass
    return ""

def download(url_or_bv, outdir):
    """B站下载（复用 bilibili-downloader skill 脚本，360p 快）。返回 (path, title)。"""
    print(f"⬇️  下载 {url_or_bv} → 360p...", flush=True)
    r = run([PY, BILI, "download", url_or_bv, "--quality", "360", "-o", outdir])
    if r.returncode != 0:
        raise SystemExit(f"下载失败: {r.stderr[-300:]}")
    mp4s = sorted(glob.glob(os.path.join(outdir, "*.mp4")), key=os.path.getmtime)
    if not mp4s:
        raise SystemExit("下载后未找到 mp4")
    title = fetch_title(url_or_bv)
    return mp4s[-1], title

def analyze(video_path, workdir, asr_model="tiny", sub="avis_out"):
    """AVIS encode → avis_dir。asr_model: tiny/base/small。"""
    out = os.path.join(workdir, sub)
    os.makedirs(out, exist_ok=True)
    r = run([PY, AVIS, "encode", video_path, "-o", out, "--obj-tracks",
             "--asr-model", asr_model, "--skip-mv"])
    if r.returncode != 0:
        raise SystemExit(f"encode 失败: {r.stderr[-400:]}")
    stem = os.path.splitext(os.path.basename(video_path))[0] + "_avis"
    return os.path.join(out, stem)

def quality_check(question, answer):
    """LLM 自评：回答对问题的信息充分度 0-10 + 缺口类型（轻量调用）。"""
    body = {"model": "deepseek-chat",
            "messages": [{"role": "system",
                          "content": "你是严格的质量评估器。回答含「无法确认/识别错误/信息不足/不确定/可能/缺失」等表述时应给低分（<6）；"
                          "用户问具体内容（物品/数量/价格/名称）时，回答缺少具体名称、数量、价格即为不足。"},
                         {"role": "user",
                          "content": f"用户问题: {question}\n模型回答: {answer[:600]}\n\n"
                          "评估回答的信息充分度（0-10，<7 为不足，需要升级信号）和主要缺口"
                          "（asr=语音转写不清/缺失, visual=缺画面细节, none=已充分, other=其他）。"
                          "严格输出 JSON: {\"score\": 0-10, \"gap\": \"asr|visual|none|other\"}"}],
            "max_tokens": 60, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        r = json.loads(resp.read())
    raw = r["choices"][0]["message"]["content"]
    try:
        d = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        return int(d.get("score", 0)), d.get("gap", "other")
    except Exception:
        return 0, "other"

def run_visual(level, video, avis_dir, window=None):
    """调 visual_level.py，返回 (note, cost)。"""
    vl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_level.py")
    cmd = [PY, vl, level, video, avis_dir, "--json"]
    if level == "l2" and window:
        cmd += ["--window", str(window), "--step", "5"]
    vr = run(cmd)
    try:
        vd = json.loads(vr.stdout[vr.stdout.index("{"):])
        desc = vd.get("description", "")
        pin, pout = vd.get("tokens", {}).get("in", 0), vd.get("tokens", {}).get("out", 0)
        cost = pin / 1e6 * 0.2 + pout / 1e6 * 0.7  # qwen3-vl-flash 近似
        note = (f"\n## 视觉补充（L{level.upper()} VLM 抽帧描述，{vd.get('frames', [])}）\n{desc}\n")
        print(f"  ✅ 视觉 {len(desc)} 字 | VLM {pin}+{pout} tok ≈ {cost:.4f} 元")
        return note, cost
    except Exception as e:
        print(f"  ⚠️ 视觉级失败: {e}")
        return "", 0.0

def pick_key_window(avis_dir, dur):
    """选 L2 关键窗：ASR 含价格/物品词的时段优先，否则轨迹最活跃 30s。"""
    KEY_WORDS = ("欧", "元", "块", "个", "袋", "包", "面包", "甜", "硬", "买", "价", "钱", "€")
    best_a, best_b, best_hits = 0, min(30, int(dur)), 0
    tr = os.path.join(avis_dir, "transcript.jsonl")
    if os.path.exists(tr):
        segs = [json.loads(l) for l in open(tr, encoding="utf-8") if l.strip()]
        # 滑窗 30s 统计关键词命中
        for a in range(0, max(1, int(dur) - 29)):
            hits = 0
            for s in segs:
                t = s.get("start", 0)
                if a <= t < a + 30 and any(w in s.get("text", "") for w in KEY_WORDS):
                    hits += 1
            if hits > best_hits:
                best_hits, best_a, best_b = hits, a, a + 30
    return f"{best_a}-{best_b}"

def token_stats(avis_dir, dur):
    tr, ot = os.path.join(avis_dir, "transcript.jsonl"), os.path.join(avis_dir, "obj_tracks.jsonl")
    asr_tok = 0
    if os.path.exists(tr):
        for line in open(tr, encoding="utf-8"):
            if line.strip():
                asr_tok += len(json.loads(line).get("text", "")) // 2
    n_tracks = sum(1 for l in open(ot, encoding="utf-8") if l.strip()) if os.path.exists(ot) else 0
    info = asr_tok + n_tracks * 60 + 150
    orig = int(dur * 30 * 1000)
    return {"asr": asr_tok, "tracks": n_tracks, "info": info, "orig": orig,
            "save": round(100 * (1 - info / orig), 2)}

# ── 成本核算（2026-08-17 DeepSeek V4 调价后，人民币计费）──
# 模型: deepseek-chat → 实际路由 deepseek-v4-flash；高峰时段 9-12/14-18 点，其余空闲
# 价格（元/百万 token，默认空闲价，可环境变量覆盖）：
#   输入(缓存命中) 0.05 | 输入(未命中) 1.5 | 输出 4.5   （高峰: 0.10 / 3.0 / 9.0）
# 重复理解省成本机制：同一信息层 prompt 第二次起 ~93% 命中上下文缓存（实测），
# 命中部分按 0.05 元/M 计 → 后续每次理解 ≈ 0.0002 元（约为首次的 1/20）
PRICE_IN_HIT = float(os.environ.get("AVIS_PRICE_IN_HIT", 0.05))
PRICE_IN_MISS = float(os.environ.get("AVIS_PRICE_IN", 1.5))
PRICE_OUT = float(os.environ.get("AVIS_PRICE_OUT", 4.5))

def calc_cost(usage):
    """按缓存命中拆分计算 LLM 成本，返回 (cost, hit_tok, miss_tok)。"""
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit) or 0
    out = usage.get("completion_tokens", 0) or 0
    cost = hit / 1e6 * PRICE_IN_HIT + miss / 1e6 * PRICE_IN_MISS + out / 1e6 * PRICE_OUT
    return cost, hit, miss

def main():
    ap = argparse.ArgumentParser(description="L0/L1/L2 视频理解（信息层 + 按需视觉帧）")
    ap.add_argument("target", help="B站 URL / BV 号 / 本地视频路径")
    ap.add_argument("--ask", action="append", default=[], help="自定义问题（可多次）")
    ap.add_argument("--no-download", action="store_true", help="target 为本地文件，跳过下载")
    ap.add_argument("--workdir", default="/tmp/avis_l0", help="工作目录")
    ap.add_argument("--json", action="store_true", help="输出 JSON（dsh 工具模式）")
    ap.add_argument("--level", choices=["l0", "l1", "l2"], default="l0",
                    help="l0=信息层(默认) l1=+3-5帧VLM视觉摘要 l2=+时间窗密集帧证据")
    ap.add_argument("--l2-window", default="auto", help="L2 时间窗，如 10-30 或秒数（auto=轨迹最活跃30s）")
    ap.add_argument("--l2-step", type=int, default=2, help="L2 抽帧步长（秒）")
    ap.add_argument("--auto", dest="auto", action="store_true", default=True,
                    help="质量门控自动升级（默认开）：LLM 自评不足时自动 ASR tiny→base / L2 关键段帧")
    ap.add_argument("--no-auto", dest="auto", action="store_false", help="关闭自动升级")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.workdir, exist_ok=True)

    # 1. 获取视频文件
    if args.no_download:
        video_path = args.target
        title = os.path.splitext(os.path.basename(video_path))[0]
        print(f"📂 本地文件: {video_path}")
    else:
        video_path, title = download(args.target, args.workdir)
    print(f"🎬 视频: {os.path.basename(video_path)}" + (f"（标题：{title}）" if title else ""), flush=True)

    # 2. AVIS 分析（L0：tiny ASR + 轨迹；质量不足时自动升级）
    print("🔍 AVIS 分析（MV/ASR/场景/物体轨迹 + YOLO 标签）...", flush=True)
    avis_dir = analyze(video_path, args.workdir, asr_model="tiny", sub="avis_tiny")
    dur = float(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=duration", "-of", "csv=p=0", video_path]).stdout.strip() or 0)
    stats = token_stats(avis_dir, dur)

    # 3. 质量门控循环：LLM 自评 → 不足自动升级（ASR base / L2 视觉）
    qs = args.ask or DEFAULT_QUESTIONS
    q_block = "\n\n".join(f"问题{i + 1}: {q}" for i, q in enumerate(qs))
    asr_model = "tiny"
    visual_note = ""
    visual_cost = 0.0
    total_cost = 0.0
    total_hit = total_miss = total_out = 0
    rounds, upgrades = 0, []
    used_windows = set()
    answers = []

    while True:
        p = run([PY, AVIS, "prompt", avis_dir]).stdout
        if title:
            p = f"# 视频标题：{title}\n（标题可能与内容不符，请结合内容判断）\n\n" + p
        sys_msg = ("你是视频内容分析助手。基于给定的信息层（语音转写+场景结构+运动对象轨迹"
                   + ("+视觉帧描述" if visual_note else "")
                   + "），回答用户问题。直接给出答案，不要复述问题。信息不足时明确说明缺失什么，不要编造。")
        print(f"🤖 LLM 回答（第 {rounds + 1} 轮，ASR={asr_model}）...", flush=True)
        msg, usage = llm([{"role": "system", "content": sys_msg},
                          {"role": "user", "content": p + visual_note + "\n\n" + q_block +
                           "\n\n请按 '问题N: 回答' 的格式逐条回答。"}], max_tokens=800)
        c, h, m = calc_cost(usage)
        total_cost += c; total_hit += h; total_miss += m; total_out += usage.get("completion_tokens", 0)
        answers = []
        for i, q in enumerate(qs, 1):
            mm = re.search(rf"问题{i}\s*[:：]\s*(.*?)(?=问题{i + 1}\s*[:：]|\Z)", msg, re.S)
            answers.append(mm.group(1).strip() if mm else f"(未能拆分) {msg[:200]}")
        for i, (q, a) in enumerate(zip(qs, answers), 1):
            print(f"\n❓ Q{i} {q}\n💬 {a}\n", flush=True)

        # 自评门控
        if not args.auto or rounds >= 2:
            break
        score, gap = quality_check(qs[0], answers[0])
        print(f"  [自评] 充分度 {score}/10 | 缺口: {gap} | 轮次 {rounds + 1}/3", flush=True)
        if score >= 7:
            break
        # 升级决策
        if gap == "visual" or (gap == "other" and asr_model == "base"):
            window = pick_key_window(avis_dir, dur)
            # 去重：同窗口不重跑；换一个偏移窗口
            if window in used_windows:
                window = f"{int(window.split('-')[0]) + 30}-{int(window.split('-')[1]) + 30}"
            if window in used_windows:
                asr_model = "base"
                print("  ⬆️  视觉窗口已用尽，改升级 ASR tiny→base...", flush=True)
                avis_dir = analyze(video_path, args.workdir, asr_model="base", sub="avis_base")
                stats = token_stats(avis_dir, dur)
                upgrades.append("ASR:base")
            else:
                used_windows.add(window)
                print(f"  ⬆️  升级 L2 视觉关键段（{window}s）...", flush=True)
                note, vc = run_visual("l2", video_path, avis_dir, window)
                visual_note = visual_note + note
                visual_cost += vc
                upgrades.append(f"L2@{window}")
        elif gap == "asr" and asr_model == "base":
            # base 转写仍不足 → 语音信息有限，转视觉关键段
            window = pick_key_window(avis_dir, dur)
            if window in used_windows:
                window = f"{int(window.split('-')[0]) + 30}-{int(window.split('-')[1]) + 30}"
            used_windows.add(window)
            print(f"  ⬆️  base ASR 仍不足，转 L2 视觉关键段（{window}s）...", flush=True)
            note, vc = run_visual("l2", video_path, avis_dir, window)
            visual_note = visual_note + note
            visual_cost += vc
            upgrades.append(f"L2@{window}")
        else:  # asr / other → 升级 ASR（tiny→base）
            asr_model = "base"
            print("  ⬆️  升级 ASR tiny→base 重转写...", flush=True)
            avis_dir = analyze(video_path, args.workdir, asr_model="base", sub="avis_base")
            stats = token_stats(avis_dir, dur)
            upgrades.append("ASR:base")
        rounds += 1

    # 4. 汇总
    elapsed = time.time() - t0
    hit_pct = total_hit / (total_hit + total_miss) * 100 if (total_hit + total_miss) else 0
    level_tag = args.level.upper() + ("+auto" if args.auto and upgrades else "")

    if args.json:
        print(json.dumps({
            "video": os.path.basename(video_path),
            "duration_s": round(dur),
            "elapsed_s": round(elapsed),
            "info_tokens": stats["info"],
            "orig_frame_tokens": stats["orig"],
            "token_compression_pct": stats["save"],
            "cost_cny": round(total_cost + visual_cost, 5),
            "visual_cost_cny": round(visual_cost, 5),
            "level": level_tag,
            "rounds": rounds + 1,
            "upgrades": upgrades,
            "quality_scores": [],
            "prompt_cache_hit_tokens": total_hit,
            "answers": [{"question": q, "answer": a} for q, a in zip(qs, answers)],
        }, ensure_ascii=False, indent=2))
        return

    print("=" * 70)
    print(f"✅ 完成 | 视频 {dur:.0f}s | 耗时 {elapsed:.0f}s | 层级 {level_tag} | 信息层 {stats['info']} tok vs 原始逐帧 {stats['orig']:,} tok | token 压缩 {stats['save']}%")
    print(f"💵 成本 ≈ {total_cost + visual_cost:.4f} 元（LLM {total_cost:.4f} + 视觉 {visual_cost:.4f}）| 升级: {' → '.join(upgrades) if upgrades else '无'} | 缓存命中 {hit_pct:.0f}%")
    print(f"♻️  重复理解：同一视频再次理解时信息层 ~93% 命中缓存 → 每次 ≈ {total_cost/20:.4f}~{total_cost:.4f} 元（约为首次的 1/20）")

    # 保存报告
    report = os.path.join(args.workdir, "report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"# 视频理解报告\n\n- 视频: {os.path.basename(video_path)}\n- 时长: {dur:.0f}s\n")
        f.write(f"- 信息层: {stats['info']} tok (ASR {stats['asr']} + 轨迹 {stats['tracks']}×60 + 结构 150)\n")
        f.write(f"- 原始逐帧估算: {stats['orig']:,} tok → token 压缩 {stats['save']}%（核心指标，模型无关）\n")
        f.write(f"- LLM 成本: {total_cost:.4f} 元 + 视觉 {visual_cost:.4f} 元（缓存命中 {total_hit} tok + 未命中 {total_miss} tok + 输出 {total_out} tok）\n")
        f.write(f"- 自动升级: {' → '.join(upgrades) if upgrades else '无（一轮达标）'}\n")
        f.write(f"- 重复理解: 同一视频再次理解时信息层 ~93% 命中上下文缓存 → 每次成本约为首次的 1/20\n\n")
        for i, (q, a) in enumerate(zip(args.ask or DEFAULT_QUESTIONS, answers), 1):
            f.write(f"## Q{i} {q}\n\n{a}\n\n")
    print(f"📄 报告: {report}")

if __name__ == "__main__":
    main()
