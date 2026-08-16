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

def download(url_or_bv, outdir):
    """B站下载（复用 bilibili-downloader skill 脚本，360p 快）。"""
    print(f"⬇️  下载 {url_or_bv} → 360p...", flush=True)
    r = run([PY, BILI, "download", url_or_bv, "--quality", "360", "-o", outdir])
    if r.returncode != 0:
        raise SystemExit(f"下载失败: {r.stderr[-300:]}")
    mp4s = sorted(glob.glob(os.path.join(outdir, "*.mp4")), key=os.path.getmtime)
    if not mp4s:
        raise SystemExit("下载后未找到 mp4")
    return mp4s[-1]

def analyze(video_path, workdir):
    """AVIS encode → avis_dir。"""
    out = os.path.join(workdir, "avis_out")
    os.makedirs(out, exist_ok=True)
    r = run([PY, AVIS, "encode", video_path, "-o", out, "--obj-tracks", "--asr-model", "tiny"])
    if r.returncode != 0:
        raise SystemExit(f"encode 失败: {r.stderr[-400:]}")
    stem = os.path.splitext(os.path.basename(video_path))[0] + "_avis"
    return os.path.join(out, stem)

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

def main():
    ap = argparse.ArgumentParser(description="L0 一键视频理解")
    ap.add_argument("target", help="B站 URL / BV 号 / 本地视频路径")
    ap.add_argument("--ask", action="append", default=[], help="自定义问题（可多次）")
    ap.add_argument("--no-download", action="store_true", help="target 为本地文件，跳过下载")
    ap.add_argument("--workdir", default="/tmp/avis_l0", help="工作目录")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.workdir, exist_ok=True)

    # 1. 获取视频文件
    if args.no_download:
        video_path = args.target
        print(f"📂 本地文件: {video_path}")
    else:
        video_path = download(args.target, args.workdir)
    print(f"🎬 视频: {os.path.basename(video_path)}", flush=True)

    # 2. AVIS 分析
    print("🔍 AVIS 分析（MV/ASR/场景/物体轨迹 + YOLO 标签）...", flush=True)
    avis_dir = analyze(video_path, args.workdir)

    # 3. 信息层 → 摘要
    p = run([PY, AVIS, "prompt", avis_dir]).stdout
    dur = float(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=duration", "-of", "csv=p=0", video_path]).stdout.strip() or 0)
    stats = token_stats(avis_dir, dur)

    sys_msg = ("你是视频内容分析助手。基于给定的信息层（语音转写+场景结构+运动对象轨迹），"
               "回答用户问题。直接给出答案，不要复述问题。信息不足时明确说明缺失什么，不要编造。")
    print("🤖 LLM 摘要 + 问答...", flush=True)
    qs = args.ask or DEFAULT_QUESTIONS
    # 单次调用：信息层只进一次，多问题合并（省 ~60% 输入 token）
    q_block = "\n\n".join(f"问题{i + 1}: {q}" for i, q in enumerate(qs))
    msg, usage = llm([{"role": "system", "content": sys_msg},
                      {"role": "user", "content": p + "\n\n" + q_block +
                       "\n\n请按 '问题N: 回答' 的格式逐条回答。"}], max_tokens=800)
    total_in = usage.get("prompt_tokens", 0)
    total_out = usage.get("completion_tokens", 0)
    # 拆分段落到 answers
    answers = []
    for i, q in enumerate(qs, 1):
        m = re.search(rf"问题{i}\s*[:：]\s*(.*?)(?=问题{i + 1}\s*[:：]|\Z)", msg, re.S)
        answers.append(m.group(1).strip() if m else f"(未能拆分) {msg[:200]}")
    for i, (q, a) in enumerate(zip(qs, answers), 1):
        print(f"\n❓ Q{i} {q}\n💬 {a}\n", flush=True)

    # 4. 汇总
    # ── 成本核算（2026-08-17 DeepSeek V4 调价后，人民币计费）──
    # 模型: deepseek-chat → 实际路由 deepseek-v4-flash
    # 价格（元/百万 token）：
    #   输入(缓存未命中) 1.5 (空闲) / 3.0 (高峰)  输入(缓存命中) 0.05/0.10
    #   输出              4.5 (空闲) / 9.0 (高峰)
    # 高峰时段: 9-12 / 14-18 点；其余为空闲时段
    # 注: 原实现误乘 7.2 汇率（DeepSeek 本币计费）已移除
    PRICE_IN_MISS = float(os.environ.get("AVIS_PRICE_IN", 1.5))
    PRICE_OUT = float(os.environ.get("AVIS_PRICE_OUT", 4.5))
    cost = (total_in / 1e6 * PRICE_IN_MISS + total_out / 1e6 * PRICE_OUT)
    elapsed = time.time() - t0
    print("=" * 70)
    print(f"✅ 完成 | 视频 {dur:.0f}s | 耗时 {elapsed:.0f}s | 信息层 {stats['info']} tok vs 原始逐帧 {stats['orig']:,} tok | token 压缩 {stats['save']}%")
    print(f"💵 LLM 成本 ≈ {cost:.4f} 元（v4-flash 空闲价；输入 {total_in} tok / 输出 {total_out} tok；'降本'为 token 压缩率，非账单对比）")

    # 保存报告
    report = os.path.join(args.workdir, "report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"# 视频理解报告\n\n- 视频: {os.path.basename(video_path)}\n- 时长: {dur:.0f}s\n")
        f.write(f"- 信息层: {stats['info']} tok (ASR {stats['asr']} + 轨迹 {stats['tracks']}×60 + 结构 150)\n")
        f.write(f"- 原始逐帧估算: {stats['orig']:,} tok → token 压缩 {stats['save']}%\n")
        f.write(f"- LLM 成本: {cost:.4f} 元（deepseek-v4-flash 空闲价；token 压缩率为信息层/逐帧估算 token 比，非账单对比）\n\n")
        for i, (q, a) in enumerate(zip(args.ask or DEFAULT_QUESTIONS, answers), 1):
            f.write(f"## Q{i} {q}\n\n{a}\n\n")
    print(f"📄 报告: {report}")

if __name__ == "__main__":
    main()
