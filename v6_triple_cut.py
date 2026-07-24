#!/usr/bin/env python3
"""
v6_triple_cut.py — 三人群定向切片

每条切片: 品名 + 价格 + 一个定向卖点
    切片1 → price 人群: 性价比/优惠/补贴
    切片2 → quality 人群: 品牌/成分/专利
    切片3 → effect 人群: 功效/效果/对比

LLM 语义选段 → 连续裁切 → 独立输出
"""
import argparse, json, subprocess, sys, os
from pathlib import Path

def call_llm(prompt, api_key, model="deepseek-v4-flash", max_tokens=8192):
    import urllib.request
    data = json.dumps({
        "model": model, "temperature": 0.1, "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "直播切片专家。只输出JSON。"},
            {"role": "user", "content": prompt}
        ]
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
        content = body["choices"][0]["message"]["content"]
        if not content or not content.strip():
            print(f"  LLM returned empty. Full response: {json.dumps(body, ensure_ascii=False)[:500]}")
            sys.exit(1)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to salvage truncated JSON
            print(f"  ⚠️ JSON truncated, attempting recovery...")
            # Try closing braces
            for attempt in [content + "\n  ]\n}", content + "}]}"]:
                try:
                    return json.loads(attempt)
                except:
                    pass
            print(f"  LLM returned non-JSON: {content[:200]}...")
            sys.exit(1)


def build_prompt(transcript, max_chars=25000):
    lines = "\n".join(f"[{ts:.1f}s] {txt}" for ts, txt in transcript[:int(max_chars/50)])
    
    return f"""分析以下直播带货ASR转录，找到讲品段落。

对第一个产品，从ASR中找出**三段**，分别对应三类人群：

1. price (价格敏感): 讲价格/优惠/补贴/性价比的段落
2. quality (品质导向): 讲品牌/成分/专利/质感的段落  
3. effect (效果焦虑): 讲功效/效果/前后对比/改善的段落

每段必须包含: 品名介绍 + 价格提及 + 定向卖点
每段30-50秒，从ASR中取连续时间。

输出JSON:
{{
  "product_name": "品名",
  "price": "价格",
  "clips": [
    {{
      "target": "price",
      "start": 秒数,
      "end": 秒数,
      "hook": "吸引该人群的一句话(≤15字)"
    }},
    {{
      "target": "quality",
      "start": 秒数,
      "end": 秒数,
      "hook": "吸引该人群的一句话(≤15字)"
    }},
    {{
      "target": "effect",
      "start": 秒数,
      "end": 秒数,
      "hook": "吸引该人群的一句话(≤15字)"
    }}
  ]
}}

⚠️ 三个时间段不重叠，每个30-50秒。

ASR:
{lines}"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--out-dir", default="./v6_out")
    p.add_argument("--api-key")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--min-dur", type=int, default=25)
    p.add_argument("--max-dur", type=int, default=55)
    p.add_argument("--padding", type=int, default=3)
    p.add_argument("--skip-llm", action="store_true")
    args = p.parse_args()
    
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not args.skip_llm and not api_key:
        print("ERROR: need API key"); sys.exit(1)
    
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("[1/2] 加载 ASR + LLM...")
    raw = [json.loads(l) for l in open(args.transcript)]
    transcript = [(s["start"], s["text"]) for s in raw]
    total_dur = transcript[-1][0]
    print(f"  {len(transcript)} 句, {total_dur:.0f}s")
    
    if args.skip_llm:
        llm_path = out_dir / "llm_v6.json" if (out_dir / "llm_v6.json").exists() else Path("/tmp/v6_out/llm_v6.json")
        result = json.loads(llm_path.read_text()) if llm_path.exists() else None
        if not result:
            print("ERROR: no cached LLM"); sys.exit(1)
    else:
        prompt = build_prompt(transcript)
        print(f"  Prompt: {len(prompt)} chars → {args.model}")
        result = call_llm(prompt, api_key, model=args.model)
        with open(out_dir / "llm_v6.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    product = result.get("product_name", "未知")
    price = result.get("price", "?")
    clips = result.get("clips", [])
    
    print(f"  {product} — {price} | {len(clips)}条")
    for c in clips:
        print(f"    [{c['target']:7s}] {c['start']:.0f}-{c['end']:.0f}s | {c['hook']}")
    
    if not clips:
        print("⚠️ 无切片"); return
    
    print(f"\n[2/2] 裁切...")
    
    results = []
    for c in clips:
        target = c["target"]
        start = max(0, c["start"] - args.padding)
        end = min(total_dur, c["end"] + args.padding)
        dur = end - start
        
        if dur < args.min_dur:
            end = min(total_dur, start + args.min_dur)
            dur = end - start
        if dur > args.max_dur:
            end = start + args.max_dur
            dur = end - start
        
        name = f"{product}_{target}_{int(start)}-{int(end)}s.mp4"
        out_path = out_dir / name
        
        r = subprocess.run(
            f"ffmpeg -y -v error -ss {start} -to {end} "
            f"-i '{args.video}' "
            f"-c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k "
            f"'{out_path}'",
            shell=True, capture_output=True, text=True)
        
        if r.returncode != 0:
            print(f"  ⚠️ {name}: {r.stderr[-80:]}")
            continue
        
        size = out_path.stat().st_size / 1024
        results.append((out_path, c))
        print(f"  ✅ {name} ({dur:.0f}s, {size:.0f}KB) — {c['hook']}")
    
    # Save copywriting sheet
    copy_path = out_dir / "copywriting.txt"
    target_names = {"price": "💰 价格敏感", "quality": "🔬 品质导向", "effect": "✨ 效果焦虑"}
    with open(copy_path, "w") as f:
        f.write(f"# {product} — {price}\n\n")
        for out_path, c in results:
            t = c["target"]
            f.write(f"## {target_names.get(t, t)}\n")
            f.write(f"文件: {out_path.name}\n")
            f.write(f"时间: {c['start']:.0f}s - {c['end']:.0f}s\n")
            f.write(f"Hook: {c['hook']}\n")
            f.write(f"字幕建议: {c['hook']}\n\n")
    
    print(f"\n{'=' * 60}")
    print(f"完成: {len(results)} 条 | 文案: {copy_path.name}")
    for out_path, _ in results:
        print(f"  {out_path.name}")
    print(f"输出: {out_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
