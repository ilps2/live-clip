# live-clip

**Low-Cost Video Understanding & Intelligent Clipping for E-commerce Livestreams**

> Independent Research · v1.0 · 2026-07-23  
> All experiments are reproducible. Code and data included.

---

## TL;DR

We prove that **a full e-commerce livestream can be understood and clipped at <5% the cost of uniform multimodal LLM processing** — without using a single LLM call for detection, deduplication, boundary finding, or editing decisions.

The key insight: **cheap, free-tier signals (H.264 codec motion vectors, ASR transcripts, platform metadata) are sufficient for >90% of the work if you fuse them correctly.** The LLM is reserved for final confirmation only on selected segments.

---

## What This Repo Contains

| Directory | What | Status |
|---|---|---|
| `docs/` | Full research paper (Chinese + English abstract) | ✅ |
| `motion-signature/` | Codec MV extraction, motion signature classification, temporal pairing experiments | ✅ 6 experiments |
| `livestream-highlight/` | End-to-end highlight detection pipeline (danmaku capture, ASR, promo detection, scoring) | ✅ |
| `dynamic-precision-voxel/` | Low-resolution temporal classification (SVM 27% vs LSTM 87%+) | ✅ |
| `motion-signature/real_fusion/` | Sample 1 & 2 full fusion data, V1–V3 metrics, final clips | ✅ |

---

## The Framework: Hierarchical Signals × Hierarchical Compute

```
L0: Platform metadata (danmaku, viewer count, product-link events)  Cost ≈ 0
L1: Audio stream (VAD, ASR, promo-speech patterns)                  Cost ≈ pennies/hr
L2: Screen template events (price popups, layout changes)           Cost = low
L3: Codec motion vector field (exported from H.264 bitstream!)      Cost ≈ 0  ← key insight
L4: Multimodal LLM (only on filtered segments)                      Cost = high but sparse
```

**Design principle: compute follows information density. Any layer can independently degrade or go absent.**

---

## Three Novel Mechanisms

### 1. Spatiotemporal Dynamic Precision
Motion vectors at 1fps are nearly lossless for second-scale temporal patterns. But sub-second trajectories (e.g., a 3-second back-and-forth product display) need 5fps to recover. The solution: **mixed sampling** — 1fps by default, auto-boost to 5fps at motion peaks. The original "dynamic precision voxel" theory extends from X-Y to X-Y-T.

### 2. Segment-Level Signal Convergence
Individual cheap signals have high false-positive rates — but their noise is independent. True highlight events drive **all three signals simultaneously**. The key: convergence must be judged at the **segment level**, not frame-by-frame, because signal arrival times are naturally staggered within an event (you show the product *before* you push the sale).

### 3. Semantic Time-Compression Editing
Live commerce follows a serial structure: display → explain → push. We can restructure this non-linearly: overlay the *best product-display visuals* with *later sales-push audio* on the same timeline. Result: **5.6× information density** compared to the original flow. Applicable when no visible face/lip-sync is present in the frame.

---

## Key Experimental Results

### Motion Vectors Are Free
PyAV `export_mvs` extracts H.264 encoder motion vectors directly from the bitstream — no pixel decoding, no optical flow computation needed. 21,956 frames from a real livestream exported successfully. **This is the entire cost story's foundation.**

### Motion Signatures Distinguish Events
8-dimensional spatial signatures + RandomForest achieve 100% on synthetic 4-class events (camera-move / product-display / gesture / static). Real livestream motion fields are visually clean and class-distinguishable.

### Temporal Dimension Is Necessary
On 4 event pairs that are *identical in single-frame statistics* but differ only in temporal structure, GRU sequence classification achieves 0.93–1.00 vs single-frame ~0.62. **1fps is sufficient for second-scale patterns; sub-second trajectories require the 5fps boost.**

### Real-World Highlight Detection
| Sample | Style | V1 F1 | V2 F1 (rules frozen) |
|--------|-------|-------|----------------------|
| Sample 1 | Chatty-style (5.3 min) | 0.33 | 1.00* |
| Sample 2 | Hard-sell beauty (12.2 min) | — | Event-level P/R = 1.0 (3/3) |

*Sample 1 n=1 with 3 hand-tuned hyperparameters. Sample 2 used **pre-registered frozen rules** — predictions locked before annotation.*

### Editing Output (Sample 2)
| | Original Cycle | Faithful Clip | Compressed Clip |
|---|---|---|---|
| Duration | ~250s | 58.3s | 44.8s |
| Info density | 1× | ~4.3× | **~5.6×** |

The compressed clip overlays product-demo visuals with time-shifted promotional audio. User adjudication confirmed effectiveness for non-face scenarios.

---

## Cost Estimate

| Approach | Per 4-hour livestream |
|---|---|
| Uniform VLM sampling (1fps) | $20–100 |
| **This framework** (MV free + ASR $0.50 + CPU detection ≈ $0 + L4 on 10–20 min only $1–3) | **$2–5** |

**10–30× reduction.** And in all experiments above, L4 (multimodal LLM) was called **zero times** — detection, deduplication, boundary finding, and editing routing were entirely handled by cheap signals.

---

## Limitations (Honest)

1. Only 2 real samples tested (n=1 × 2); V2 rules contain 3 hand-tuned hyperparameters — generalization evidence is thin
2. Motion signature real-world accuracy is lower than synthetic (product class barely triggered in sample 2)
3. Compressed clip audio-seam naturalness tested by single user only
4. Danmaku (L0) channel not used in real experiments due to platform API limitations
5. All classifiers validated on synthetic data at high metrics; real data conclusions are qualitative + small-sample quantitative

---

## Open Problems

1. V2 rule stability across 5–10 diverse samples (overfitting defense is priority #1)
2. Mixed face/non-face routing: auto-switch between faithful and compressed editing within one event
3. Full verification on face-free livestream categories (jewelry, food, nail art)
4. Digital human livestream signal degradation: how to compensate when danmaku is weak and speech is scripted
5. Comment OCR (in-frame danmaku from screen recording) to recover L0 signal
6. Formal definition of a "cost-per-semantic-understanding" evaluation benchmark
7. ASR typo-robust semantic deduplication (embedding-based)

---

## Quick Start

```bash
# Install dependencies
pip install av scikit-learn faster-whisper numpy matplotlib

# Run motion signature extraction on any MP4
cd motion-signature
python extract_mv.py --input your_video.mp4 --out results/mv.npz
python visualize.py  # generates quiver plots + timeline

# Run highlight detection pipeline (Bilibili livestream)
cd livestream-highlight
python danmaku.py --room <ROOM_ID> --duration 3600 --out results/danmaku.jsonl
python recorder.py --url https://live.bilibili.com/<ROOM_ID> --duration 3600 --out results/live.flv
python asr.py --video results/live.flv --out results/transcript.jsonl
python highlight_score.py --danmaku results/danmaku.jsonl --promo results/promo_events.jsonl
```

---

## Related Work & Differentiation

- **Pireel / CapCut (transcript-driven editing)**: Single audio signal, editing-UI focused. This work: multi-signal hierarchical, understanding-oriented, extends to automatic clip generation.
- **Video LLMs (GPT-4o / Gemini)**: Cost structure prohibits "understand every livestream." This framework is complementary — it's the *selector* that decides when to call them.
- **Event cameras / video token compression research**: Share the "compute follows information" meta-principle. This work contributes a domain-specific (e-commerce livestream) end-to-end engineering closure.

---

## Author

**Chu Li (楚离)** — Independent Researcher

*"I derived the core principle — compute should follow information density — from first principles, without prior exposure to related literature. The experiments confirmed the intuition was directionally correct. I'm now working to turn this into a reproducible, openly-documented research program."*

- GitHub: [@ilps2](https://github.com/ilps2)
- Project Home: https://github.com/ilps2/live-clip

---

## Citation

If you find this work useful:

```bibtex
@techreport{chu2026liveclip,
  title={Low-Cost Video Understanding and Intelligent Clipping for E-commerce Livestreams: From Hierarchical Signals to Semantic Time-Compression Editing},
  author={Chu Li},
  year={2026},
  url={https://github.com/ilps2/live-clip}
}
```

## License

MIT
