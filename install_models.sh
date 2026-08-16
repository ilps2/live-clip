#!/bin/bash
# video-understand 依赖安装脚本（全自动·国内镜像版）
# 用法：bash install_models.sh
# 说明：模型走国内镜像自动下载（hf-mirror.com + gh-proxy），无需手动上传/分享
set -e

MODELS_DIR="${AVIS_MODELS_DIR:-$HOME/Desktop/live-clip-repo/models}"

echo "==> 1/3 安装软件依赖（pip 清华镜像）"
python3 -m pip install -q faster-whisper ultralytics opencv-python huggingface_hub \
  -i https://pypi.tuna.tsinghua.edu.cn/simple || echo "  ⚠️ pip 装失败，请手动装（见 README）"

echo "==> 2/3 下载 whisper 模型 tiny + base（hf-mirror.com 国内镜像）"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python3 - << 'EOF'
import os
from huggingface_hub import snapshot_download
for m in ("Systran/faster-whisper-tiny", "Systran/faster-whisper-base"):
    p = snapshot_download(m)
    print(f"  ✅ {m} → {p}")
EOF

echo "==> 3/3 下载 yolov8n.pt（gh-proxy 镜像）"
mkdir -p "$MODELS_DIR"
curl -sL --max-time 120 "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" \
  -o "$MODELS_DIR/yolov8n.pt"
echo "  ✅ yolov8n.pt → $MODELS_DIR ($(du -h "$MODELS_DIR/yolov8n.pt" | cut -f1))"

echo ""
echo "✅ 全部完成！验证："
echo "  python3 -c \"from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu')\""
echo "  python3 -c \"from ultralytics import YOLO; YOLO('$MODELS_DIR/yolov8n.pt')\""
