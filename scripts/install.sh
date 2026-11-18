#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null; then
    echo "请先安装 uv，再运行此脚本。" >&2
    exit 1
fi
if ! command -v ffmpeg >/dev/null; then
    echo "请先安装 FFmpeg（Ubuntu: apt install ffmpeg；macOS: brew install ffmpeg）。" >&2
    exit 1
fi

uv sync --locked
