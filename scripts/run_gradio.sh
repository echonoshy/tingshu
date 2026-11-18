#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Gradio health checks must reach its local server even behind an HTTP proxy.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1"
exec uv run --locked python -m tools.webui "$@"
