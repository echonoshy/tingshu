#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-project --python 3.12 python scripts/install_weights.py "$@"
