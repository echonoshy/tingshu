#!/bin/bash
set -e

exec python tools/generate_web_data.py "$@"
