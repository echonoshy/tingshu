#!/bin/bash

# Ensure the script exits on error
set -e

RESOURCES_DIR="web/resources/leijun"

# Run the Python script with the resources directory as argument
python tools/generate_web_data.py --resources_dir "$RESOURCES_DIR"
