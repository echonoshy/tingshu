#!/bin/bash

python -m tools.webui \
    --llama-checkpoint-path "weights" \
    --decoder-checkpoint-path "weights/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq \
    --compile