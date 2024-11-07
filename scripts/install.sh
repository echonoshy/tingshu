#!/bin/bash

# Install model inference environment on Linux (GPU)

# 有些系统权限/tmp目录需要手动设定权限
chmod 1777 /tmp

# 更新包管理器并安装系统依赖项
apt update -y && apt install -y \
    libsox-dev \
    ffmpeg \
    build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0

# 安装指定版本的 PyTorch 和相关依赖
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# 更新动态链接库缓存
ldconfig

# 安装 Python 项目及其可选的 `stable` 依赖项
pip3 install -e .[stable]
