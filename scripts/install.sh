#!/bin/bash

# Install model inference enviroment on Linux (GPU)

# 安装 pytorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# (Ubuntu / Debian 用户) 安装 sox + ffmpeg
apt update
apt install libsox-dev ffmpeg

# (Ubuntu / Debian 用户) 安装 pyaudio
apt install build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0
    
# 安装 fish-speech
pip3 install -e .[stable]