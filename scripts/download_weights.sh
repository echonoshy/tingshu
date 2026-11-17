#!/bin/bash

# 定义下载 URL 和目标文件名
URL="https://github.com/echonoshy/tingshu/releases/download/v1.0.0/weights.tar.gz"
FILE="weights.tar.gz"

# 下载文件
echo "Downloading weights from $URL..."
wget -O $FILE $URL

# 检查下载是否成功
if [ $? -ne 0 ]; then
  echo "Download failed. Please check the URL or your network connection."
  exit 1
fi

# 解压文件
echo "Extracting $FILE..."
tar -xzvf $FILE

# 检查解压是否成功
if [ $? -ne 0 ]; then
  echo "Extraction failed. Please check the downloaded file."
  exit 1
fi

# 删除下载的压缩文件
rm $FILE
echo "Download and extraction completed successfully!"
