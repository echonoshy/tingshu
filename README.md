# Tingshu (听舒)
> AI 合成有声书 | AI-narrated audiobooks

[![GitHub stars](https://img.shields.io/github/stars/echonoshy/tingshu?style=social)](https://github.com/echonoshy/tingshu/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/echonoshy/tingshu?style=social)](https://github.com/echonoshy/tingshu/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/echonoshy/tingshu?style=social)](https://github.com/echonoshy/tingshu/watchers)
[![GitHub repo size](https://img.shields.io/github/repo-size/echonoshy/tingshu)](https://github.com/echonoshy/tingshu)
[![GitHub language count](https://img.shields.io/github/languages/count/echonoshy/tingshu)](https://github.com/echonoshy/tingshu)
[![GitHub top language](https://img.shields.io/github/languages/top/echonoshy/tingshu)](https://github.com/echonoshy/tingshu)
[![GitHub last commit](https://img.shields.io/github/last-commit/echonoshy/tingshu?color=red)](https://github.com/echonoshy/tingshu/commits/master)

## 项目预览 | Preview

[点击进入项目主页｜Home](https://echonoshy.github.io/tingshu/web)

![Web Interface Preview](assets/web-view.jpg)

## 快速开始 | Quick Start

直接打开[线上书架](https://echonoshy.github.io/tingshu/web/)即可收听已上架章节。已有音频为 AI 合成朗读，素材记录见[来源与授权说明](docs/media-sources.md)。

生成音频前准备 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和 FFmpeg。当前锁定环境使用 **Python 3.12**，适用于 Linux x86-64 和 Apple Silicon macOS 14+。有 CUDA 时启用 GPU，其他设备默认使用 CPU。

```bash
./scripts/install.sh
./scripts/download_weights.sh
./scripts/run_gradio.sh
```

下载脚本会校验 [v1.0.0 发布包](https://github.com/echonoshy/tingshu/releases/tag/v1.0.0)的大小和 SHA-256，并安装到 `weights/`。已有该压缩包时可用 `./scripts/download_weights.sh --archive /path/to/weights.tar.gz`，已有模型目录会保留。

批量生成需要将已获许可的 `reference.wav`、`reference.txt` 和章节文本放进所选书目目录。

```bash
./scripts/generate_web_data.sh --resources_dir web/resources/leijun
uv run --locked python -m unittest discover -s tests -v
```

## 项目优化 ｜ TODO
- [ ] 1. 使用新方案替代tts方式生成的录音缺乏情感，顿挫，听感不好问题。     
      Replace the TTS-generated recordings to address issues of lacking emotion, intonation, and poor listening experience.  

- [ ] 2. 更新web ui, 替换静态加载模式    
      Update web ui and switch from static loading mode. 

- [ ] 3. 增加英文版自传数据内容    
      Add autobiographical content in the English version.

- [ ] 4. 适当增加背景音乐内容、音效      
      Appropriately enhance background music and sound effects. 

## 免责声明 | Disclaimer

本项目仅供学习和研究使用。使用者须遵守当地的法律法规，包括但不限于 DMCA 相关法律。我们不对任何非法使用承担责任。

This project is for research and learning purposes only. Users must comply with local laws and regulations, including but not limited to DMCA-related laws. We do not take any responsibility for illegal usage.

## 技术鸣谢 | Credits

本项目基于以下开源项目构建：

- [VITS2](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Fish Speech](https://github.com/fishaudio/fish-speech)

