# MIDI 导唱生成 · Vocal Synthesizer

**v1.0** — 读取 **ACE Studio 风格 MIDI**（音符 + 歌词），调用本地 **DiffSinger** 合成自然导唱 WAV，供跟唱练习使用。

## 功能

- **批量输入**：拖入单/多个 `.mid` 文件，或整个文件夹（自动递归所有 `.mid` / `.midi`）
- 支持 ACE Studio / KTV 歌词格式：
  - `baby#1` / `baby#2` — 一词多音节
  - `baby` + `-` — 延音（slur，不换字）
- 中 / 英 / 日 / 韩混合歌词自动识别语种
- 自动解析音符、歌词对齐，构建 DiffSinger `.ds` 输入
- Pitch + Variance + Acoustic 三阶段推理，输出与 MIDI 时长严格一致
- 现代 Fluent UI，队列进度可视化，启动 Splash

## 直接使用（Windows）

从 [Releases](https://github.com/wyh-alt/Vocal-Synthesizer/releases) 下载 `VocalSynthesizer-v1.0.0-win64.zip`，解压任意位置，双击 `VocalSynthesizer.exe` 即可运行。**无需安装 Python 或任何依赖，声库已内置。**

## 环境要求

- Windows 10+
- Python 3.10+
- 本地 [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger) 安装（含 variance / acoustic 模型与 NSF-HiFiGAN 声码器）

## 安装

```bash
pip install -r requirements.txt
python scripts/create_icon.py
```

## 运行

```bash
python main.py
```

## DiffSinger / 声库配置

本项目已内置：

| 资源 | 路径 |
|------|------|
| DiffSinger 2.5.1 | `DiffSinger-2.5.1/`（参考用，PyTorch 训练/导出） |
| **Nishiren v2.0** | `Nishiren Diffsinger v2.0/`（**默认 ONNX 声库**） |

界面「Nishiren / ONNX 声库」中填写声库目录，选择说话人（Standard / Sweet / Soft / Power / Emotional / 2P）。语言留空或填 `auto` 时按 MIDI 歌词内容自动识别（支持中/英/日/韩混合歌词，每个音符各自判断语种），也可手动指定 `en` / `zh` / `ja` / `ko` 强制统一语言。

> Nishiren 是 OpenUTAU 格式 ONNX 声库，**不能**用 `DiffSinger-2.5.1/scripts/infer.py` 直接推理；程序已内置 ONNX 推理链（pitch → variance → acoustic → vocoder）。

## 工作流建议

1. 用 MS_json 导出「合并导出（同轨）」MIDI
2. 打开程序，把 `.mid` / 目录拖入队列
3. （可选）指定统一输出目录
4. **开始批量生成** → 逐个产出 `*_导唱.wav`

## 从源码打包 exe

```bash
pip install -r requirements.txt pyinstaller
python scripts/create_icon.py
pyinstaller --clean --noconfirm VocalSynthesizer.spec
# 产物在 dist/VocalSynthesizer/
```

打包会自动内置 `Nishiren Diffsinger v2.0/` 声库（约 1GB）。确保打包前该目录存在。

## 项目结构

```
core/           MIDI 解析、音素转换、DS 构建、DiffSinger 调用、音频对齐
ui/             PyQt6 Fluent 界面
scripts/        图标生成、测试脚本
main.py         程序入口
```

## 测试

```bash
python scripts/test_pipeline.py
```
