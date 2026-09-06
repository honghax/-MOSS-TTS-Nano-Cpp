# MOSS-TTS-Nano C++ 语音克隆工具

基于 **ONNX Runtime C++ API** 直接实现的零样本语音克隆推理工具，是 [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)（OpenMOSS 团队，Apache-2.0）的纯 C++ 本地实现。

> [English README](./README_EN.md)

- **无需 Python / PyTorch**，也不需要浏览器运行时
- 使用官方导出的 ONNX 权重（TTS + Audio Tokenizer），本地 CPU 推理
- 支持中文 / 日语 / 英文等多语言
- 提供 **自包含单文件 exe**：内置模型权重、tokenizer 与 onnxruntime.dll，接收方零依赖开箱即用
- 输出 **48 kHz 双声道** WAV（PCM16）

---

## 特性

- **零样本语音克隆**：输入一段 5~15 秒参考音频 + 目标文本，即可克隆说话人音色与语气
- **纯 C++ 全流程**：参考音频编码（codec encode）→ prefill → 自回归逐帧采样 → 流式解码（decode_step）→ 写 WAV，全程不依赖 python/torch
- **线程数自动优化**：默认按 CPU 逻辑核数自动选择（上限 8，对齐 Python 版），可用 `--threads` 覆盖
- **长文本自动分块**：按 token 预算切句合并，块间自动插入静音停顿
- **数值对齐**：随机采样使用 NumPy 2.x PCG64（seed=1234）的无 torch 兼容实现；参考音频统一经 ffmpeg 转为 48kHz 双声道，保证与 Python 版音质一致
- **两种发布形态**：
  - `moss_tts_cpp.exe` — 普通版，需要项目目录内的 `models/` 与 `onnxruntime.dll`
  - `moss_tts_cpp_selfcontained.exe` — 自包含版，单文件即可独立推理

---

## 快速开始

### 自包含版（推荐分发）

```bat
moss_tts_cpp_selfcontained.exe --prompt-audio-path 参考音频.wav --text-file 文本.txt --output 输出.wav
```

不需要安装任何依赖，首次运行会自动把内置权重解压到 `%TEMP%\moss_tts_selfcontained`（无需重复解压）。

### 普通版（源码构建后）

```bat
moss_tts_cpp.exe --prompt-audio-path 参考音频.wav --text-file 文本.txt --output 输出.wav
```

需要同目录 `onnxruntime.dll`，并能找到 `models/` 模型目录。

### 交互模式

不带任何参数启动（或加 `--interactive`）即进入交互模式：模型只加载一次，可反复合成多条语音，参数名可省略 `--` 前缀。

```
> prompt-audio-path 参考音频.wav --text-file 文本.txt --output 1.wav
```
输入 `exit` / `quit` / `退出` 结束。

---

## 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--prompt-audio-path <wav>` / `-p` | 参考音频路径（必填，用于克隆音色） |
| `--text <文本或文件>` / `-t` | 待合成文本；命令行模式可直接传文本，交互模式下为 UTF-8 文本文件路径 |
| `--text-file <utf8.txt>` | 待合成文本（UTF-8 编码的文本文件，推荐，避免中文/日语乱码） |
| `--output <wav>` / `-o` | 输出路径（默认 `generated_audio/cpp_output.wav`） |
| `--model-dir <dir>` | 模型根目录（默认自动定位到仓库 `models/`） |
| `--ffmpeg <path>` | ffmpeg.exe 路径（参考音频非 48kHz/双声道时自动转换用） |
| `--threads <n>` | 推理线程数（默认按 CPU 逻辑核自动，上限 8） |
| `--interactive` | 强制进入交互模式 |
| `--help` / `-h` | 查看帮助 |

> `--text` 与 `--text-file` 二选一，同时提供时以 `--text` 为准。

### ffmpeg 查找顺序

`--ffmpeg` 参数 → 环境变量 `MOSS_FFMPEG` → 系统 PATH。参考音频若非 48kHz/双声道，会自动调用 ffmpeg 统一转换为 **48kHz 双声道 PCM16**；找不到 ffmpeg 时回退到内置重采样（音质可能有差异，建议安装）。

---

## 参考音频建议

参考音频决定克隆出的音色与语气：

- **时长 5~15 秒**，过长会显著增加耗时且收益递减
- **干净、无背景噪音、无 BGM**
- **单说话人，语速平稳**；采样率/声道任意，程序会自动转换
- 语气强烈的音频会让结果带上该语气；如需更中性的音色，可用 ffmpeg 拼接多种语气：

```bash
ffmpeg -i a.wav -i b.wav -i c.wav -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]" -map "[out]" -c:a pcm_s16le mixed_ref.wav
```

---

## 构建

### 依赖

| 依赖 | 说明 |
| --- | --- |
| MSVC（Visual Studio 2022） | C++17 编译器 |
| Windows 10/11 SDK | 链接系统库 |
| CMake ≥ 3.20 + Ninja | 构建工具 |
| onnxruntime Win-x64 **1.29.0** | CPU 版，放到 `cpp_deps/onnxruntime-win-x64-1.29.0` |
| sentencepiece **0.2.2**（静态） | 文本 tokenizer，源码在 `sentencepiece-0.2.2`（含 abseil 依赖） |

> `cpp/build_win.ps1` 假定工具链位于 `F:\vs`，如路径不同请按需修改脚本中的 `$Cmake` / `$Ninja`。

### 一键构建

```powershell
powershell -ExecutionPolicy Bypass -File cpp\build_win.ps1
```

脚本会依次完成：配置/编译/安装 **sentencepiece** → 配置/编译 **moss_tts_cpp**，产物在 `cpp-build\moss_tts_cpp.exe`。

### 打包自包含 exe

```bash
python cpp/package_selfcontained.py
```

把 `models/` 权重 + `onnxruntime.dll` 打包成 payload 追加到 exe 尾部，生成 `cpp-build\moss_tts_cpp_selfcontained.exe`。运行时会检测尾部魔数（`MOSS_TTS_PKG_01`，15 字节）并自动解压到 `%TEMP%\moss_tts_selfcontained`。

### 模型

ONNX 权重从 HuggingFace 下载（Apache-2.0），放到 `models/` 对应目录：

| 模型 | 仓库 |
| --- | --- |
| TTS 语言模型（0.1B） | [OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX) |
| 音频编解码器（20M） | [OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX) |

```bash
# 目录结构
models/
├── MOSS-TTS-Nano-100M-ONNX/
│   ├── browser_poc_manifest.json
│   ├── moss_tts_prefill.onnx / moss_tts_decode_step.onnx / ...
│   ├── tokenizer.model
│   └── ...
└── MOSS-Audio-Tokenizer-Nano-ONNX/
    ├── moss_audio_tokenizer_encode.onnx
    ├── moss_audio_tokenizer_decode_step.onnx / ...
    └── ...
```

---

## 项目结构

```
moss-tts-cpp/
├── cpp/                    # C++ 源码与构建脚本
│   ├── src/main.cpp        # 全部推理逻辑（单文件）
│   ├── CMakeLists.txt
│   ├── build_win.ps1       # Windows 一键构建脚本
│   ├── package_selfcontained.py  # 自包含 exe 打包脚本
│   └── nlohmann_json.hpp   # 第三方 JSON 头（utils 镜像）
├── cpp_deps/               # onnxruntime-win-x64-1.29.0（本地依赖）
├── models/                 # ONNX 权重（需自行从 HuggingFace 下载）
├── sentencepiece-0.2.2/    # sentencepiece 源码 + 构建/安装产物
├── assets/                 # 演示素材（demo.jsonl、参考音频、图片）
├── generated_audio/        # 默认输出目录
└── cpp-build/              # 构建产物（exe / dll / selfcontained）
```

### 推理流程（对照 `ort_cpu_runtime.py`）

1. 参考音频 → `codec_encode` → prompt audio codes
2. 文本 → sentencepiece tokenize → 组装 request rows
3. `prefill` → global_hidden + 12 层 KV cache
4. 自回归循环：`local_fixed_sampled_frame` 采样一帧 → `decode_step` 更新 KV
5. 生成的 frames → `codec_decode_step` 流式解码 → 波形 → 写 48kHz 双声道 WAV

采样模式固定为 `fixed`（在模型内完成，主机提供 PCG64 随机数）。长文本按 token 预算自动切句分块（`VOICE_CLONE_MAX_TEXT_TOKENS=75`），伪句块间插入 4 字以内 0.40s / 其余 0.24s 静音。

---

## 性能参考

- 线程数默认 = CPU 逻辑核数（上限 8），可用 `--threads` 调整
- 在 i7-10700 上，`创世纪`（约 1100 字 / 13 块）8 线程约 256s 生成约 270s 音频，约为 Python 版（8 线程）的 **2.1 倍**
- 受硬件与文本长度影响较大，仅供参考

> 提示：Intel CPU 高负载睿频时若遇断电/蓝屏，可在 Windows 电源选项中把「最大处理器状态」限制到 95% 附近以降低功耗。

---

## 许可

本项目基于 [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)（OpenMOSS 团队）的 ONNX 导出实现，其模型权重与派生代码沿用原始许可证 **Apache License 2.0**，详见 [LICENSE](./LICENSE)。

第三方依赖（onnxruntime、sentencepiece、nlohmann_json）各自遵循其原始许可证，使用时请予以遵守。

## 引用

若您的科研或产品使用了 MOSS-TTS，请按官方要求引用：

```bibtex
@misc{openmoss2026mossttsnano,
  title={MOSS-TTS-Nano},
  author={OpenMOSS Team},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/OpenMOSS/MOSS-TTS-Nano}
}
```