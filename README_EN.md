# MOSS-TTS-Nano C++ Voice Clone Tool

A zero-shot voice cloning inference tool implemented directly with the **ONNX Runtime C++ API**, based on [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) (OpenMOSS team, Apache-2.0).

- **No Python / PyTorch** required, and no browser runtime needed
- Uses the official ONNX weights (TTS + Audio Tokenizer) for local CPU inference
- Supports Chinese / Japanese / English and other languages
- Ships a **self-contained single-file exe**: models, tokenizer, and onnxruntime.dll are all embedded — zero dependencies for the receiver
- Outputs **48 kHz stereo** WAV (PCM16)

> [中文文档 (Chinese README)](./README.md)

---

## Features

- **Zero-shot voice cloning**: give a 5–15 second reference clip + target text and clone the speaker's timbre and tone
- **Pure C++ pipeline**: reference-audio encoding (codec encode) → prefill → autoregressive per-frame sampling → streaming decode (decode_step) → WAV export, no python/torch anywhere
- **Auto thread tuning**: defaults to your CPU logical-core count (capped at 8, aligned with the Python version); override with `--threads`
- **Long-text chunking**: sentences are grouped to a token budget with silence padding inserted between chunks
- **Numerical alignment**: random sampling uses a torch-free NumPy 2.x PCG64 (seed=1234) compatible implementation; reference audio is converted to 48 kHz stereo via ffmpeg to keep the same quality as the Python version
- **Two distribution flavors**:
  - `moss_tts_cpp.exe` — standard build, needs the `models/` directory and `onnxruntime.dll` beside it
  - `moss_tts_cpp_selfcontained.exe` — self-contained single file, runs standalone

---

## Quick Start

### Self-contained build (recommended for distribution)

```bat
moss_tts_cpp_selfcontained.exe --prompt-audio-path ref.wav --text-file text.txt --output out.wav
```

No installation required. On first run it auto-extracts the embedded weights to `%TEMP%\moss_tts_selfcontained` (skipped on later runs).

### Standard build (after building from source)

```bat
moss_tts_cpp.exe --prompt-audio-path ref.wav --text-file text.txt --output out.wav
```

Needs `onnxruntime.dll` in the same directory and access to the `models/` directory.

### Interactive mode

Launch with no arguments (or pass `--interactive`) to enter interactive mode: the model is loaded only once and you can synthesize many clips; parameter names may omit the `--` prefix.

```
> prompt-audio-path ref.wav --text-file text.txt --output 1.wav
```
Type `exit` / `quit` to leave.

---

## Command-Line Arguments

| Argument | Description |
| --- | --- |
| `--prompt-audio-path <wav>` / `-p` | Reference audio path (required, for voice cloning) |
| `--text <text-or-file>` / `-t` | Text to synthesize; direct text on the CLI, UTF-8 text file path in interactive mode |
| `--text-file <utf8.txt>` | Text to synthesize (UTF-8 file; recommended to avoid CJK garbling) |
| `--output <wav>` / `-o` | Output path (default `generated_audio/cpp_output.wav`) |
| `--model-dir <dir>` | Model root directory (default locates `models/` automatically) |
| `--ffmpeg <path>` | ffmpeg.exe path (used to auto-convert reference audio when not 48 kHz/stereo) |
| `--threads <n>` | Inference threads (default: CPU logical cores, capped at 8) |
| `--interactive` | Force interactive mode |
| `--help` / `-h` | Show help |

> Use either `--text` or `--text-file`; if both are given, `--text` wins.

### ffmpeg lookup order

`--ffmpeg` argument → env var `MOSS_FFMPEG` → system `PATH`. If the reference audio is not 48 kHz/stereo it is auto-converted to **48 kHz stereo PCM16** via ffmpeg; if ffmpeg is missing it falls back to the built-in resampler (quality may differ; installing ffmpeg is recommended).

---

## Reference Audio Tips

The reference audio determines the cloned timbre and tone:

- **5–15 seconds** long; longer clips add significant time with diminishing returns
- **Clean** — no background noise or BGM
- **Single speaker, steady pace**; any sample rate/channels work (auto-converted)
- A clip with very strong emotion will bias the output; for a more neutral voice, concatenate several emotional tones with ffmpeg:

```bash
ffmpeg -i a.wav -i b.wav -i c.wav -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]" -map "[out]" -c:a pcm_s16le mixed_ref.wav
```

---

## Building

### Dependencies

| Dependency | Notes |
| --- | --- |
| MSVC (Visual Studio 2022) | C++17 compiler |
| Windows 10/11 SDK | System libraries |
| CMake ≥ 3.20 + Ninja | Build tools |
| onnxruntime Win-x64 **1.29.0** | CPU build, placed under `cpp_deps/onnxruntime-win-x64-1.29.0` |
| sentencepiece **0.2.2** (static) | Text tokenizer; source in `sentencepiece-0.2.2` (bundles abseil) |

> `cpp/build_win.ps1` assumes the toolchain is at `F:\vs`; edit the `$Cmake` / `$Ninja` variables if your paths differ.

### One-click build

```powershell
powershell -ExecutionPolicy Bypass -File cpp\build_win.ps1
```

This configures/builds/installs **sentencepiece**, then configures and builds **moss_tts_cpp**. The result is at `cpp-build\moss_tts_cpp.exe`.

### Package the self-contained exe

```bash
python cpp/package_selfcontained.py
```

Appends the `models/` weights + `onnxruntime.dll` as a payload to the exe, producing `cpp-build\moss_tts_cpp_selfcontained.exe`. At runtime it detects the trailing magic (`MOSS_TTS_PKG_01`, 15 bytes) and auto-extracts to `%TEMP%\moss_tts_selfcontained`.

### Models

ONNX weights (Apache-2.0) are downloaded from HuggingFace into `models/`:

| Model | Repo |
| --- | --- |
| TTS language model (0.1B) | [OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX) |
| Audio codec (20M) | [OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX) |

```
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

## Project Layout

```
moss-tts-cpp/
├── cpp/                    # C++ source and build scripts
│   ├── src/main.cpp        # all inference logic (single file)
│   ├── CMakeLists.txt
│   ├── build_win.ps1       # Windows one-click build script
│   ├── package_selfcontained.py  # self-contained exe packager
│   └── nlohmann_json.hpp   # third-party JSON header
├── cpp_deps/               # onnxruntime-win-x64-1.29.0 (local dependency)
├── models/                 # ONNX weights (download from HuggingFace)
├── sentencepiece-0.2.2/    # sentencepiece source + build/install outputs
├── assets/                 # demo assets (demo.jsonl, reference audio, images)
├── generated_audio/        # default output directory
└── cpp-build/              # build outputs (exe / dll / selfcontained)
```

### Inference pipeline (mirrors `ort_cpu_runtime.py`)

1. Reference audio → `codec_encode` → prompt audio codes
2. Text → sentencepiece tokenize → assemble request rows
3. `prefill` → global_hidden + 12-layer KV cache
4. Autoregressive loop: `local_fixed_sampled_frame` samples a frame → `decode_step` updates the KV cache
5. Generated frames → `codec_decode_step` streaming decode → waveform → 48 kHz stereo WAV

Sampling mode is fixed (`fixed`; sampling happens inside the model, with PCG64 random numbers fed by the host). Long text is chunked automatically on a token budget (`VOICE_CLONE_MAX_TEXT_TOKENS=75`) with 0.40s (≤4 words) / 0.24s (others) of silence between pseudo-sentences.

---

## Performance Notes

- Default thread count = CPU logical cores (capped at 8); adjust with `--threads`
- On an i7-10700, processing `创世纪` (≈1100 chars / 13 chunks) with 8 threads takes ≈256s to produce ≈270s of audio — about **2.1× faster** than the Python version (8 threads)
- Varies greatly with hardware and text length

---

## License

This project is built on [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) (OpenMOSS team) and its ONNX exports. Both the model weights and this derived code follow the upstream license, **Apache License 2.0** — see [LICENSE](./LICENSE).

Third-party dependencies (onnxruntime, sentencepiece, nlohmann_json) are each governed by their own licenses; please comply with them when using.

## Citation

If you use MOSS-TTS in research or a product, please cite as requested by the authors:

```bibtex
@misc{openmoss2026mossttsnano,
  title={MOSS-TTS-Nano},
  author={OpenMOSS Team},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/OpenMOSS/MOSS-TTS-Nano}
}
```