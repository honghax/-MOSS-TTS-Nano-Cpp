# 打包 MOSS-TTS-Nano 自包含 exe
# 把 models 权重 + onnxruntime.dll 打包成 payload 追加到 moss_tts_cpp.exe 尾部
# 生成 moss_tts_cpp_selfcontained.exe（单个 exe 即可独立推理）
# 用法：python cpp/package_selfcontained.py [--exe cpp-build/moss_tts_cpp.exe] [--out cpp-build/moss_tts_cpp_selfcontained.exe]
import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # MOSS-TTS-Nano 根
MODELS = ROOT / "models"
MAGIC = b"MOSS_TTS_PKG_01"  # 16 字节

# 需要打包的文件（保持相对 models 的路径结构）
def collect_files():
    files = {}  # rel_path_in_payload -> abs_path
    # onnxruntime.dll（payload 根目录）
    dll = ROOT / "cpp_deps" / "onnxruntime-win-x64-1.29.0" / "lib" / "onnxruntime.dll"
    files["onnxruntime.dll"] = dll
    # models 下的权重/配置
    for sub in ["MOSS-TTS-Nano-100M-ONNX", "MOSS-Audio-Tokenizer-Nano-ONNX"]:
        d = MODELS / sub
        if not d.is_dir():
            raise FileNotFoundError(f"missing model dir: {d}")
        for p in sorted(d.iterdir()):
            if p.is_dir() and p.name == "download":
                continue  # 跳过下载缓存
            if p.suffix.lower() in (".onnx", ".data", ".json", ".model") and p.name != "README.md":
                files[f"models/{sub}/{p.name}"] = p
    # 校验核心文件齐全
    for required in [
        "models/MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
        "models/MOSS-TTS-Nano-100M-ONNX/tts_browser_onnx_meta.json",
        "models/MOSS-TTS-Nano-100M-ONNX/tokenizer.model",
        "models/MOSS-TTS-Nano-100M-ONNX/moss_tts_global_shared.data",
        "models/MOSS-TTS-Nano-100M-ONNX/moss_tts_local_shared.data",
        "models/MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json",
        "models/MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_encode.data",
        "models/MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_decode_shared.data",
    ]:
        if required not in files:
            raise FileNotFoundError(f"required file missing: {required}")
    return files


def build_payload(files):
    # 布局：[索引区][文件数据区][u64 payload_size][15B magic]
    # 索引：u32 num_files; 每项 u32 name_len + name + u64 size + u64 offset
    # offset 为相对 payload 起点的绝对偏移（C++ 端 payload_start + offset 直接定位）
    name_lens = [(rel, len(rel.encode("utf-8"))) for rel in files]
    index_len = 4 + sum(4 + nlen + 16 for _, nlen in name_lens)
    # 数据区起点 = 索引区长度
    offset = index_len
    entries = []
    data = b""
    for rel, abs_path in files.items():
        blob = abs_path.read_bytes()
        entries.append((rel, len(blob), offset))
        data += blob
        offset += len(blob)
    index = struct.pack("<I", len(entries))
    for rel, size, off in entries:
        name = rel.encode("utf-8")
        index += struct.pack("<I", len(name)) + name + struct.pack("<QQ", size, off)
    payload = index + data
    tail = struct.pack("<Q", len(payload)) + MAGIC
    return payload + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(ROOT / "cpp-build" / "moss_tts_cpp.exe"))
    ap.add_argument("--out", default=str(ROOT / "cpp-build" / "moss_tts_cpp_selfcontained.exe"))
    args = ap.parse_args()

    exe = Path(args.exe)
    if not exe.is_file():
        sys.exit(f"exe not found: {exe}")
    files = collect_files()
    payload = build_payload(files)
    total = exe.stat().st_size + len(payload)
    out = Path(args.out)
    with open(exe, "rb") as f, open(out, "wb") as g:
        g.write(f.read())
        g.write(payload)
    print(f"packed {len(files)} files, payload={len(payload)/1024/1024:.1f} MB")
    print(f"output: {out} ({total/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
