// MOSS-TTS-Nano C++ 全流程推理（onnxruntime C++ API 直接实现）
// 对照参考：ort_cpu_runtime.py（纯 ORT 无 torch 的 Python 实现）
// 流程：参考音频 -> codec_encode -> prompt_audio_codes
//       文本 -> sentencepiece tokenize -> 组装 request rows
//       prefill -> global_hidden + KV cache
//       自回归循环：local_fixed_sampled_frame 采样一帧 -> decode_step 更新 KV
//       generated frames -> codec_decode_full -> 波形 -> 写 WAV
// 采样模式固定为 fixed（与 infer_onnx.py 默认一致，采样在模型内完成）

#ifdef _WIN32
#define NOMINMAX   // 防止 windows.h 的 min/max 宏污染 std::min/std::max
#endif
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#endif
#undef min
#undef max
#undef OR
#undef AND

#include "nlohmann_json.hpp"

using json = nlohmann::json;

// 自包含 payload 魔数（与 package_selfcontained.py 一致，15 字节，不含结尾 \0）
static constexpr char kPkgMagic[] = "MOSS_TTS_PKG_01";
static constexpr size_t kPkgMagicLen = sizeof(kPkgMagic) - 1; // 15

// UTF-8 -> UTF-16（onnxruntime 在 Windows 下 Session 模型路径为宽字符 ORTCHAR_T）
static std::wstring utf8_to_wchar(const std::string& s) {
    std::wstring out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            out.push_back(static_cast<wchar_t>(c));
            ++i;
        } else if ((c >> 5) == 0x6) {
            out.push_back(static_cast<wchar_t>(((c & 0x1F) << 6) |
                                               (static_cast<unsigned char>(s[i + 1]) & 0x3F)));
            i += 2;
        } else if ((c >> 4) == 0xE) {
            out.push_back(static_cast<wchar_t>(((c & 0x0F) << 12) |
                                               ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
                                               (static_cast<unsigned char>(s[i + 2]) & 0x3F)));
            i += 3;
        } else {
            out.push_back(0xFFFD);
            i += 4; // 补充平面字符简化处理（本地模型路径不会出现）
        }
    }
    return out;
}

// UTF-16 -> UTF-8
static std::string utf8_from_wchar(const std::wstring& w) {
    if (w.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(len, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), &s[0], len, nullptr, nullptr);
    return s;
}

// 宽字符路径打开文件（MSVC 扩展：ifstream/ofstream 支持 wchar_t* 路径，兼容中文路径）
static std::ifstream open_ifstream(const std::string& utf8_path) {
    return std::ifstream(utf8_to_wchar(utf8_path), std::ios::binary);
}
static std::ofstream open_ofstream(const std::string& utf8_path) {
    return std::ofstream(utf8_to_wchar(utf8_path), std::ios::binary);
}

// 默认线程数：按 CPU 逻辑核自动选择（上限 8，与 Python 版 --cpu-threads 8 对齐，避免超线程争抢）
static int default_threads_() {
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) return 4;
    return static_cast<int>(std::min<unsigned>(hc, 8));
}

// 查找 ffmpeg：优先 --ffmpeg 参数，其次环境变量 MOSS_FFMPEG，最后 PATH
static std::string find_ffmpeg_(const std::string& hint) {
    if (!hint.empty()) return hint;
    const char* env = std::getenv("MOSS_FFMPEG");
    if (env && *env) return env;
    wchar_t buf[4096];
    if (SearchPathW(nullptr, L"ffmpeg.exe", nullptr, 4096, buf, nullptr))
        return utf8_from_wchar(buf);
    return "";
}

// 探测 WAV 头：采样率与声道数（读取前 64 字节）
static void probe_wav_format_(const std::string& path, int& sample_rate, int& channels) {
    std::ifstream f = open_ifstream(path);
    if (!f) throw std::runtime_error("cannot open wav: " + path);
    char buf[64];
    f.read(buf, 64);
    std::streamsize n = f.gcount();
    if (n < 44 || std::memcmp(buf, "RIFF", 4) != 0 || std::memcmp(buf + 8, "WAVE", 4) != 0)
        throw std::runtime_error("not a RIFF wav: " + path);
    int off = 12;
    while (off + 8 <= n) {
        uint32_t sz = 0;
        std::memcpy(&sz, buf + off + 4, 4);
        if (std::memcmp(buf + off, "fmt ", 4) == 0) {
            if (off + 20 > n) throw std::runtime_error("truncated fmt chunk");
            uint16_t ch = 0;
            uint32_t sr = 0;
            std::memcpy(&ch, buf + off + 10, 2);
            std::memcpy(&sr, buf + off + 12, 4);
            channels = ch;
            sample_rate = static_cast<int>(sr);
            return;
        }
        off += 8 + static_cast<int>(sz);
    }
    throw std::runtime_error("no fmt chunk in wav: " + path);
}

// 用 ffmpeg 把音频转成 48kHz 双声道 PCM16 WAV（与 Python/torchaudio 参考处理一致）。
// 用 CreateProcessW 直接启动，避免 _wsystem 经 cmd.exe 时嵌套引号导致路径解析失败。
static std::string convert_audio_48k_stereo_(const std::string& ffmpeg, const std::string& in, const std::string& out) {
    std::wstring cmd = L"\"" + utf8_to_wchar(ffmpeg) + L"\" -y -i \"" + utf8_to_wchar(in) +
                       L"\" -ar 48000 -ac 2 -c:a pcm_s16le \"" + utf8_to_wchar(out) + L"\"";
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi))
        return "";
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec = 1;
    GetExitCodeProcess(pi.hProcess, &ec);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return ec == 0 ? out : "";
}

// 自解压：检测 exe 尾部自包含 payload，解压到 %TEMP%\moss_tts_selfcontained
// 返回 true 表示自包含模式，out_dir 为解压目录（onnxruntime.dll 也在其中）
static bool extract_self(const std::string& exe_path, std::string& out_dir) {
    std::ifstream f = open_ifstream(exe_path);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    int64_t filesize = static_cast<int64_t>(f.tellg());
    constexpr int64_t kTailLen = 8 + 15; // [u64 payload_size][15B magic]
    if (filesize < kTailLen) return false;
    // 读尾部 23 字节
    f.seekg(filesize - kTailLen, std::ios::beg);
    char tail[23];
    f.read(tail, kTailLen);
    uint64_t payload_size = 0;
    std::memcpy(&payload_size, tail, 8);
    if (std::memcmp(tail + 8, kPkgMagic, kPkgMagicLen) != 0) return false;
    if (payload_size > static_cast<uint64_t>(filesize - kTailLen)) return false;
    int64_t payload_start = filesize - kTailLen - static_cast<int64_t>(payload_size);

    // 解压目标目录
    wchar_t tmp[32768];
    DWORD tlen = GetTempPathW(32768, tmp);
    if (tlen == 0) return false;
    std::wstring base = std::wstring(tmp) + L"moss_tts_selfcontained";
    CreateDirectoryW(base.c_str(), nullptr);
    out_dir = utf8_from_wchar(base);

    // 已解压则跳过（二次启动免解压）
    std::string flag = out_dir + "\\models\\MOSS-TTS-Nano-100M-ONNX\\browser_poc_manifest.json";
    if (GetFileAttributesW(utf8_to_wchar(flag).c_str()) != INVALID_FILE_ATTRIBUTES) return true;

    std::cout << "extracting embedded assets to " << out_dir << " ..." << std::endl;
    f.seekg(payload_start, std::ios::beg);
    uint32_t num_files = 0;
    f.read(reinterpret_cast<char*>(&num_files), 4);
    struct Entry {
        std::string name;
        uint64_t size;
        uint64_t offset;
    };
    std::vector<Entry> entries;
    entries.reserve(num_files);
    for (uint32_t i = 0; i < num_files; ++i) {
        uint32_t nlen = 0;
        f.read(reinterpret_cast<char*>(&nlen), 4);
        std::string name(nlen, '\0');
        f.read(&name[0], nlen);
        uint64_t size = 0, off = 0;
        f.read(reinterpret_cast<char*>(&size), 8);
        f.read(reinterpret_cast<char*>(&off), 8);
        entries.push_back({std::move(name), size, off});
    }
    std::vector<char> buf(1 << 20);
    for (const auto& e : entries) {
        std::string dst = out_dir + "\\" + e.name;
        std::replace(dst.begin(), dst.end(), '/', '\\'); // 统一为反斜杠
        size_t pos = dst.find_last_of("\\/");
        if (pos != std::string::npos) { // 逐级创建父目录
            std::wstring dir = utf8_to_wchar(dst.substr(0, pos));
            size_t cur = 1;
            while ((cur = dir.find(L'\\', cur)) != std::string::npos) {
                if (!CreateDirectoryW(dir.substr(0, cur).c_str(), nullptr) &&
                    GetLastError() != ERROR_ALREADY_EXISTS) {
                    std::cerr << "extract mkdir fail: " << utf8_from_wchar(dir.substr(0, cur))
                              << " err=" << GetLastError() << std::endl;
                    return false;
                }
                ++cur;
            }
            if (!CreateDirectoryW(dir.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
                std::cerr << "extract mkdir fail: " << utf8_from_wchar(dir) << " err=" << GetLastError() << std::endl;
                return false;
            }
        }
        std::ofstream g = open_ofstream(dst);
        if (!g) {
            std::cerr << "extract open fail: " << dst << " errno=" << errno << std::endl;
            return false;
        }
        uint64_t left = e.size, at = static_cast<uint64_t>(payload_start) + e.offset;
        while (left > 0) {
            size_t chunk = static_cast<size_t>(std::min<uint64_t>(left, buf.size()));
            f.seekg(static_cast<int64_t>(at), std::ios::beg);
            f.read(buf.data(), static_cast<std::streamsize>(chunk));
            g.write(buf.data(), static_cast<std::streamsize>(chunk));
            if (!g) {
                std::cerr << "extract write fail: " << dst << " errno=" << errno << std::endl;
                return false;
            }
            at += chunk;
            left -= chunk;
        }
    }
    std::cout << "extract done." << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// 模型常量（与 tts_browser_onnx_meta.json / codec_browser_onnx_meta.json 一致）
// ---------------------------------------------------------------------------
constexpr int N_VQ = 16;                  // 音频码本数量
constexpr int ROW_WIDTH = N_VQ + 1;       // 每行 token 宽度
constexpr int HIDDEN_SIZE = 768;          // 全局模型 hidden
constexpr int GLOBAL_LAYERS = 12;         // 全局 transformer 层数
constexpr int GLOBAL_HEADS = 12;          // 全局注意力头数
constexpr int HEAD_DIM = 64;              // 每头维度
constexpr int AUDIO_CODEBOOK_SIZE = 1024; // 音频码本大小
constexpr int CODEC_SAMPLE_RATE = 48000;  // 输出采样率
constexpr int CODEC_CHANNELS = 2;         // 输出声道数
constexpr int VOICE_CLONE_MAX_TEXT_TOKENS = 75; // 长文本分块 token 预算

// ---------------------------------------------------------------------------
// 简易 UTF-8 字符迭代器（用于按字符切句、判断标点，保证多字节安全）
// ---------------------------------------------------------------------------
struct Utf8Iter {
    const std::string& s;
    size_t pos = 0;
    explicit Utf8Iter(const std::string& str) : s(str) {}
    std::string next() { // 返回当前位置的一个 UTF-8 字符并前进
        if (pos >= s.size()) return {};
        size_t start = pos;
        unsigned char c = static_cast<unsigned char>(s[pos]);
        size_t len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        pos = std::min(s.size(), start + len);
        return s.substr(start, pos - start);
    }
    bool done() const { return pos >= s.size(); }
};

// 判断 UTF-8 字符是否为 CJK
static bool is_cjk(const std::string& ch) {
    if (ch.size() != 3) return false;
    auto b1 = static_cast<unsigned char>(ch[0]);
    auto b2 = static_cast<unsigned char>(ch[1]);
    auto b3 = static_cast<unsigned char>(ch[2]);
    unsigned int cp = ((b1 & 0x0F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
    return (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0xAC00 && cp <= 0xD7AF);
}

// 句子结束标点（对应 Python SENTENCE_END_PUNCTUATION）
static bool is_sentence_end(const std::string& ch) {
    if (ch.size() == 1) {
        char c = ch[0];
        return c == '.' || c == '!' || c == '?' || c == ';';
    }
    return ch == "。" || ch == "！" || ch == "？" || ch == "；";
}

// 子句分隔标点（对应 Python CLAUSE_SPLIT_PUNCTUATION）
static bool is_clause_split(const std::string& ch) {
    if (ch.size() == 1) {
        char c = ch[0];
        return c == ',' || c == ';' || c == ':';
    }
    return ch == "，" || ch == "、" || ch == "；" || ch == "：";
}

// 轻量文本规范化：trim + 压缩空白 + 换行替换为空格（UTF-8 安全）
static std::string normalize_tts_text(const std::string& text) {
    std::string s = text;
    for (auto& c : s) {
        if (c == '\r' || c == '\n') c = ' ';
    }
    std::string out;
    out.reserve(s.size());
    bool last_space = false;
    for (char c : s) {
        if (c == ' ' || c == '\t') {
            if (!last_space) out.push_back(' ');
            last_space = true;
        } else {
            out.push_back(c);
            last_space = false;
        }
    }
    size_t b = out.find_first_not_of(' ');
    size_t e = out.find_last_not_of(' ');
    if (b == std::string::npos) return "";
    return out.substr(b, e - b + 1);
}

// ---------------------------------------------------------------------------
// WAV 读取（支持 PCM8/16/24/32 与 float32），输出 channel-major float 数组
// ---------------------------------------------------------------------------
struct WavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<std::vector<float>> data; // data[ch][n]
};

static WavData read_wav(const std::string& path) {
    std::ifstream f = open_ifstream(path);
    if (!f) throw std::runtime_error("cannot open wav: " + path);
    char buf[4];
    auto read4 = [&f](char* b) { f.read(b, 4); };
    read4(buf);
    if (std::memcmp(buf, "RIFF", 4) != 0) throw std::runtime_error("not a RIFF wav: " + path);
    f.seekg(4, std::ios::cur);
    read4(buf);
    if (std::memcmp(buf, "WAVE", 4) != 0) throw std::runtime_error("not WAVE: " + path);

    int fmt = 0, channels = 0, sample_rate = 0, bits = 0;
    std::vector<int16_t> pcm16;
    std::vector<float> f32;
    bool have_data = false;
    while (f.read(buf, 4)) {
        std::string chunk(buf, 4);
        uint32_t size = 0;
        f.read(reinterpret_cast<char*>(&size), 4);
        if (chunk == "fmt ") {
            uint16_t fmt_tag = 0;
            f.read(reinterpret_cast<char*>(&fmt_tag), 2);
            f.read(reinterpret_cast<char*>(&channels), 2);
            f.read(reinterpret_cast<char*>(&sample_rate), 4);
            f.seekg(6, std::ios::cur);
            f.read(reinterpret_cast<char*>(&bits), 2);
            fmt = fmt_tag;
            f.seekg(size - 16, std::ios::cur);
        } else if (chunk == "data") {
            have_data = true;
            if (fmt == 3) {
                size_t n = size / 4;
                f32.resize(n);
                f.read(reinterpret_cast<char*>(f32.data()), size);
            } else if (fmt == 1) {
                size_t n = size / (bits / 8);
                if (bits == 16) {
                    pcm16.resize(n);
                    f.read(reinterpret_cast<char*>(pcm16.data()), size);
                } else if (bits == 8) {
                    std::vector<uint8_t> u8(n);
                    f.read(reinterpret_cast<char*>(u8.data()), size);
                    pcm16.resize(n);
                    for (size_t i = 0; i < n; ++i)
                        pcm16[i] = static_cast<int16_t>((static_cast<int>(u8[i]) - 128) * 256);
                } else if (bits == 24) {
                    std::vector<uint8_t> b(size);
                    f.read(reinterpret_cast<char*>(b.data()), size);
                    pcm16.resize(n);
                    for (size_t i = 0; i < n; ++i) {
                        int32_t v = (b[i * 3] | (b[i * 3 + 1] << 8) | (b[i * 3 + 2] << 16));
                        if (v & 0x800000) v |= ~0xFFFFFF;
                        pcm16[i] = static_cast<int16_t>(v >> 8);
                    }
                } else if (bits == 32) {
                    std::vector<int32_t> i32(n);
                    f.read(reinterpret_cast<char*>(i32.data()), size);
                    pcm16.resize(n);
                    for (size_t i = 0; i < n; ++i) pcm16[i] = static_cast<int16_t>(i32[i] >> 16);
                } else {
                    throw std::runtime_error("unsupported wav bits: " + std::to_string(bits));
                }
            } else {
                throw std::runtime_error("unsupported wav fmt tag: " + std::to_string(fmt));
            }
        } else {
            f.seekg(size + (size & 1), std::ios::cur);
        }
    }
    if (!have_data) throw std::runtime_error("wav has no data chunk: " + path);

    std::vector<float> samples;
    if (!f32.empty()) samples = f32;
    else {
        samples.resize(pcm16.size());
        for (size_t i = 0; i < pcm16.size(); ++i) samples[i] = static_cast<float>(pcm16[i]) / 32768.0f;
    }

    WavData out;
    out.sample_rate = sample_rate;
    out.channels = channels;
    size_t frames = samples.size() / static_cast<size_t>(channels);
    out.data.resize(static_cast<size_t>(channels));
    for (int ch = 0; ch < channels; ++ch) {
        out.data[ch].resize(frames);
        for (size_t i = 0; i < frames; ++i) out.data[ch][i] = samples[i * channels + ch];
    }
    return out;
}

// 高质量重采样（windowed-sinc + kaiser 窗，接近 torchaudio 的 sinc_interp_kaiser 品质）
static float kaiser_i0(float x) {
    // I0 修正贝塞尔函数近似（Abramowitz & Stegun）
    float y = x * 0.5f;
    float t = y * y;
    return 1.0f + t * (1.0f + t * (0.25f + t * (0.0277778f + t * (0.00173611f +
             t * (0.0000642881f + t * (0.00000138327f))))));
}

static std::vector<std::vector<float>> resample_channels(
    const std::vector<std::vector<float>>& in, int in_rate, int out_rate) {
    if (in_rate == out_rate) return in;
    constexpr double kPi = 3.14159265358979323846;
    const float beta = 14.769656f;  // kaiser 窗参数（与 torchaudio 默认一致）
    const int half = 32;            // 半窗长度（输入采样域）
    std::vector<std::vector<float>> out(in.size());
    double ratio = static_cast<double>(in_rate) / static_cast<double>(out_rate);
    // 预计算 kaiser 窗（对称）
    std::vector<float> win(static_cast<size_t>(2 * half + 1));
    for (int i = -half; i <= half; ++i) {
        float x = static_cast<float>(i) / static_cast<float>(half);
        win[i + half] = kaiser_i0(beta * std::sqrt(1.0f - x * x)) / kaiser_i0(beta);
    }
    for (size_t ch = 0; ch < in.size(); ++ch) {
        const auto& src = in[ch];
        size_t n = static_cast<size_t>(std::ceil((src.size() - 1) * out_rate / static_cast<double>(in_rate))) + 1;
        out[ch].resize(n);
        for (size_t i = 0; i < n; ++i) {
            double pos = i * ratio;
            int64_t center = static_cast<int64_t>(std::llround(pos));
            double frac = pos - center;
            double acc = 0.0, wsum = 0.0;
            for (int k = -half; k <= half; ++k) {
                int64_t idx = center + k;
                if (idx < 0 || idx >= static_cast<int64_t>(src.size())) continue;
                double x = static_cast<double>(k) - frac; // 距中心距离（输入采样单位）
                double sinc = (x == 0.0) ? 1.0 : std::sin(kPi * x) / (kPi * x);
                double w = win[k + half];
                acc += src[idx] * sinc * w;
                wsum += w;
            }
            out[ch][i] = (wsum > 0) ? static_cast<float>(acc / wsum) : 0.0f;
        }
    }
    return out;
}

// 写 48kHz stereo PCM16 WAV（与 Python _write_waveform_to_wav 一致）
static void write_wav(const std::string& path, const std::vector<std::vector<float>>& channels,
                      int sample_rate) {
    std::ofstream f = open_ofstream(path);
    if (!f) throw std::runtime_error("cannot write wav: " + path);
    int ch = static_cast<int>(channels.size());
    size_t frames = channels[0].size();
    uint32_t data_bytes = static_cast<uint32_t>(frames * ch * 2);
    auto wr = [&f](const char* d, size_t n) { f.write(d, static_cast<std::streamsize>(n)); };
    wr("RIFF", 4);
    uint32_t riff_size = 36 + data_bytes;
    wr(reinterpret_cast<const char*>(&riff_size), 4);
    wr("WAVE", 4);
    wr("fmt ", 4);
    uint32_t fmt_size = 16;
    uint16_t fmt_tag = 1, num_ch = static_cast<uint16_t>(ch);
    uint32_t sr = static_cast<uint32_t>(sample_rate), br = sr * ch * 2;
    uint16_t align = static_cast<uint16_t>(ch * 2), bits = 16;
    wr(reinterpret_cast<const char*>(&fmt_size), 4);
    wr(reinterpret_cast<const char*>(&fmt_tag), 2);
    wr(reinterpret_cast<const char*>(&num_ch), 2);
    wr(reinterpret_cast<const char*>(&sr), 4);
    wr(reinterpret_cast<const char*>(&br), 4);
    wr(reinterpret_cast<const char*>(&align), 2);
    wr(reinterpret_cast<const char*>(&bits), 2);
    wr("data", 4);
    wr(reinterpret_cast<const char*>(&data_bytes), 4);
    for (size_t i = 0; i < frames; ++i) {
        for (int c = 0; c < ch; ++c) {
            float v = std::max(-1.0f, std::min(1.0f, channels[c][i]));
            int16_t s = static_cast<int16_t>(std::round(v * 32767.0f));
            wr(reinterpret_cast<const char*>(&s), 2);
        }
    }
}

// ---------------------------------------------------------------------------
// 运行时：管理 onnxruntime sessions 与 sentencepiece tokenizer
// ---------------------------------------------------------------------------
class MossTtsCpp {
public:
    MossTtsCpp(const std::string& model_dir, int threads, const std::string& ffmpeg_hint = "")
        : env_(ORT_LOGGING_LEVEL_WARNING, "moss-tts-cpp"),
          threads_(threads <= 0 ? default_threads_() : threads) {
        model_dir_ = model_dir;
        ffmpeg_ = find_ffmpeg_(ffmpeg_hint);
        if (!model_dir_.empty() && model_dir_.back() != '/' && model_dir_.back() != '\\')
            model_dir_ += "\\";
        load_configs_();
        load_sessions_();
        std::string sp_path = join_path_(manifest_dir_, manifest_["model_files"]["tokenizer_model"].get<std::string>());
        auto status = sp_.Load(sp_path);
        if (!status.ok()) throw std::runtime_error("sentencepiece load failed: " + status.ToString());
    }

    // 主入口：给定参考音频 + 文本，输出合成波形（channel-major）
    std::vector<std::vector<float>> synthesize(const std::string& prompt_wav, const std::string& text) {
        auto prompt_codes = encode_reference_audio(prompt_wav);
        auto chunks = split_voice_clone_text(text, VOICE_CLONE_MAX_TEXT_TOKENS);
        std::vector<std::vector<float>> all;
        for (size_t i = 0; i < chunks.size(); ++i) {
            auto wave = synthesize_chunk(chunks[i], prompt_codes);
            if (!wave.empty()) {
                if (all.empty()) all = wave;
                else {
                    for (size_t c = 0; c < all.size(); ++c)
                        all[c].insert(all[c].end(), wave[c].begin(), wave[c].end());
                }
            }
            if (i + 1 < chunks.size()) {
                int word_count = 0;
                std::istringstream words(chunks[i]);
                std::string word;
                while (words >> word) ++word_count;
                double pause_seconds = word_count <= 4 ? 0.40 : 0.24;
                size_t pause = static_cast<size_t>(std::llround(CODEC_SAMPLE_RATE * pause_seconds));
                for (auto& chh : all) chh.insert(chh.end(), pause, 0.0f);
            }
            std::cout << "[chunk " << (i + 1) << "/" << chunks.size() << "] tokens="
                      << encode_text(chunks[i]).size() << " done" << std::endl;
        }
        return all;
    }

private:
    Ort::Env env_;
    int threads_;
    std::string model_dir_;
    std::string ffmpeg_; // ffmpeg.exe 路径（参考音频转 48kHz 用）
    json manifest_, tts_meta_, codec_meta_;

    Ort::Session* prefill_ = nullptr;
    Ort::Session* decode_ = nullptr;
    Ort::Session* local_fixed_ = nullptr;
    Ort::Session* codec_encode_ = nullptr;
    Ort::Session* codec_decode_ = nullptr;
    Ort::Session* codec_decode_step_ = nullptr;

    sentencepiece::SentencePieceProcessor sp_;

    // NumPy 2.x PCG64（default_rng(1234) 的 bit generator）兼容实现。
    // 初始 state/inc 由 SeedSequence(1234) 在 Python 中导出，seed 固定为 1234。
    struct U128 { uint64_t lo; uint64_t hi; };
    U128 pcg_state_{0xf69b873d9fe45409ULL, 0x160ad84006fe21eaULL};
    static constexpr U128 pcg_inc_{0xd0f51ce6006e4325ULL, 0x50c8fb163c7cea4eULL};
    static constexpr U128 pcg_mult_{0x4385df649fccf645ULL, 0x2360ed051fc65da4ULL};

    static U128 pcg_add_(U128 a, U128 b) {
        uint64_t lo = a.lo + b.lo;
        uint64_t hi = a.hi + b.hi + (lo < b.lo ? 1 : 0);
        return {lo, hi};
    }
    static uint64_t pcg_rotr_(uint64_t x, unsigned r) {
        return r ? ((x >> r) | (x << (64 - r))) : x;
    }
    float next_random_() {
        // numpy: pcg_setseq_128_xsl_rr_64_random_r 先 step 后 output
        uint64_t hi_temp;
        uint64_t plo = _umul128(pcg_state_.lo, pcg_mult_.lo, &hi_temp);
        uint64_t h1 = pcg_state_.hi * pcg_mult_.lo + pcg_state_.lo * pcg_mult_.hi;
        uint64_t ph = hi_temp + h1;
        pcg_state_ = pcg_add_({plo, ph}, pcg_inc_);
        uint64_t out = pcg_rotr_(pcg_state_.hi ^ pcg_state_.lo,
                                 static_cast<unsigned>(pcg_state_.hi >> 58));
        double d = static_cast<double>(out >> 11) * (1.0 / 9007199254740992.0);
        return static_cast<float>(std::min(0.99999994, std::max(0.0, d)));
    }

    std::vector<std::string> prefill_out_names_, decode_in_names_, decode_out_names_;
    std::string manifest_dir_, tts_dir_, codec_dir_; // 各配置文件所在目录

    std::string join_path_(const std::string& dir, const std::string& rel) {
        std::string p = rel;
        std::replace(p.begin(), p.end(), '/', '\\');
        return dir + p;
    }
    // 目录 + 末尾分隔符
    static std::string parent_dir_(const std::string& path) {
        size_t pos = path.find_last_of("\\/");
        return path.substr(0, pos + 1);
    }

    void load_configs_() {
        // 候选 manifest 路径（对应 Python MANIFEST_CANDIDATE_RELATIVE_PATHS）
        const char* candidates[] = {
            "browser_poc_manifest.json",
            "MOSS-TTS-Nano-100M-ONNX\\browser_poc_manifest.json",
            "MOSS-TTS-Nano-ONNX-CPU\\browser_poc_manifest.json",
        };
        std::string manifest_path;
        for (const char* c : candidates) {
            std::string p = model_dir_ + c;
            std::ifstream f(p);
            if (f.good()) {
                manifest_path = p;
                break;
            }
        }
        if (manifest_path.empty())
            throw std::runtime_error("browser_poc_manifest.json not found under " + model_dir_);
        manifest_dir_ = parent_dir_(manifest_path);

        manifest_ = json::parse(std::ifstream(manifest_path));
        std::string tts_meta_path = join_path_(manifest_dir_, manifest_["model_files"]["tts_meta"].get<std::string>());
        std::string codec_meta_path = join_path_(manifest_dir_, manifest_["model_files"]["codec_meta"].get<std::string>());
        tts_dir_ = parent_dir_(tts_meta_path);
        codec_dir_ = parent_dir_(codec_meta_path);
        tts_meta_ = json::parse(std::ifstream(tts_meta_path));
        codec_meta_ = json::parse(std::ifstream(codec_meta_path));
        prefill_out_names_ = tts_meta_["onnx"]["prefill_output_names"].get<std::vector<std::string>>();
        decode_in_names_ = tts_meta_["onnx"]["decode_input_names"].get<std::vector<std::string>>();
        decode_out_names_ = tts_meta_["onnx"]["decode_output_names"].get<std::vector<std::string>>();
    }

    Ort::Session* make_session_(const std::string& dir, const std::string& rel_path) {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(threads_);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(ORT_ENABLE_ALL); // C 枚举，不带 Ort:: 前缀
        std::string full = join_path_(dir, rel_path);
        std::wstring wfull = utf8_to_wchar(full);
        return new Ort::Session(env_, wfull.c_str(), so);
    }

    void load_sessions_() {
        const auto& files = tts_meta_["files"];
        prefill_ = make_session_(tts_dir_, files["prefill"].get<std::string>());
        decode_ = make_session_(tts_dir_, files["decode_step"].get<std::string>());
        local_fixed_ = make_session_(tts_dir_, files["local_fixed_sampled_frame"].get<std::string>());
        const auto& cfiles = codec_meta_["files"];
        codec_encode_ = make_session_(codec_dir_, cfiles["encode"].get<std::string>());
        codec_decode_ = make_session_(codec_dir_, cfiles["decode_full"].get<std::string>());
        codec_decode_step_ = make_session_(codec_dir_, cfiles["decode_step"].get<std::string>());
    }

    // ---- tensor 构造与推理辅助 ----
    // CreateTensor 只包装数据指针不拷贝，数据必须活到 Run 完成；
    // 用 owned_* 成员持有（Run 为同步调用，每次 Run 前 clear 即可）
    std::vector<std::vector<int32_t>> owned_i32_;
    std::vector<std::vector<float>> owned_f32_;
    void clear_owned_() {
        owned_i32_.clear();
        owned_f32_.clear();
    }

    Ort::Value int32_tensor(std::vector<int> data, const std::vector<int64_t>& shape) {
        owned_i32_.emplace_back(data.begin(), data.end());
        auto* p = owned_i32_.back().data();
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return Ort::Value::CreateTensor<int32_t>(mem, p, owned_i32_.back().size(),
                                                 shape.data(), shape.size());
    }
    Ort::Value float_tensor(std::vector<float> data, const std::vector<int64_t>& shape) {
        owned_f32_.push_back(std::move(data));
        auto* p = owned_f32_.back().data();
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return Ort::Value::CreateTensor<float>(mem, p, owned_f32_.back().size(),
                                               shape.data(), shape.size());
    }

    // 运行 session 并返回输出（按模型输出顺序）
    // 注意：GetInputNameAllocated 返回 RAII 指针，必须保活到 Run 完成（否则悬空）
    std::vector<Ort::Value> run_(Ort::Session* sess, std::vector<Ort::Value>&& inputs) {
        Ort::AllocatorWithDefaultOptions alloc;
        std::vector<Ort::AllocatedStringPtr> in_own, out_own;
        std::vector<const char*> in_names, out_names;
        for (size_t i = 0; i < sess->GetInputCount(); ++i) {
            in_own.push_back(sess->GetInputNameAllocated(i, alloc));
            in_names.push_back(in_own.back().get());
        }
        for (size_t i = 0; i < sess->GetOutputCount(); ++i) {
            out_own.push_back(sess->GetOutputNameAllocated(i, alloc));
            out_names.push_back(out_own.back().get());
        }
        return sess->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(),
                         out_names.data(), out_names.size());
    }

    // ---- 文本 tokenize ----
    std::vector<int> encode_text(const std::string& text) {
        std::vector<int> ids;
        auto st = sp_.Encode(text, &ids);
        if (!st.ok()) throw std::runtime_error("sentencepiece encode failed: " + st.ToString());
        return ids;
    }

    // ---- 参考音频编码 ----
    std::vector<std::vector<int>> encode_reference_audio(const std::string& wav_path) {
        std::string input = wav_path;
        std::string temp;
        // 参考音频统一为 48kHz 双声道（与 Python/torchaudio 参考处理一致）。
        // 采样率或声道不匹配时，优先用 ffmpeg 转换，避免内置 resample 与 torchaudio 数值差异导致音质偏差。
        int sr = 0, ch = 0;
        probe_wav_format_(wav_path, sr, ch);
        if (sr != CODEC_SAMPLE_RATE || ch != CODEC_CHANNELS) {
            if (!ffmpeg_.empty()) {
                wchar_t tmpdir[1024];
                DWORD tmp_len = GetTempPathW(1024, tmpdir);
                std::string tmpfile = (tmp_len ? utf8_from_wchar(tmpdir) : ".")
                    + "moss_tts_ref_convert.wav";
                temp = convert_audio_48k_stereo_(ffmpeg_, wav_path, tmpfile);
                if (!temp.empty()) input = temp;
            }
            if (input == wav_path)
                std::cerr << "warning: reference audio not 48kHz/stereo and ffmpeg unavailable; "
                             "falling back to built-in resampler (may differ from Python)" << std::endl;
        }
        WavData wav = read_wav(input);
        auto channels = resample_channels(wav.data, wav.sample_rate, CODEC_SAMPLE_RATE);
        if (!temp.empty()) _wremove(utf8_to_wchar(temp).c_str()); // 删除临时转换文件
        if (channels.empty()) throw std::runtime_error("reference wav has no channels");
        int current_channels = static_cast<int>(channels.size());
        if (current_channels == CODEC_CHANNELS) {
            // already in the codec layout
        } else if (current_channels == 1 && CODEC_CHANNELS > 1) {
            auto mono = channels[0];
            channels.assign(CODEC_CHANNELS, mono);
        } else if (current_channels > 1 && CODEC_CHANNELS == 1) {
            std::vector<float> mono(channels[0].size(), 0.0f);
            for (const auto& channel : channels)
                for (size_t i = 0; i < mono.size(); ++i) mono[i] += channel[i];
            for (auto& sample : mono) sample /= static_cast<float>(current_channels);
            channels.assign(1, std::move(mono));
        } else {
            throw std::runtime_error("unsupported reference audio channel conversion: " +
                                     std::to_string(current_channels) + " -> " +
                                     std::to_string(CODEC_CHANNELS));
        }
        size_t n = channels[0].size();
        // 组装 channel-major [1, 2, N] float32（与 torchaudio 布局一致）
        std::vector<float> chmaj(n * CODEC_CHANNELS);
        for (int c = 0; c < CODEC_CHANNELS; ++c)
            for (size_t i = 0; i < n; ++i)
                chmaj[c * n + i] = channels[c][i];

        clear_owned_();
        std::vector<Ort::Value> inputs;
        inputs.push_back(float_tensor(std::move(chmaj), {1, CODEC_CHANNELS, static_cast<int64_t>(n)}));
        inputs.push_back(int32_tensor({static_cast<int>(n)}, {1}));
        auto out = run_(codec_encode_, std::move(inputs));
        auto* codes_t = out[0].GetTensorData<int32_t>();
        auto code_len = out[1].GetTensorData<int32_t>()[0];
        int num_q = codec_meta_["codec_config"]["num_quantizers"].get<int>();
        if (code_len <= 0 || code_len > static_cast<int>(wav.data[0].size()))
            throw std::runtime_error("codec encode returned invalid audio_code_lengths");
        std::vector<std::vector<int>> codes(code_len, std::vector<int>(num_q));
        for (int i = 0; i < code_len; ++i)
            for (int q = 0; q < num_q; ++q)
                codes[i][q] = codes_t[i * num_q + q];
        std::cout << "reference audio codes: " << code_len << " frames" << std::endl;
        return codes;
    }

    // ---- request rows 组装（对照 build_voice_clone_request_rows） ----
    std::vector<std::vector<int>> build_request_rows(const std::vector<std::vector<int>>& prompt_codes,
                                                     const std::vector<int>& text_ids) {
        auto cfg = manifest_["tts_config"];
        int pad = cfg["audio_pad_token_id"].get<int>();
        int start = cfg["audio_start_token_id"].get<int>();
        int end = cfg["audio_end_token_id"].get<int>();
        int slot = cfg["audio_user_slot_token_id"].get<int>();
        auto prefix = manifest_["prompt_templates"]["user_prompt_prefix_token_ids"].get<std::vector<int>>();
        auto after_ref = manifest_["prompt_templates"]["user_prompt_after_reference_token_ids"].get<std::vector<int>>();
        auto asst_prefix = manifest_["prompt_templates"]["assistant_prompt_prefix_token_ids"].get<std::vector<int>>();

        auto text_row = [&](int tid) {
            std::vector<int> row(ROW_WIDTH, pad);
            row[0] = tid;
            return row;
        };
        auto audio_row = [&](const std::vector<int>& code) {
            std::vector<int> row(ROW_WIDTH, pad);
            row[0] = slot;
            for (size_t i = 0; i < code.size() && i < static_cast<size_t>(N_VQ); ++i) row[i + 1] = code[i];
            return row;
        };

        std::vector<std::vector<int>> rows;
        for (int tid : prefix) rows.push_back(text_row(tid));
        rows.push_back(text_row(start));
        for (const auto& code : prompt_codes) rows.push_back(audio_row(code));
        rows.push_back(text_row(end));
        for (int tid : after_ref) rows.push_back(text_row(tid));
        for (int tid : text_ids) rows.push_back(text_row(tid));
        for (int tid : asst_prefix) rows.push_back(text_row(tid));
        rows.push_back(text_row(start));
        return rows;
    }

    // ---- prefill：得到 global_hidden 与 12 层 KV cache ----
    void run_prefill(const std::vector<std::vector<int>>& rows,
                     std::vector<float>& global_hidden,
                     std::vector<std::vector<float>>& past,
                     int& past_valid) {
        int seq = static_cast<int>(rows.size());
        std::vector<int> input_ids(seq * ROW_WIDTH);
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < ROW_WIDTH; ++j)
                input_ids[i * ROW_WIDTH + j] = rows[i][j];
        std::vector<int> mask(seq, 1);

        clear_owned_();
        std::vector<Ort::Value> inputs;
        inputs.push_back(int32_tensor(std::move(input_ids), {1, seq, ROW_WIDTH}));
        inputs.push_back(int32_tensor(std::move(mask), {1, seq}));
        auto out = run_(prefill_, std::move(inputs));

        auto named = named_outputs_(prefill_, out);
        auto* gh = named["global_hidden"].GetTensorData<float>();
        global_hidden.assign(gh + (seq - 1) * HIDDEN_SIZE, gh + seq * HIDDEN_SIZE);

        past.clear();
        past.resize(2 * GLOBAL_LAYERS);
        int64_t kv_len = static_cast<int64_t>(seq);
        for (int l = 0; l < GLOBAL_LAYERS; ++l) {
            std::string kname = "present_key_" + std::to_string(l);
            std::string vname = "present_value_" + std::to_string(l);
            const auto* k = named[kname].GetTensorData<float>();
            const auto* v = named[vname].GetTensorData<float>();
            size_t elems = static_cast<size_t>(kv_len * GLOBAL_HEADS * HEAD_DIM);
            past[2 * l].assign(k, k + elems);
            past[2 * l + 1].assign(v, v + elems);
        }
        past_valid = seq;
        std::cout << "prefill seq=" << seq << " done" << std::endl;
    }

    // 按名字组织输出的辅助（含名字缓存）
    std::vector<std::string> in_names_cache_, out_names_cache_;
    std::vector<const char*> in_cstr_, out_cstr_;
    void refresh_name_caches_(Ort::Session* sess) {
        Ort::AllocatorWithDefaultOptions alloc;
        in_names_cache_.clear();
        out_names_cache_.clear();
        for (size_t i = 0; i < sess->GetInputCount(); ++i)
            in_names_cache_.push_back(sess->GetInputNameAllocated(i, alloc).get());
        for (size_t i = 0; i < sess->GetOutputCount(); ++i)
            out_names_cache_.push_back(sess->GetOutputNameAllocated(i, alloc).get());
        in_cstr_.clear();
        out_cstr_.clear();
        for (auto& n : in_names_cache_) in_cstr_.push_back(n.c_str());
        for (auto& n : out_names_cache_) out_cstr_.push_back(n.c_str());
    }
    std::vector<Ort::Value> run_named_(Ort::Session* sess, std::vector<Ort::Value>&& inputs) {
        refresh_name_caches_(sess);
        return sess->Run(Ort::RunOptions{nullptr}, in_cstr_.data(), inputs.data(), inputs.size(),
                         out_cstr_.data(), out_cstr_.size());
    }
    std::map<std::string, Ort::Value> named_outputs_(Ort::Session* sess, std::vector<Ort::Value>& out) {
        refresh_name_caches_(sess);
        std::map<std::string, Ort::Value> m;
        for (size_t i = 0; i < out.size(); ++i) m[out_names_cache_[i]] = std::move(out[i]);
        return m;
    }

    std::vector<std::vector<float>> decode_streaming_audio(const std::vector<std::vector<int>>& generated) {
        if (generated.empty()) return {};
        const auto& streaming = codec_meta_["streaming_decode"];
        std::map<std::string, std::vector<int>> int_state;
        std::map<std::string, std::vector<float>> float_state;
        auto shape_size = [](const json& shape) {
            size_t size = 1;
            for (const auto& dim : shape) size *= static_cast<size_t>(dim.get<int64_t>());
            return size;
        };
        for (const auto& spec : streaming["transformer_offsets"]) {
            int_state[spec["input_name"].get<std::string>()] = std::vector<int>(shape_size(spec["shape"]), 0);
        }
        for (const auto& spec : streaming["attention_caches"]) {
            int_state[spec["offset_input_name"].get<std::string>()] = std::vector<int>(shape_size(spec["offset_shape"]), 0);
            int_state[spec["cached_positions_input_name"].get<std::string>()] = std::vector<int>(shape_size(spec["positions_shape"]), -1);
            float_state[spec["cached_keys_input_name"].get<std::string>()] = std::vector<float>(shape_size(spec["cache_shape"]), 0.0f);
            float_state[spec["cached_values_input_name"].get<std::string>()] = std::vector<float>(shape_size(spec["cache_shape"]), 0.0f);
        }
        std::vector<std::vector<float>> wave(CODEC_CHANNELS);
        const auto input_names = codec_meta_["onnx"]["decode_step_input_names"].get<std::vector<std::string>>();
        for (const auto& frame : generated) {
            clear_owned_();
            std::vector<Ort::Value> inputs;
            std::vector<int> codes(N_VQ, 0);
            for (int q = 0; q < N_VQ && q < static_cast<int>(frame.size()); ++q) codes[q] = frame[q];
            for (const auto& name : input_names) {
                if (name == "audio_codes") {
                    inputs.push_back(int32_tensor(codes, {1, 1, N_VQ}));
                } else if (name == "audio_code_lengths") {
                    inputs.push_back(int32_tensor({1}, {1}));
                } else if (int_state.count(name)) {
                    const auto& values = int_state[name];
                    std::vector<int64_t> shape;
                    for (const auto& spec : streaming["transformer_offsets"])
                        if (spec["input_name"].get<std::string>() == name) shape = spec["shape"].get<std::vector<int64_t>>();
                    for (const auto& spec : streaming["attention_caches"])
                        if (spec["offset_input_name"].get<std::string>() == name) shape = spec["offset_shape"].get<std::vector<int64_t>>();
                    for (const auto& spec : streaming["attention_caches"])
                        if (spec["cached_positions_input_name"].get<std::string>() == name) shape = spec["positions_shape"].get<std::vector<int64_t>>();
                    inputs.push_back(int32_tensor(values, shape));
                } else {
                    const auto& values = float_state[name];
                    std::vector<int64_t> shape;
                    for (const auto& spec : streaming["attention_caches"])
                        if (spec["cached_keys_input_name"].get<std::string>() == name || spec["cached_values_input_name"].get<std::string>() == name) shape = spec["cache_shape"].get<std::vector<int64_t>>();
                    inputs.push_back(float_tensor(values, shape));
                }
            }
            auto out = run_named_(codec_decode_step_, std::move(inputs));
            auto named = named_outputs_(codec_decode_step_, out);
            int audio_len = named["audio_lengths"].GetTensorData<int32_t>()[0];
            if (audio_len < 0) throw std::runtime_error("codec streaming returned negative audio length");
            if (audio_len > 0) {
                const auto* audio = named["audio"].GetTensorData<float>();
                for (int c = 0; c < CODEC_CHANNELS; ++c)
                    wave[c].insert(wave[c].end(), audio + c * audio_len, audio + (c + 1) * audio_len);
            }
            for (const auto& spec : streaming["transformer_offsets"]) {
                auto in_name = spec["input_name"].get<std::string>();
                auto out_name = spec["output_name"].get<std::string>();
                const auto* p = named[out_name].GetTensorData<int32_t>();
                int_state[in_name].assign(p, p + int_state[in_name].size());
            }
            for (const auto& spec : streaming["attention_caches"]) {
                auto update_i32 = [&](const char* input_key, const char* output_key) {
                    auto in_name = spec[input_key].get<std::string>();
                    auto out_name = spec[output_key].get<std::string>();
                    const auto* p = named[out_name].GetTensorData<int32_t>();
                    int_state[in_name].assign(p, p + int_state[in_name].size());
                };
                auto update_f32 = [&](const char* input_key, const char* output_key) {
                    auto in_name = spec[input_key].get<std::string>();
                    auto out_name = spec[output_key].get<std::string>();
                    const auto* p = named[out_name].GetTensorData<float>();
                    float_state[in_name].assign(p, p + float_state[in_name].size());
                };
                update_i32("offset_input_name", "offset_output_name");
                update_i32("cached_positions_input_name", "cached_positions_output_name");
                update_f32("cached_keys_input_name", "cached_keys_output_name");
                update_f32("cached_values_input_name", "cached_values_output_name");
            }
        }
        return wave;
    }

    // ---- 单 chunk 合成 ----
    std::vector<std::vector<float>> synthesize_chunk(const std::string& text,
                                                     const std::vector<std::vector<int>>& prompt_codes) {
        auto text_ids = encode_text(text);
        auto rows = build_request_rows(prompt_codes, text_ids);

        std::vector<float> global_hidden;
        std::vector<std::vector<float>> past;
        int past_valid = 0;
        run_prefill(rows, global_hidden, past, past_valid);

        auto cfg = manifest_["tts_config"];
        int pad = cfg["audio_pad_token_id"].get<int>();
        int slot = cfg["audio_assistant_slot_token_id"].get<int>();
        int max_frames = manifest_["generation_defaults"]["max_new_frames"].get<int>();

        std::vector<int> rep_mask(N_VQ * AUDIO_CODEBOOK_SIZE, 0);
        std::vector<std::vector<int>> generated;

        for (int step = 0; step < max_frames; ++step) {
            // local_fixed_sampled_frame（采样在模型内完成，主机提供随机数）
            std::vector<float> asst_u(1, next_random_());
            std::vector<float> audio_u(N_VQ);
            for (auto& u : audio_u) u = next_random_();

            clear_owned_();
            std::vector<Ort::Value> inputs;
            inputs.push_back(float_tensor(std::move(global_hidden), {1, HIDDEN_SIZE}));
            inputs.push_back(int32_tensor(rep_mask, {1, N_VQ, AUDIO_CODEBOOK_SIZE}));
            inputs.push_back(float_tensor(std::move(asst_u), {1}));
            inputs.push_back(float_tensor(std::move(audio_u), {1, N_VQ}));
            auto out = run_named_(local_fixed_, std::move(inputs));
            auto o = named_outputs_(local_fixed_, out);
            bool should_continue = o["should_continue"].GetTensorData<int32_t>()[0] != 0;
            auto* frame_ids = o["frame_token_ids"].GetTensorData<int32_t>();
            if (!should_continue) break;

            std::vector<int> frame(frame_ids, frame_ids + N_VQ);
            generated.push_back(frame);
            for (int q = 0; q < N_VQ; ++q)
                if (frame[q] >= 0 && frame[q] < AUDIO_CODEBOOK_SIZE)
                    rep_mask[q * AUDIO_CODEBOOK_SIZE + frame[q]] = 1;

            // decode_step：更新 global_hidden 与 KV cache
            std::vector<int> next_row(ROW_WIDTH, pad);
            next_row[0] = slot;
            for (int q = 0; q < N_VQ; ++q) next_row[q + 1] = frame[q];

            clear_owned_();
            std::vector<Ort::Value> dinputs;
            dinputs.push_back(int32_tensor(std::move(next_row), {1, 1, ROW_WIDTH}));
            dinputs.push_back(int32_tensor({past_valid}, {1}));
            for (int l = 0; l < GLOBAL_LAYERS; ++l) {
                int64_t L = static_cast<int64_t>(past_valid);
                dinputs.push_back(float_tensor(std::move(past[2 * l]), {1, L, GLOBAL_HEADS, HEAD_DIM}));
                dinputs.push_back(float_tensor(std::move(past[2 * l + 1]), {1, L, GLOBAL_HEADS, HEAD_DIM}));
            }
            auto dout = run_named_(decode_, std::move(dinputs));
            auto dn = named_outputs_(decode_, dout);
            auto* gh = dn["global_hidden"].GetTensorData<float>();
            global_hidden.assign(gh, gh + HIDDEN_SIZE); // [1,1,768] -> [768]
            past_valid += 1;
            for (int l = 0; l < GLOBAL_LAYERS; ++l) {
                std::string kname = "present_key_" + std::to_string(l);
                std::string vname = "present_value_" + std::to_string(l);
                int64_t L = static_cast<int64_t>(past_valid);
                size_t elems = static_cast<size_t>(L * GLOBAL_HEADS * HEAD_DIM);
                const auto* k = dn[kname].GetTensorData<float>();
                const auto* v = dn[vname].GetTensorData<float>();
                past[2 * l].assign(k, k + elems);
                past[2 * l + 1].assign(v, v + elems);
            }
        }
        std::cout << "generated frames: " << generated.size() << std::endl;
        if (!generated.empty()) {
            std::cout << "first frame:";
            for (int v : generated[0]) std::cout << " " << v;
            std::cout << std::endl;
            std::cout << "last frame:";
            for (int v : generated.back()) std::cout << " " << v;
            std::cout << std::endl;
        }
        if (generated.empty()) return {};

        // Python CLI 默认使用 streaming=True；保持同一路径以避免 full/step 解码边界差异。
        return decode_streaming_audio(generated);
    }

    // ---- 长文本分块（简化版 split_voice_clone_text） ----
    static std::string utf8_char_at(const std::string& s, size_t i) {
        if (i >= s.size()) return {};
        size_t start = i;
        while (start > 0 && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) --start;
        unsigned char c = static_cast<unsigned char>(s[start]);
        size_t len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        return s.substr(start, std::min(len, s.size() - start));
    }

    std::vector<std::string> split_text_by_token_budget(const std::string& text, int max_tokens) {
        std::vector<std::string> pieces;
        std::string remaining = text;
        while (!remaining.empty()) {
            if (static_cast<int>(encode_text(remaining).size()) <= max_tokens) {
                pieces.push_back(remaining);
                break;
            }
            int lo = 1, hi = static_cast<int>(remaining.size());
            int best = 1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                std::string cand = remaining.substr(0, mid);
                if (static_cast<int>(encode_text(cand).size()) <= max_tokens) {
                    best = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            int cut = best;
            for (int i = best - 1; i >= std::max(0, best - 25); --i) {
                std::string ch = utf8_char_at(remaining, static_cast<size_t>(i));
                bool boundary = (ch.size() == 1 && (ch[0] == ' ' || ch[0] == ',' || ch[0] == ';' || ch[0] == ':')) ||
                                ch == "，" || ch == "、" || ch == "；" || ch == "：" ||
                                ch == "。" || ch == "！" || ch == "？";
                if (boundary) {
                    cut = i + 1;
                    break;
                }
            }
            if (cut <= 0) cut = best;
            std::string piece = normalize_tts_text(remaining.substr(0, static_cast<size_t>(cut)));
            if (piece.empty()) piece = remaining.substr(0, static_cast<size_t>(best));
            pieces.push_back(piece);
            remaining = normalize_tts_text(remaining.substr(static_cast<size_t>(cut)));
        }
        return pieces;
    }

    std::vector<std::string> split_voice_clone_text(const std::string& text, int max_tokens) {
        std::string prepared = normalize_tts_text(text);
        if (prepared.empty()) return {};
        if (static_cast<int>(encode_text(prepared).size()) <= max_tokens) return {prepared};

        // 按句子结束标点切句
        std::vector<std::string> sentences;
        std::string cur;
        Utf8Iter it(prepared);
        while (!it.done()) {
            std::string ch = it.next();
            cur += ch;
            if (is_sentence_end(ch)) {
                std::string s = normalize_tts_text(cur);
                if (!s.empty()) sentences.push_back(s);
                cur.clear();
            }
        }
        std::string tail = normalize_tts_text(cur);
        if (!tail.empty()) sentences.push_back(tail);

        // 句子按 token 预算合并成 chunk
        std::vector<std::string> chunks;
        std::string current;
        for (const auto& s : sentences) {
            int s_tok = static_cast<int>(encode_text(s).size());
            if (s_tok > max_tokens) {
                if (!current.empty()) {
                    chunks.push_back(current);
                    current.clear();
                }
                auto sub = split_text_by_token_budget(s, max_tokens);
                chunks.insert(chunks.end(), sub.begin(), sub.end());
                continue;
            }
            if (!current.empty() && static_cast<int>(encode_text(current + s).size()) > max_tokens) {
                chunks.push_back(current);
                current = s;
            } else {
                current = current.empty() ? s : current + s;
            }
        }
        if (!current.empty()) chunks.push_back(current);
        return chunks;
    }
};

// ---------------------------------------------------------------------------
// 命令行入口
// ---------------------------------------------------------------------------
// 宽字符入口：Windows 命令行参数非 ASCII 文本以 ANSI 传入会乱码，
// 因此用 wmain 获取宽字符参数并统一转为 UTF-8 处理（支持日语/中文文本）
// 无参数启动或 --interactive 进入交互模式（模型只加载一次，可反复合成）

// 简单 shell 风格分词（支持双引号/单引号包裹的值，如 --text "你好世界"）
static std::vector<std::wstring> tokenize_line_(const std::wstring& line) {
    std::vector<std::wstring> tokens;
    std::wstring cur;
    bool in_dq = false, in_sq = false;
    for (wchar_t ch : line) {
        if (in_dq) {
            if (ch == L'"') in_dq = false;
            else cur += ch;
        } else if (in_sq) {
            if (ch == L'\'') in_sq = false;
            else cur += ch;
        } else if (ch == L'"') {
            in_dq = true;
        } else if (ch == L'\'') {
            in_sq = true;
        } else if (ch == L' ' || ch == L'\t') {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
        } else {
            cur += ch;
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

static bool is_known_param_(const std::wstring& t) {
    static const wchar_t* kKnown[] = {
        L"prompt-audio-path", L"p", L"text", L"t", L"text-file",
        L"output", L"o", L"model-dir", L"ffmpeg", L"threads", L"interactive",
        L"help", L"h",
    };
    for (const wchar_t* k : kKnown)
        if (t == k) return true;
    return false;
}

struct CliArgs_ {
    std::string prompt_wav, text, text_file, output, model_dir, ffmpeg;
    int threads = 0; // 0 = 自动（按 CPU 核数，上限 8）
    bool help = false;
    bool interactive = false;
};

static CliArgs_ parse_args_(const std::vector<std::string>& args) {
    CliArgs_ a;
    auto need = [&](int i, const char* aname) -> std::string {
        if (i + 1 >= static_cast<int>(args.size())) {
            std::cerr << "missing value for " << aname << std::endl;
            std::exit(1);
        }
        return args[i + 1];
    };
    for (int i = 1; i < static_cast<int>(args.size()); ++i) {
        const std::string& s = args[i];
        if (s == "--prompt-audio-path" || s == "-p") a.prompt_wav = need(i++, "prompt");
        else if (s == "--text" || s == "-t") a.text = need(i++, "text");
        else if (s == "--text-file") a.text_file = need(i++, "text-file");
        else if (s == "--output" || s == "-o") a.output = need(i++, "output");
        else if (s == "--model-dir") a.model_dir = need(i++, "model");
        else if (s == "--ffmpeg") a.ffmpeg = need(i++, "ffmpeg");
        else if (s == "--threads") a.threads = std::stoi(need(i++, "threads"));
        else if (s == "--interactive") a.interactive = true;
        else if (s == "--help" || s == "-h") a.help = true;
        else {
            std::cerr << "unknown arg: " << s << std::endl;
            std::exit(1);
        }
    }
    return a;
}

static void print_usage_() {
    std::cout
        << "usage: moss_tts_cpp --prompt-audio-path <wav> --text <text>|--text-file <utf8.txt>\n"
        << "       [--output <wav>] [--model-dir <dir>] [--ffmpeg <path>] [--threads n] [--interactive]\n"
        << "参考音频若非 48kHz/双声道，将自动调用 ffmpeg 转换为 48kHz 双声道 PCM16（--ffmpeg 指定路径，\n"
        << "否则查找环境变量 MOSS_FFMPEG 或 PATH；找不到时回退到内置重采样）\n"
        << "无参数启动进入交互模式：输入与命令行相同的参数（可省略 -- 前缀），\n"
        << "例如：prompt-audio-path ref.wav --text 文本.txt --output 1.wav\n"
        << "交互模式下输入 exit / quit / 退出 结束。\n";
}

// exe 所在目录向上两级 + models（例如 cpp-build\moss_tts_cpp.exe -> 项目根\models），
// 使普通版 exe 从任意工作目录运行都能找到模型
static std::string default_model_dir_(const std::string& exe_path) {
    size_t p1 = exe_path.find_last_of("\\/");
    if (p1 == std::string::npos) return "models";
    size_t p2 = exe_path.find_last_of("\\/", p1 - 1);
    if (p2 == std::string::npos) return "models";
    std::string base = exe_path.substr(0, p2);
    return base + "\\models";
}

static bool dir_exists_(const std::string& path) {
    DWORD attr = GetFileAttributesW(utf8_to_wchar(path).c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

static bool path_is_file_(const std::string& path) {
    DWORD attr = GetFileAttributesW(utf8_to_wchar(path).c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

static void ensure_parent_dir_(const std::string& path) {
    size_t pos = path.find_last_of("\\/");
    if (pos == std::string::npos || pos == 0) return;
    std::string dir = path.substr(0, pos);
    if (dir_exists_(dir)) return;
    ensure_parent_dir_(dir);
    CreateDirectoryW(utf8_to_wchar(dir).c_str(), nullptr);
}

static std::string load_text_file_(const std::string& path) {
    std::ifstream tf = open_ifstream(path);
    if (!tf) throw std::runtime_error("cannot open text file: " + path);
    std::stringstream ss;
    ss << tf.rdbuf();
    std::string text = ss.str();
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF)
        text.erase(0, 3); // 去 UTF-8 BOM
    return text;
}

// 执行一次合成；tts 非空时复用已加载的模型实例（交互模式），否则临时创建
static int synthesize_once_(const CliArgs_& a, MossTtsCpp* tts) {
    std::string text = a.text;
    if (!a.text_file.empty()) text = load_text_file_(a.text_file);
    if (a.prompt_wav.empty() || text.empty()) {
        std::cerr << "require --prompt-audio-path and --text or --text-file" << std::endl;
        return 1;
    }
    try {
        std::unique_ptr<MossTtsCpp> owned;
        MossTtsCpp* t = tts;
        if (!t) {
            owned.reset(new MossTtsCpp(a.model_dir, a.threads, a.ffmpeg));
            t = owned.get();
        }
        auto wave = t->synthesize(a.prompt_wav, text);
        if (wave.empty()) {
            std::cerr << "no audio generated" << std::endl;
            return 1;
        }
        ensure_parent_dir_(a.output); // 自动创建输出目录（如 generated_audio\）
        write_wav(a.output, wave, CODEC_SAMPLE_RATE);
        size_t total = wave[0].size();
        std::cout << "saved " << a.output << " sample_rate=" << CODEC_SAMPLE_RATE
                  << " samples=" << total
                  << " seconds=" << (total / static_cast<double>(CODEC_SAMPLE_RATE)) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

static int interactive_mode_(CliArgs_ base) {
    try {
        MossTtsCpp tts(base.model_dir, base.threads, base.ffmpeg); // 模型只加载一次
        std::cout
            << "=== MOSS-TTS 交互模式 ===\n"
            << "输入与命令行相同的参数，参数名可省略 -- 前缀：\n"
            << "  prompt-audio-path <wav> --text <utf8.txt> --output <wav>\n"
            << "  prompt-audio-path <wav> --text-file <utf8.txt> --output <wav>\n"
            << "--text 参数为文本文件路径（UTF-8 编码），避免控制台直接输入中文被截断\n"
            << "命令：help 查看帮助，exit/quit/退出 结束\n";
        for (;;) {
            std::cout << "> " << std::flush;
            std::wstring line;
            if (!std::getline(std::wcin, line)) {
                std::wcout << std::endl;
                break;
            }
            auto toks = tokenize_line_(line);
            if (toks.empty()) continue;
            const std::wstring& first = toks[0];
            if (first == L"exit" || first == L"quit" || first == L"q" || first == L"退出") {
                std::cout << "bye" << std::endl;
                break;
            }
            if (first == L"help" || first == L"h") { print_usage_(); continue; }
            std::vector<std::string> args;
            args.push_back("moss_tts_cpp"); // argv[0] 占位
            for (const auto& t : toks) {
                std::wstring a = t;
                if (!a.empty() && a[0] != L'-' && is_known_param_(a)) a = L"--" + a;
                args.push_back(utf8_from_wchar(a.c_str()));
            }
            CliArgs_ ca = parse_args_(args);
            ca.model_dir = base.model_dir;
            if (ca.output.empty()) ca.output = base.output;
            // 交互模式：--text 的值若为存在的文件则按文件读取（避免控制台中文输入截断）
            if (ca.text_file.empty() && !ca.text.empty() && path_is_file_(ca.text))
                ca.text = load_text_file_(ca.text);
            if (synthesize_once_(ca, &tts) == 0)
                std::cout << "done. 输入 exit 退出，或继续输入下一条参数。" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) args.push_back(utf8_from_wchar(argv[i]));

    // 自包含模式：先从 exe 自身解压权重到临时目录，并让延迟加载的 onnxruntime.dll 可被发现
    std::string self_dir;
    if (extract_self(args[0], self_dir)) {
        SetDllDirectoryW(utf8_to_wchar(self_dir).c_str());
    }

    CliArgs_ ca;
    try {
        ca = parse_args_(args);
    } catch (...) {
        return 1;
    }
    if (ca.help) {
        print_usage_();
        return 0;
    }
    if (ca.model_dir.empty()) {
        if (!self_dir.empty()) {
            ca.model_dir = self_dir + "\\models";
        } else {
            std::string cand = default_model_dir_(args[0]);
            ca.model_dir = dir_exists_(cand) ? cand : "models";
        }
    }
    if (ca.threads <= 0) ca.threads = default_threads_(); // 默认线程数按 CPU 核数
    if (ca.output.empty()) ca.output = "generated_audio/cpp_output.wav";

    if (ca.interactive || argc <= 1) return interactive_mode_(ca);
    return synthesize_once_(ca, nullptr);
}
