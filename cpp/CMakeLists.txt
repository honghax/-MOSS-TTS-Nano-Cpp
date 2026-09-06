# MOSS-TTS-Nano C++ 推理项目（onnxruntime C++ API 直接实现）
# 依赖：onnxruntime C++ 库 + sentencepiece 静态库（均为本地相对路径）
cmake_minimum_required(VERSION 3.20)
project(moss_tts_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ORT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../cpp_deps/onnxruntime-win-x64-1.29.0")
set(SP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../sentencepiece-install")

add_executable(moss_tts_cpp
    src/main.cpp
)

target_include_directories(moss_tts_cpp PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}            # nlohmann_json.hpp
    ${ORT_ROOT}/include                  # onnxruntime_cxx_api.h
    ${SP_ROOT}/include                   # sentencepiece_processor.h
    ${CMAKE_CURRENT_LIST_DIR}/../sentencepiece-0.2.2/third_party/abseil-cpp  # absl/... (sentencepiece 头文件依赖)
)

target_link_libraries(moss_tts_cpp PRIVATE
    ${ORT_ROOT}/lib/onnxruntime.lib
    ${SP_ROOT}/lib/sentencepiece.lib
)

# sentencepiece 依赖 abseil（静态库，需在 sentencepiece.lib 之后链接）
file(GLOB_RECURSE ABSL_LIBS
    "${CMAKE_CURRENT_LIST_DIR}/../sentencepiece-build/third_party/abseil-cpp/absl/*/*.lib")
target_link_libraries(moss_tts_cpp PRIVATE ${ABSL_LIBS})

if(MSVC)
    target_compile_options(moss_tts_cpp PRIVATE /O2 /EHsc /utf-8)
    # onnxruntime.dll 延迟加载：自包含模式下 main 开头 SetDllDirectory 后再解析
    target_link_options(moss_tts_cpp PRIVATE "/DELAYLOAD:onnxruntime.dll" delayimp.lib)
    # 显式指定系统库路径（链接器环境的 LIB 变量可能被沙箱过滤，
    # 裸库名 kernel32/msvcprt 等会因找不到 .lib 而链接失败——参考 fitsplat_gl 写法）
    get_filename_component(_MSVC_ROOT "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_MSVC_ROOT "${_MSVC_ROOT}/../../.." ABSOLUTE)   # .../MSVC/<ver>
    target_link_options(moss_tts_cpp PRIVATE "/LIBPATH:${_MSVC_ROOT}/lib/x64")
    file(GLOB _WIN_KITS "C:/Program Files (x86)/Windows Kits/10/Lib/*")
    foreach(_kit ${_WIN_KITS})
        if(EXISTS "${_kit}/um/x64")
            target_link_options(moss_tts_cpp PRIVATE
                "/LIBPATH:${_kit}/um/x64"
                "/LIBPATH:${_kit}/ucrt/x64")
        endif()
    endforeach()
endif()

# 运行时需要 onnxruntime.dll（动态库），拷贝到可执行文件目录
add_custom_command(TARGET moss_tts_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_ROOT}/lib/onnxruntime.dll
        $<TARGET_FILE_DIR:moss_tts_cpp>
    COMMENT "copy onnxruntime.dll to output dir"
)
