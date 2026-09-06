# MOSS-TTS-Nano C++ 项目一键构建脚本（Windows / MSVC 2022）
# 用法：powershell -ExecutionPolicy Bypass -File cpp\build_win.ps1
# 依赖：F:\vs 下的 MSVC 工具链 + Windows SDK + cmake/ninja（均来自 F:\vs）

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Cmake = "F:\vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$Ninja = "F:\vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

# 定位 MSVC 工具集（取最新版本）
$MSVCDir = Get-ChildItem "F:\vs\VC\Tools\MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1
$VSMSVC = $MSVCDir.FullName
# 定位 Windows SDK（取最新版本）
$SDKVer = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\Include" -Directory | Sort-Object Name -Descending | Select-Object -First 1
$SDK = "C:\Program Files (x86)\Windows Kits\10"

# 设置 MSVC 编译环境（等价于 vcvars64.bat 的关键变量）
$env:INCLUDE = "$VSMSVC\include;$SDK\Include\$($SDKVer.Name)\ucrt;$SDK\Include\$($SDKVer.Name)\um;$SDK\Include\$($SDKVer.Name)\shared;$SDK\Include\$($SDKVer.Name)\winrt"
$env:LIB = "$VSMSVC\lib\x64;$SDK\Lib\$($SDKVer.Name)\ucrt\x64;$SDK\Lib\$($SDKVer.Name)\um\x64"
$env:PATH = "$VSMSVC\bin\Hostx64\x64;$SDK\bin\$($SDKVer.Name)\x64;$env:PATH"

Write-Host "== configure sentencepiece =="
& $Cmake -S (Join-Path $Root "sentencepiece-0.2.2") -B (Join-Path $Root "sentencepiece-build") -G Ninja `
    -DCMAKE_MAKE_PROGRAM=$Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl `
    -DCMAKE_BUILD_TYPE=Release -DSPM_ENABLE_SHARED=OFF -DSPM_BUILD_TESTING=OFF `
    -DCMAKE_INSTALL_PREFIX=(Join-Path $Root "sentencepiece-install")
Write-Host "== build sentencepiece =="
& $Cmake --build (Join-Path $Root "sentencepiece-build")
Write-Host "== install sentencepiece =="
& $Cmake --install (Join-Path $Root "sentencepiece-build")

Write-Host "== configure moss_tts_cpp =="
& $Cmake -S (Join-Path $Root "cpp") -B (Join-Path $Root "cpp-build") -G Ninja `
    -DCMAKE_MAKE_PROGRAM=$Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl `
    -DCMAKE_BUILD_TYPE=Release
Write-Host "== build moss_tts_cpp =="
& $Cmake --build (Join-Path $Root "cpp-build")
Write-Host "Done. exe at: cpp-build\moss_tts_cpp.exe"
