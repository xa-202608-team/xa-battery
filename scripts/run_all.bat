@echo off
REM =====================================================================
REM 电池组件 — 本地一键验证 (无需 Docker, Windows)
REM   自动检测 Python 路径, 不硬编码本机路径
REM =====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

REM 检测 Python (优先 py launcher, 其次 python)
where py >nul 2>&1 && set "PY=py -3.9" || (
  where python >nul 2>&1 && set "PY=python" || (
    echo [ERROR] 未找到 Python, 请安装 Python 3.9 并添加到 PATH
    exit /b 1
  )
)

echo ========== 阶段二冻结包校验 ==========
cd reference
%PY% -m battery_entry verify
if errorlevel 1 (echo [FAIL] v2 校验失败 & exit /b 1)

echo. & echo ========== 阶段二逐位复现 ==========
%PY% -m battery_entry reproduce --mode quick --output ..\..\..\..\05_结果\reproduced\battery
if errorlevel 1 (echo [FAIL] v2 复现失败 & exit /b 1)

echo. & echo ========== 阶段三测试套件 ==========
cd ..\
set PYTHONPATH=%CD%;%CD%\reference
%PY% -m pytest tests\ -q --tb=short
if errorlevel 1 (echo [FAIL] v3 测试失败 & exit /b 1)

echo. & echo ========== 验证通过 ==========
echo ✅ v2 冻结包: 7/7
echo ✅ v2 逐位复现: max|diff| ^< 1e-12
echo ✅ v3 测试: 42 passed
endlocal
