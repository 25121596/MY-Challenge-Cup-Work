@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."

echo ============================================================
echo   云边协同 AI — 一键演示
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 未安装或未加入 PATH
    pause
    exit /b 1
)

REM 检查依赖
python -c "import psutil" >nul 2>&1
if errorlevel 1 (
    echo [INFO] 安装依赖...
    pip install psutil
)

REM 检查 llama-server 是否已在运行
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)" >nul 2>&1
if errorlevel 1 (
    echo [1/5] 启动 llama-server...
    set LLAMA_SERVER=%~dp0..\..\..\llama\llama-server.exe
    set MODEL=%~dp0..\..\..\llama\Qwen_Qwen3-1.7B-IQ4_XS.gguf
    if not exist "%LLAMA_SERVER%" (
        echo [ERROR] llama-server.exe 未找到: %LLAMA_SERVER%
        pause
        exit /b 1
    )
    start /B "" "%LLAMA_SERVER%" -m "%MODEL%" --host 127.0.0.1 --port 8080 -ngl 0 -c 2048
    echo       等待模型加载...
    timeout /t 10 /nobreak >nul
) else (
    echo [1/5] llama-server 已在运行
)

echo.
echo [2/5] 预处理数据...
python preprocessing\hydraulic_preprocess.py
if errorlevel 1 (
    echo [WARN] 预处理失败, 尝试跳过...
)

echo.
echo [3/5] 运行端到端演示...
python demo\demo_controller.py --scenario industrial_inspection --edge-endpoint http://127.0.0.1:8080

echo.
echo [4/5] 运行场景执行器...
python scenarios\scenario_runner.py industrial_inspection --endpoint http://127.0.0.1:8080 --max-samples 100

echo.
echo [5/5] 查看结果...
echo   演示结果: demo_results.json
echo   场景结果: results\ 目录
echo.

echo ============================================================
echo   演示完成!
echo ============================================================
pause
