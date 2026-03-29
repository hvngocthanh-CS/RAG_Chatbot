@echo off
REM ================================================================
REM Ollama + RAG Chatbot - Windows Native Setup
REM No Docker Required - GPU Supported
REM ================================================================

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║   RAG Chatbot - Ollama Mode (Windows Native)               ║
echo ║   No Docker Required - RTX 3060 GPU Optimized              ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check Ollama installation
echo [1/4] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Ollama is not installed!
    echo.
    echo 📥 Download and install from: https://ollama.ai/download
    echo    Or use: winget install Ollama.Ollama
    echo.
    pause
    exit /b 1
)
echo ✅ Ollama is installed

REM Check Ollama service
echo.
echo [2/4] Checking Ollama service...
curl -s http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama service not running. Starting...
    start "" ollama serve
    timeout /t 3 >nul
    echo ✅ Ollama service started
) else (
    echo ✅ Ollama service is running
)

REM Check if model exists
echo.
echo [3/4] Checking model 'phi3'...
ollama list | findstr phi3 >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Model 'phi3' not found. Downloading...
    echo    This will take ~2GB download, please wait...
    ollama pull phi3
    if errorlevel 1 (
        echo ❌ Failed to download model
        pause
        exit /b 1
    )
    echo ✅ Model downloaded successfully
) else (
    echo ✅ Model 'phi3' is ready
)

REM Start RAG API
echo.
echo [4/4] Starting RAG API Server...
echo    LLM Provider: Ollama
echo    Ollama URL: http://localhost:11434/v1
echo    API Port: 8000
echo.
echo ⚠️  Make sure .env has LLM_PROVIDER=ollama
echo.

cd /d "%~dp0\.."
python scripts\run_server.py

pause
