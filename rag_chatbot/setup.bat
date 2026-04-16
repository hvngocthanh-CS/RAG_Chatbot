@echo off
REM ================================================================
REM RAG Chatbot - Complete Setup Script (Windows)
REM Production-ready with Ollama + GPU
REM ================================================================

setlocal enabledelayedexpansion

echo.
echo ==================================================
echo RAG Chatbot - Complete Setup (Ollama Mode)
echo ==================================================
echo.

REM Step 1: Update conda
echo [1/4] Updating Conda...
call conda update -n base -c conda-forge conda -y
if errorlevel 1 goto error

REM Step 2: Create/Activate environment
echo [2/4] Activating rag_chatbot environment...
call conda activate rag_chatbot
if errorlevel 1 goto error

REM Step 3: Install PyTorch CUDA 12.6
echo [3/4] Installing PyTorch CUDA 12.6...
echo This may take 5-10 minutes, please wait...
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto error

REM Step 4: Install project dependencies
echo [4/4] Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 goto error

REM Verification
echo.
echo ==================================================
echo Verifying GPU Setup...
echo ==================================================
python check_gpu.py

if errorlevel 1 goto error

echo.
echo ==================================================
echo ✓ Setup Complete!
echo ==================================================
echo.
echo Next steps:
echo.
echo 1. Verify .env configuration:
echo    - OLLAMA_MODEL=llama3.1:8b
echo    - OLLAMA_BASE_URL=http://localhost:11434/v1
echo.
echo 2. Start Ollama + RAG Backend:
echo    scripts\start_ollama.bat
echo.
echo 3. Test endpoints:
echo    curl http://localhost:8000/health
echo.
pause
goto end

:error
echo.
echo ❌ Setup failed! Check the error messages above.
echo.
pause
exit /b 1

:end
echo.
