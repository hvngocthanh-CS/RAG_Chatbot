# 🪟 HƯỚNG DẪN CHẠY RAG CHATBOT TRÊN WINDOWS (KHÔNG DOCKER)

## ❌ VÌ SAO vLLM KHÔNG CHẠY ĐƯỢC TRÊN WINDOWS?

vLLM yêu cầu `uvloop` - một thư viện **chỉ hỗ trợ Linux/Unix**. Khi chạy trực tiếp trên Windows sẽ bị lỗi:
```
ModuleNotFoundError: No module named 'uvloop'
```

---

## ✅ GIẢI PHÁP THAY THẾ CHO WINDOWS

### **GIẢI PHÁP 1: OLLAMA (KHUYẾN NGHỊ) ⭐**

**Ưu điểm:**
- ✅ Native Windows, không cần Docker
- ✅ Hỗ trợ GPU (RTX 3060)
- ✅ OpenAI-compatible API
- ✅ Dễ cài đặt và sử dụng
- ✅ Nhiều model: phi3, llama, mistral, qwen

**Cài đặt:**

#### Bước 1: Tải Ollama
```powershell
# Tải từ website
https://ollama.ai/download

# Hoặc dùng winget
winget install Ollama.Ollama
```

#### Bước 2: Tải model
```powershell
# Model Phi-3 (tương đương vLLM config)
ollama pull phi3

# Hoặc model khác
ollama pull llama3.2
ollama pull qwen2.5
ollama pull mistral
```

#### Bước 3: Kiểm tra Ollama
```powershell
# List models
ollama list

# Test model
ollama run phi3 "Hello, who are you?"
```

#### Bước 4: Chạy RAG Chatbot với Ollama
```powershell
cd C:\thanhhuynh\Chatbot\rag_chatbot
scripts\start_ollama.bat
```

**Đã sửa file `.env`:**
- ✅ `LLM_PROVIDER=ollama`
- ✅ `OLLAMA_BASE_URL=http://localhost:11434/v1`
- ✅ `OLLAMA_MODEL=phi3`

---

### **GIẢI PHÁP 2: WSL2 + vLLM**

Chạy vLLM trong Linux subsystem trên Windows.

**Ưu điểm:**
- ✅ Có thể dùng vLLM như trên Linux
- ✅ Hỗ trợ GPU qua WSL2

**Nhược điểm:**
- ⚠️ Phức tạp hơn
- ⚠️ Cần cài WSL2 + CUDA drivers

**Cài đặt:**

#### Bước 1: Cài WSL2
```powershell
# PowerShell as Admin
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

#### Bước 2: Cài CUDA trong WSL
```bash
# Trong WSL Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

#### Bước 3: Cài vLLM trong WSL
```bash
# Trong WSL
pip install vllm

# Chạy vLLM
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-mini-4k-instruct \
    --trust-remote-code \
    --dtype float16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8001
```

#### Bước 4: Chạy API từ Windows
```powershell
# Windows PowerShell
cd C:\thanhhuynh\Chatbot\rag_chatbot
python scripts\run_server.py
```

**Sửa `.env`:**
```env
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8001/v1
```

---

### **GIẢI PHÁP 3: LM STUDIO**

Desktop app đơn giản cho Windows.

**Ưu điểm:**
- ✅ GUI thân thiện
- ✅ Hỗ trợ GPU
- ✅ OpenAI-compatible API
- ✅ Không cần code

**Cài đặt:**

#### Bước 1: Tải LM Studio
https://lmstudio.ai/

#### Bước 2: Tải model trong LM Studio
- Search "Phi-3"
- Download model

#### Bước 3: Start Local Server
- Tab "Local Server"
- Port: 1234 (hoặc 8001)
- Start Server

#### Bước 4: Sửa `.env`
```env
LLM_PROVIDER=openai
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_MODEL_NAME=phi-3-mini
OPENAI_API_KEY=not-needed
```

---

### **GIẢI PHÁP 4: TEXT GENERATION WEBUI**

**Ưu điểm:**
- ✅ Nhiều model support
- ✅ OpenAI API endpoint
- ✅ Web UI

**Cài đặt:**
```powershell
# Clone repo
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
start_windows.bat

# Enable OpenAI API extension
# Download model
# Start server với --api flag
```

---

## 🎯 KHUYẾN NGHỊ

| Giải pháp | Độ khó | Performance | GPU Support | Khuyến nghị |
|-----------|--------|-------------|-------------|-------------|
| **Ollama** | ⭐ Dễ | ⭐⭐⭐ Tốt | ✅ Có | **Tốt nhất cho Windows** |
| WSL2 + vLLM | ⭐⭐⭐ Khó | ⭐⭐⭐⭐ Rất tốt | ✅ Có | Nếu cần vLLM |
| LM Studio | ⭐ Dễ | ⭐⭐ Trung bình | ✅ Có | Nếu thích GUI |
| Text Gen WebUI | ⭐⭐ Trung bình | ⭐⭐⭐ Tốt | ✅ Có | Nếu cần nhiều model |
| **Docker** | ⭐⭐ Trung bình | ⭐⭐⭐⭐ Rất tốt | ✅ Có | **Production** |

---

## 🚀 CHẠY NGAY VỚI OLLAMA

```powershell
# 1. Cài Ollama
winget install Ollama.Ollama

# 2. Tải model
ollama pull phi3

# 3. Chạy chatbot
cd C:\thanhhuynh\Chatbot\rag_chatbot
scripts\start_ollama.bat
```

✅ **Đã sửa file `.env` để dùng Ollama!**

---

## 📊 SO SÁNH PERFORMANCE (RTX 3060 6GB)

| Metric | vLLM (Docker) | Ollama | LM Studio |
|--------|---------------|---------|-----------|
| Tokens/sec | ~40-50 | ~30-40 | ~25-35 |
| Latency | Thấp | Trung bình | Trung bình |
| Memory | 4.5GB VRAM | 4GB VRAM | 4.5GB VRAM |
| Setup time | 5 phút | 2 phút | 2 phút |
