# 🚀 HƯỚNG DẪN CHẠY HỆ THỐNG RAG CHATBOT (PRODUCTION)

## 📋 YÊU CẦU HỆ THỐNG

### Phần cứng:
- ✅ GPU NVIDIA (RTX 3060 trở lên)
- ✅ RAM: 16GB+
- ✅ Ổ cứng: 50GB trống

### Phần mềm:
- ✅ Windows 10/11
- ✅ Docker Desktop: https://www.docker.com/products/docker-desktop/
- ✅ NVIDIA Driver mới nhất: https://www.nvidia.com/download/index.aspx
- ✅ Python 3.10+
- ✅ Node.js 18+

---

## 🎯 CÁCH CHẠY (3 BƯỚC ĐƠN GIẢN)

### ✨ **CÁCH 1: CHẠY TỰ ĐỘNG (KHUYẾN NGHỊ)**

Mở PowerShell/CMD và chạy:

```bash
cd C:\thanhhuynh\Chatbot\rag_chatbot
scripts\start_production.bat
```

**Xong!** Script sẽ tự động:
1. ✅ Kiểm tra Docker & GPU
2. ✅ Start vLLM server (port 8001)
3. ✅ Đợi model load vào GPU
4. ✅ Start Backend API (port 8000)

---

### 🔧 **CÁCH 2: CHẠY TỪNG BƯỚC (MANUAL)**

#### **Bước 1: Start vLLM Server (GPU)**

**Terminal 1:**
```bash
cd C:\thanhhuynh\Chatbot\rag_chatbot
scripts\vllm_start.bat
```

**Đợi xuất hiện:**
```
✅ Container Status: Running
✅ HTTP Endpoint: Responding
✅ All Health Checks Passed!
```

⏱️ **Thời gian:** ~2-3 phút (model đang load vào GPU)

---

#### **Bước 2: Start Backend API**

**Terminal 2 (Mở cửa sổ mới):**
```bash
cd C:\thanhhuynh\Chatbot\rag_chatbot

# Kích hoạt môi trường (nếu dùng conda)
conda activate rag_chatbot

# Chạy server
python scripts/run_server.py
```

**Đợi xuất hiện:**
```
INFO: Uvicorn running on http://0.0.0.0:8000
✓ Embedding model loaded successfully
All services initialized successfully
```

---

#### **Bước 3: Start Frontend (Web UI)**

**Terminal 3 (Mở cửa sổ mới):**
```bash
cd C:\thanhhuynh\Chatbot\fe_rag_chatbot

# Cài dependencies (chỉ lần đầu)
npm install

# Chạy dev server
npm run dev
```

**Đợi xuất hiện:**
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

---

## 🌐 TRUY CẬP HỆ THỐNG

Mở trình duyệt:

| Dịch vụ | URL | Mô tả |
|---------|-----|-------|
| 🖥️ **Web UI** | http://localhost:5173 | Giao diện chat |
| 📚 **API Docs** | http://localhost:8000/api/v1/docs | Swagger UI |
| 🔍 **Health Check** | http://localhost:8000/health/live | Backend status |
| 🤖 **vLLM API** | http://localhost:8001/v1 | LLM endpoint |

---

## ✅ KIỂM TRA HỆ THỐNG

### 1. Kiểm tra GPU đang hoạt động:

```bash
nvidia-smi
```

**Kết quả mong đợi:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|   0  NVIDIA GeForce RTX 3060  On   | 00000000:01:00.0  On |                  N/A |
| 40%   52C    P2    45W / 170W |   4521MiB /  6144MiB |     15%      Default |
+-------------------------------+----------------------+----------------------+
```

👉 Kiểm tra **Memory-Usage** - Model đã load nếu thấy ~4-5GB đang dùng

---

### 2. Kiểm tra vLLM Server:

```bash
cd C:\thanhhuynh\Chatbot\rag_chatbot
scripts\vllm_health.bat
```

**Hoặc manual:**
```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

---

### 3. Kiểm tra Backend:

```bash
curl http://localhost:8000/health/live
```

**Kết quả mong đợi:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-21T...",
  "services": {
    "vector_store": "ok",
    "llm": "ok"
  }
}
```

---

### 4. Test upload & chat:

1. Mở http://localhost:5173
2. Click **"Upload Documents"**
3. Chọn file PDF/DOCX/TXT
4. Chờ upload thành công
5. Gõ câu hỏi: "What is in the document?"
6. Xem kết quả streaming

---

## 🛠️ TROUBLESHOOTING (GỠ LỖI)

### ❌ **Lỗi: "Docker is not running"**

**Giải pháp:**
```bash
# Mở Docker Desktop
# Chờ Docker khởi động xong
# Chạy lại script
```

---

### ❌ **Lỗi: "GPU not detected"**

**Giải pháp:**
```bash
# 1. Cài NVIDIA Driver mới nhất
# 2. Cài NVIDIA Container Toolkit:
#    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 3. Test GPU trong Docker:
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

### Lỗi: "vLLM model too large for GPU memory"

**Giải pháp cho RTX 3060:**
```env
VLLM_MAX_MODEL_LEN=1024  # Đã tối ưu cho 6GB
LLM_MAX_TOKENS=512       # Context output nhỏ hơn
VLLM_GPU_MEMORY_UTILIZATION=0.75
```

Restart:
```bash
scripts\vllm_stop.bat
scripts\vllm_start.bat
```

---

### ❌ **Lỗi: "Backend cannot connect to vLLM"**

**Nguyên nhân:** vLLM chưa sẵn sàng

**Giải pháp:**
```bash
# Kiểm tra vLLM:
curl http://localhost:8001/health

# Nếu lỗi 404/connection refused -> Đợi thêm 1-2 phút
# Model đang load vào GPU

# Xem logs:
docker logs vllm-server -f
```

---

### ❌ **Lỗi: "Port 8000/8001 already in use"**

**Giải pháp:**
```bash
# Tìm process đang dùng port:
netstat -ano | findstr :8000
netstat -ano | findstr :8001

# Kill process (dùng PID từ lệnh trên):
taskkill /PID <PID_NUMBER> /F

# Hoặc đổi port trong .env:
API_PORT=8002
```

---

### ❌ **Lỗi: "Out of Memory (OOM)"**

**Giải pháp cho RTX 3060 (6GB):**

Sửa file `C:\thanhhuynh\Chatbot\rag_chatbot\.env`:

```env
VLLM_GPU_MEMORY_UTILIZATION=0.60  # Giảm xuống 60%
VLLM_MAX_MODEL_LEN=1024           # Context ngắn hơn
```

Restart vLLM:
```bash
scripts\vllm_stop.bat
scripts\vllm_start.bat
```

---

## 📊 GIÁM SÁT HỆ THỐNG

### Xem GPU realtime:
```bash
nvidia-smi -l 1
```

### Xem logs vLLM:
```bash
docker logs vllm-server -f
```

### Xem logs Backend:
```bash
# Logs được in ra terminal đang chạy
# Hoặc xem file:
type logs\app.log
```

### Xem container đang chạy:
```bash
docker ps
```

---

## 🛑 TẮT HỆ THỐNG

### Tắt từng service:

```bash
# 1. Tắt Frontend: Ctrl+C trong terminal frontend
# 2. Tắt Backend: Ctrl+C trong terminal backend
# 3. Tắt vLLM:
cd C:\thanhhuynh\Chatbot\rag_chatbot
scripts\vllm_stop.bat
```

---

## 📈 TỐI ƯU HÓA PERFORMANCE

### Cho RTX 3060 (6GB VRAM):
```env
VLLM_GPU_MEMORY_UTILIZATION=0.8
VLLM_MAX_MODEL_LEN=1024
LLM_MAX_TOKENS=512
CHUNK_SIZE=500
TOP_K_RETRIEVAL=5
API_WORKERS=4
```

### Cho RTX 3090/4090 (24GB):
```env
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=4096
LLM_MAX_TOKENS=2048
CHUNK_SIZE=1000
TOP_K_RETRIEVAL=10
API_WORKERS=8
```

### Cho A100 (40GB):
```env
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_MAX_MODEL_LEN=8192
VLLM_TENSOR_PARALLEL_SIZE=1
API_WORKERS=16
```

---

## 🔄 CẬP NHẬT MODEL

### Đổi sang model khác:

Sửa `.env`:
```env
# Model nhỏ hơn (nhanh, dùng ít VRAM):
VLLM_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Model lớn hơn (chất lượng cao):
VLLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
```

Restart vLLM:
```bash
scripts\vllm_stop.bat
scripts\vllm_start.bat
```

---

## 📞 HỖ TRỢ

- 📖 Tài liệu chi tiết: `PRODUCTION_DEPLOY.md`
- 🔧 API Documentation: http://localhost:8000/api/v1/docs
- 📝 Quick Reference: `QUICKSTART.md`

---

## ✅ CHECKLIST TRƯỚC KHI CHẠY

- [ ] Docker Desktop đã cài và đang chạy
- [ ] NVIDIA Driver mới nhất (535.60+)
- [ ] GPU có ít nhất 6GB VRAM
- [ ] Python 3.10+ đã cài
- [ ] Node.js 18+ đã cài
- [ ] Port 8000, 8001, 5173 chưa bị dùng
- [ ] File `.env` đã cấu hình đúng
- [ ] Internet ổn định (để tải model lần đầu)

---

## 🎯 WORKFLOW HẰNG NGÀY

```bash
# Sáng đến văn phòng:
1. Mở Docker Desktop
2. Chạy: scripts\start_production.bat
3. Đợi 2-3 phút
4. Mở http://localhost:5173
5. Bắt đầu làm việc!

# Tối về nhà:
1. Ctrl+C các terminal
2. Chạy: scripts\vllm_stop.bat
3. Tắt Docker Desktop (optional)
```

---

**Chúc bạn sử dụng hệ thống hiệu quả! 🚀**

*Last Updated: 2026-03-21*
