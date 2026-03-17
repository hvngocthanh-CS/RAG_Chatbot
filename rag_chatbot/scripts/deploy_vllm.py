#!/usr/bin/env python3
"""
vLLM Server Deployment Script.

This script helps deploy and manage vLLM inference server
for the RAG chatbot system.

Usage:
    python scripts/deploy_vllm.py start --model <model_name>
    python scripts/deploy_vllm.py stop
    python scripts/deploy_vllm.py status
    python scripts/deploy_vllm.py health
"""
import os
import sys
import subprocess
import argparse
import time
import json
from typing import Optional

try:
    import requests
except ImportError:
    requests = None

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================
# Configuration
# ===========================================

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"

# Recommended models for RAG
RECOMMENDED_MODELS = {
    "phi-3-mini": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Fast, efficient 3.8B model. Good for constrained environments.",
        "min_gpu_memory": "8GB",
        "max_model_len": 4096
    },
    "llama-3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "description": "High quality 8B model from Meta.",
        "min_gpu_memory": "16GB",
        "max_model_len": 8192
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Excellent 7B model with good performance.",
        "min_gpu_memory": "16GB",
        "max_model_len": 32768
    },
    "qwen-2-7b": {
        "name": "Qwen/Qwen2-7B-Instruct",
        "description": "Strong multilingual model from Alibaba.",
        "min_gpu_memory": "16GB",
        "max_model_len": 32768
    },
    "gemma-2-9b": {
        "name": "google/gemma-2-9b-it",
        "description": "Google's latest instruction-tuned model.",
        "min_gpu_memory": "20GB",
        "max_model_len": 8192
    }
}


# ===========================================
# Helper Functions
# ===========================================

def check_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("GPU detected:")
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")
            return True
    except FileNotFoundError:
        pass
    return False


def check_docker() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_vllm_installed() -> bool:
    """Check if vLLM is installed."""
    try:
        import vllm
        return True
    except ImportError:
        return False


def get_vllm_health(host: str = "localhost", port: int = 8000) -> dict:
    """Check vLLM server health."""
    if requests is None:
        return {"status": "unknown", "error": "requests library not installed"}
    
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        return {"status": "healthy" if response.status_code == 200 else "unhealthy"}
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "error": "Server not reachable"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_available_models(host: str = "localhost", port: int = 8000) -> list:
    """Get list of available models from vLLM server."""
    if requests is None:
        return []
    
    try:
        response = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
    except:
        pass
    return []


# ===========================================
# Commands
# ===========================================

def start_vllm_docker(
    model: str,
    port: int = DEFAULT_PORT,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    quantization: Optional[str] = None,
    use_cpu: bool = False
):
    """Start vLLM server using Docker."""
    print(f"\n🚀 Starting vLLM server with model: {model}")
    
    # Build docker command
    image = "vllm/vllm-cpu:latest" if use_cpu else "vllm/vllm-openai:latest"
    
    cmd = [
        "docker", "run", "-d",
        "--name", "rag-vllm",
        "-p", f"{port}:8000",
        "-v", "vllm-cache:/root/.cache/huggingface",
    ]
    
    # Add GPU support
    if not use_cpu:
        cmd.extend(["--gpus", "all"])
    
    cmd.append(image)
    
    # vLLM arguments
    cmd.extend(["--model", model, "--trust-remote-code"])
    
    if use_cpu:
        cmd.extend(["--dtype", "float32"])
    else:
        cmd.extend([
            "--dtype", "auto",
            "--gpu-memory-utilization", str(gpu_memory_utilization)
        ])
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Stop existing container if any
    subprocess.run(["docker", "rm", "-f", "rag-vllm"], capture_output=True)
    
    # Start new container
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Container started successfully!")
        print("\n⏳ Waiting for model to load (this may take a few minutes)...")
        
        # Wait for server to be ready
        for i in range(60):  # 5 minutes timeout
            time.sleep(5)
            health = get_vllm_health("localhost", port)
            if health["status"] == "healthy":
                print(f"\n✅ vLLM server is ready!")
                print(f"   API endpoint: http://localhost:{port}/v1")
                return True
            print(f"   Loading... ({(i+1)*5}s)")
        
        print("⚠️  Server started but health check timed out.")
        print("   Check logs with: docker logs rag-vllm")
    else:
        print(f"❌ Failed to start container: {result.stderr}")
    
    return False


def start_vllm_native(
    model: str,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    quantization: Optional[str] = None
):
    """Start vLLM server natively (requires vLLM installed)."""
    if not check_vllm_installed():
        print("❌ vLLM is not installed. Install with: pip install vllm")
        print("   Or use Docker: python scripts/deploy_vllm.py start --docker")
        return False
    
    print(f"\n🚀 Starting vLLM server with model: {model}")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_memory_utilization)
    ]
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    print(f"Command: {' '.join(cmd)}")
    print("\n📝 Starting server (Ctrl+C to stop)...")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped.")
    
    return True


def stop_vllm():
    """Stop vLLM server."""
    print("\n🛑 Stopping vLLM server...")
    result = subprocess.run(["docker", "rm", "-f", "rag-vllm"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Server stopped successfully.")
    else:
        print("⚠️  No running server found or error stopping.")


def show_status(port: int = DEFAULT_PORT):
    """Show vLLM server status."""
    print("\n📊 vLLM Server Status")
    print("=" * 50)
    
    # Check Docker container
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=rag-vllm", "--format", "{{.Status}}"],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        print(f"Docker container: {result.stdout.strip()}")
    else:
        print("Docker container: Not found")
    
    # Check health
    health = get_vllm_health("localhost", port)
    print(f"Health: {health['status']}")
    
    if health["status"] == "healthy":
        # Get loaded models
        models = get_available_models("localhost", port)
        if models:
            print(f"Loaded models: {', '.join(models)}")
    
    print("=" * 50)


def show_models():
    """Show recommended models."""
    print("\n📚 Recommended Models for RAG")
    print("=" * 60)
    
    for key, info in RECOMMENDED_MODELS.items():
        print(f"\n{key}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Min GPU Memory: {info['min_gpu_memory']}")
        print(f"  Max Context: {info['max_model_len']} tokens")
    
    print("\n" + "=" * 60)
    print("Usage: python scripts/deploy_vllm.py start --model <model_name>")


# ===========================================
# Main
# ===========================================

def main():
    parser = argparse.ArgumentParser(
        description="vLLM Server Deployment Script for RAG Chatbot"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start vLLM server")
    start_parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model to load (default: {DEFAULT_MODEL})"
    )
    start_parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to expose (default: {DEFAULT_PORT})"
    )
    start_parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)"
    )
    start_parser.add_argument(
        "--max-model-len",
        type=int,
        help="Maximum model context length"
    )
    start_parser.add_argument(
        "--quantization", "-q",
        choices=["awq", "gptq", "squeezellm", "fp8"],
        help="Quantization method"
    )
    start_parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker to run vLLM"
    )
    start_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU only (slower)"
    )
    
    # Stop command
    subparsers.add_parser("stop", help="Stop vLLM server")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show server status")
    status_parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    
    # Models command
    subparsers.add_parser("models", help="Show recommended models")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check server health")
    health_parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    
    args = parser.parse_args()
    
    if args.command == "start":
        # Check prerequisites
        has_gpu = check_nvidia_gpu()
        has_docker = check_docker()
        
        if args.cpu:
            print("⚠️  Running in CPU mode (this will be slow)")
        elif not has_gpu:
            print("⚠️  No NVIDIA GPU detected. Consider using --cpu flag.")
        
        if args.docker or not check_vllm_installed():
            if not has_docker:
                print("❌ Docker is required but not found.")
                print("   Install Docker or install vLLM: pip install vllm")
                sys.exit(1)
            
            start_vllm_docker(
                model=args.model,
                port=args.port,
                gpu_memory_utilization=args.gpu_memory,
                max_model_len=args.max_model_len,
                quantization=args.quantization,
                use_cpu=args.cpu
            )
        else:
            start_vllm_native(
                model=args.model,
                port=args.port,
                gpu_memory_utilization=args.gpu_memory,
                max_model_len=args.max_model_len,
                quantization=args.quantization
            )
    
    elif args.command == "stop":
        stop_vllm()
    
    elif args.command == "status":
        show_status(args.port)
    
    elif args.command == "models":
        show_models()
    
    elif args.command == "health":
        health = get_vllm_health("localhost", args.port)
        print(json.dumps(health, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
