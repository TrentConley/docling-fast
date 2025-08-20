#!/bin/bash
# start_server.sh - Optimized Docling API startup

# Optimized for RTX 3090: Allow multiple workers for better throughput
WORKERS=2  # Start with 2 uvicorn workers, each can handle multiple concurrent PDFs

echo "========================================="
echo "Docling API Server Startup (RTX 3090 Optimized)"
echo "========================================="
echo "Hardware detected:"
echo "  CPUs: $(nproc)"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
    echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
fi
echo "========================================="

# Optimized GPU detection and worker calculation for RTX 3090
if command -v nvidia-smi &> /dev/null; then
    VRAM_TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
    VRAM_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [ ! -z "$VRAM_TOTAL_MB" ] && [ ! -z "$VRAM_FREE_MB" ]; then
        echo "GPU detected: Total VRAM ${VRAM_TOTAL_MB}MB, Free ${VRAM_FREE_MB}MB"
        
        # RTX 3090 with 24GB can handle more workers
        # Conservative estimate: 2-3GB per worker, leave 4GB buffer for large files
        VRAM_WORKERS=$(( (VRAM_FREE_MB - 4096) / 2048 ))
        
        # Cap workers based on VRAM, but allow up to 4 for RTX 3090
        MAX_VRAM_WORKERS=4
        if [ $VRAM_WORKERS -gt $MAX_VRAM_WORKERS ]; then
            VRAM_WORKERS=$MAX_VRAM_WORKERS
        fi
        
        if [ $VRAM_WORKERS -gt 0 ]; then
            if [ $VRAM_WORKERS -lt $WORKERS ]; then
                echo "Limiting to $VRAM_WORKERS workers based on GPU memory"
                WORKERS=$VRAM_WORKERS
            else
                echo "GPU memory sufficient for $WORKERS workers (could support up to $VRAM_WORKERS)"
            fi
        else
            echo "Insufficient GPU memory, falling back to 1 worker"
            WORKERS=1
        fi
    fi
else
    echo "nvidia-smi not found, assuming CPU-only mode"
    WORKERS=1
fi

echo "Configuration:"
echo "  Uvicorn workers: $WORKERS"
echo "  ProcessPool workers per uvicorn worker: 8 (auto-scaling)"
echo "  Max concurrent PDFs: $(( WORKERS * 8 ))"
echo "========================================="

# Set GPU-only environment variables
export REQUIRE_GPU=true
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only

echo "GPU-only mode enabled (REQUIRE_GPU=true)"
echo "RTX 3090 optimizations active:"
echo "  - Enhanced CUDA memory management"
echo "  - TF32 acceleration enabled"
echo "  - Up to 500MB file size support"
echo "  - Batch processing for small files"
echo "========================================="

# Start uvicorn with RTX 3090 optimized settings
echo "Starting Docling API server..."
echo "Access the API at: http://localhost:5001"
echo "Health check: http://localhost:5001/"
echo "========================================="

uvicorn app:app \
  --host 0.0.0.0 \
  --port 5001 \
  --workers $WORKERS \
  --loop asyncio \
  --access-log \
  --log-level info \
  --worker-class uvicorn.workers.UvicornWorker \
  --backlog 2048 \
  --limit-max-requests 1000 \
  --timeout-keep-alive 30
