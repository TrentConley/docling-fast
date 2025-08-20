#!/bin/bash
# start_server.sh - Optimized Docling API startup

# Force a single uvicorn worker (we batch inside the process)
WORKERS=1

echo "========================================="
echo "Docling API Server Startup"
echo "========================================="

# Check for GPU and adjust workers if present
if command -v nvidia-smi &> /dev/null; then
    VRAM_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [ ! -z "$VRAM_FREE_MB" ]; then
        # Each worker needs ~4GB for one PDF, with 2GB reserved for safety
        VRAM_WORKERS=$(( (VRAM_FREE_MB - 2048) / 4096 ))
        if [ $VRAM_WORKERS -gt 0 ] && [ $VRAM_WORKERS -lt $WORKERS ]; then
            echo "GPU detected with ${VRAM_FREE_MB}MB free"
            WORKERS=$VRAM_WORKERS
            echo "Limiting to $WORKERS workers based on GPU memory"
        fi
    fi
fi

echo "Using $WORKERS worker (single-worker mode)"
echo "========================================="

# Set GPU-only environment variables
export REQUIRE_GPU=true
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only

echo "GPU-only mode enabled (REQUIRE_GPU=true)"
echo "Each uvicorn worker processes 1 PDF at a time"
echo "========================================="

# Start uvicorn with optimized settings
uvicorn app:app \
  --host 0.0.0.0 \
  --port 5001 \
  --workers $WORKERS \
  --loop asyncio \
  --access-log \
  --log-level info
