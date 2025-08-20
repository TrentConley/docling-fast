#!/bin/bash
# start_server.sh - Optimized Docling API startup

# Calculate optimal workers (CPU count - 1 for system)
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKERS=$((CPU_COUNT - 1))
WORKERS=$((WORKERS < 1 ? 1 : WORKERS))  # Ensure at least 1 worker

echo "========================================="
echo "Docling API Server Startup"
echo "========================================="

# Check for GPU and adjust workers if present
if command -v nvidia-smi &> /dev/null; then
    VRAM_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [ ! -z "$VRAM_FREE_MB" ]; then
        # Conservative: 3GB per process, with 1GB reserved
        VRAM_WORKERS=$(( (VRAM_FREE_MB - 1024) / 3072 ))
        if [ $VRAM_WORKERS -gt 0 ] && [ $VRAM_WORKERS -lt $WORKERS ]; then
            echo "GPU detected with ${VRAM_FREE_MB}MB free"
            WORKERS=$VRAM_WORKERS
            echo "Limiting to $WORKERS workers based on GPU memory"
        fi
    fi
fi

echo "System CPUs: $CPU_COUNT"
echo "Using $WORKERS workers"
echo "========================================="

# Start uvicorn with optimized settings
uvicorn app:app \
  --host 0.0.0.0 \
  --port 5001 \
  --workers $WORKERS \
  --loop asyncio \
  --access-log \
  --log-level info
