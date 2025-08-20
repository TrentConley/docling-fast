# Docling Memory Requirements Analysis

## Model Sizes

Docling uses several deep learning models for document processing:

### 1. Layout Detection Models
- **docling-layout-old**: 164MB (.safetensors)
- **docling-models layout**: 164MB (.safetensors)
- **Model Type**: RT-DETR (Real-Time Detection Transformer)

### 2. Table Structure Recognition
- **TableFormer Accurate**: 203MB (.safetensors)
- **TableFormer Fast**: 139MB (.safetensors)

### 3. OCR Models (RapidOCR)
- **Text Detection**: ~50MB
- **Text Recognition**: ~100MB
- **Multiple language models**: ~50MB each

## Memory Consumption Per Process

### Base Memory Requirements

Each process will load:
1. **Python runtime**: ~50-100MB
2. **FastAPI/Docling libraries**: ~200MB
3. **Model loading overhead**: ~100MB
4. **Total base**: ~350-400MB

### Model Memory (When Loaded)

Models are loaded on-demand:
- **Layout model in memory**: ~500MB (3x model size due to PyTorch overhead)
- **TableFormer in memory**: ~600MB (for accurate) or ~420MB (for fast)
- **OCR models in memory**: ~300-400MB
- **Peak per process**: ~1.5-2GB when all models are active

### PDF Processing Memory

Additional memory per PDF being processed:
- **PDF file buffer**: Size of PDF (typically 1-10MB)
- **Image rendering**: ~50-200MB per page (depends on resolution)
- **Intermediate results**: ~10-50MB
- **Peak per PDF**: ~100-300MB

## Total Memory Formula

```
Total RAM = Base_System + (Workers × Worker_Memory) + (Processes × Process_Memory)

Where:
- Base_System = 2GB (OS + other services)
- Worker_Memory = 400MB (FastAPI worker overhead)
- Process_Memory = 2GB (models + processing overhead)
```

### Examples by System Size

#### 4-Core System (3 workers × 3 processes)
- Base: 2GB
- Workers: 3 × 400MB = 1.2GB
- Processes: 9 × 2GB = 18GB
- **Total RAM needed: ~21GB** (24GB recommended)

#### 8-Core System (7 workers × 7 processes)
- Base: 2GB
- Workers: 7 × 400MB = 2.8GB
- Processes: 49 × 2GB = 98GB
- **Total RAM needed: ~100GB** (128GB recommended)

#### 16-Core System (15 workers × 15 processes)
- Base: 2GB
- Workers: 15 × 400MB = 6GB
- Processes: 225 × 2GB = 450GB
- **Total RAM needed: ~450GB** (512GB recommended)

## GPU/CUDA Requirements

### VRAM Requirements Per GPU Process

When using CUDA acceleration:

1. **Model VRAM**:
   - Layout model: ~500MB VRAM
   - TableFormer: ~600MB VRAM
   - OCR models: ~400MB VRAM
   - **Total models**: ~1.5GB VRAM

2. **Processing VRAM**:
   - Image tensors: ~200-500MB per batch
   - Intermediate features: ~500MB
   - **Peak processing**: ~1GB VRAM

3. **Total per process**: ~2.5GB VRAM

### GPU Recommendations

#### For Production Use:

**Low Volume (10-20 concurrent PDFs)**
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- Processes: 4-5 GPU processes
- RAM: 32GB system memory

**Medium Volume (50-100 concurrent PDFs)**
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Processes: 8-10 GPU processes
- RAM: 64GB system memory

**High Volume (200+ concurrent PDFs)**
- GPU: NVIDIA A100 (40GB/80GB VRAM)
- Processes: 15-30 GPU processes
- RAM: 128-256GB system memory

**Multi-GPU Setup**
- 2× RTX 4090: ~20 processes total
- 4× RTX 4090: ~40 processes total
- 8× A100: ~240 processes total

## Memory Optimization Strategies

### 1. Reduce Worker Count
```bash
# For memory-constrained systems
uvicorn app:app --workers 2 --host 0.0.0.0 --port 5001
```

### 2. Limit Process Pool Size
Edit `app.py`:
```python
MAX_WORKERS = min(2, cpu_count - 1)  # Limit to 2 processes per worker
```

### 3. Use Fast Models
Configure Docling to use faster (smaller) models:
```python
config = ConversionConfig(
    table_structure_model="fast"  # Uses 139MB model instead of 203MB
)
```

### 4. Enable Model Sharing (Advanced)
Use shared memory for models across processes (requires custom implementation).

## Monitoring Commands

### Check Memory Usage
```bash
# Real-time memory monitoring
htop

# Process-specific memory
ps aux | grep python | awk '{sum+=$6} END {print "Total RSS: " sum/1024 " MB"}'

# GPU memory (if using CUDA)
nvidia-smi
```

### Memory Profiling Script
```bash
#!/bin/bash
# monitor_memory.sh

while true; do
    echo "=== $(date) ==="
    echo "System Memory:"
    free -h
    echo ""
    echo "Python Processes:"
    ps aux | grep python | grep -E "(uvicorn|app.py)" | awk '{printf "PID: %s RSS: %.1f MB CMD: %s\n", $2, $6/1024, $11}'
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory:"
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
    fi
    echo ""
    sleep 5
done
```

## Recommendations Summary

### CPU-Only Deployment
- **Minimum**: 16GB RAM for 2-4 processes
- **Recommended**: 32GB RAM for 8-12 processes
- **Production**: 64-128GB RAM for 20-50 processes

### GPU-Accelerated Deployment
- **Minimum**: RTX 3060 (12GB) + 32GB RAM
- **Recommended**: RTX 4090 (24GB) + 64GB RAM
- **Production**: Multiple A100s + 256GB+ RAM

### Scaling Formula
```
Concurrent PDFs = min(
    (Available_RAM - 4GB) / 2GB,
    (Available_VRAM / 2.5GB) if using GPU,
    CPU_Cores × 3
)
```

Choose the minimum of these three constraints for optimal performance without thrashing.
