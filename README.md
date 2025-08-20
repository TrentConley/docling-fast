# Docling PDF Processing API

A simple FastAPI service that processes PDFs with true parallelism using Docling.

## Features

- **True Parallelism**: Uses ProcessPoolExecutor to bypass Python's GIL
- **Multiple Workers**: Each uvicorn worker has its own process pool
- **Smart OCR**: Only OCRs pages without text layer for speed
- **Saves Markdown**: Automatically saves converted files to `./output/`
- **Simple**: One endpoint, minimal dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Model Setup (If Hugging Face is Blocked)

If you can't access Hugging Face directly, you can use the pre-downloaded models included in this repository:

### Download Repository with Models

First, clone the repository and download the models:

#### On CentOS/RHEL:
```bash
# Install Git LFS (required for downloading large model files)
sudo yum install -y git-lfs
git lfs install

# Clone the repository
git clone https://github.com/TrentConley/docling-fast.git
cd docling-fast

# Download the actual model files (this may take a while - ~2GB)
git lfs pull
```

#### On Ubuntu/Debian:
```bash
# Install Git LFS
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install

# Clone the repository
git clone https://github.com/TrentConley/docling-fast.git
cd docling-fast

# Download the model files
git lfs pull
```

#### On macOS:
```bash
# Install Git LFS if not already installed
brew install git-lfs
git lfs install

# Clone the repository
git clone https://github.com/TrentConley/docling-fast.git
cd docling-fast

# Download the model files
git lfs pull
```

### Copy Models to Hugging Face Cache

```bash
# Create Hugging Face cache directory if it doesn't exist
mkdir -p ~/.cache/huggingface/hub

# Copy the models from this repo to your local cache
cp -r models/models--ds4sd--docling-models ~/.cache/huggingface/hub/
cp -r models/models--ds4sd--docling-layout-old ~/.cache/huggingface/hub/
cp -r models/models--SWHL--RapidOCR ~/.cache/huggingface/hub/
```

### Verify Models are in Place

After copying, verify the models are correctly placed:

```bash
ls -la ~/.cache/huggingface/hub/models--ds4sd--docling-models/
ls -la ~/.cache/huggingface/hub/models--ds4sd--docling-layout-old/
ls -la ~/.cache/huggingface/hub/models--SWHL--RapidOCR/
```

This ensures Docling can find the models locally without downloading from Hugging Face.

## Usage

### Optimal Startup Command

The best way to start the API for maximum performance while avoiding system thrashing:

```bash
# Auto-detect CPU cores and use n-1 workers (reserves 1 CPU for system)
export CPU_COUNT=$(python -c "import os; print(max(1, os.cpu_count() - 1))")
uvicorn app:app --host 0.0.0.0 --port 5001 --workers $CPU_COUNT

# Or as a one-liner:
uvicorn app:app --host 0.0.0.0 --port 5001 --workers $(python -c "import os; print(max(1, os.cpu_count() - 1))")
```

### Manual Worker Configuration

For specific environments or testing:

```bash
# Low resource system (1-2 workers)
uvicorn app:app --host 0.0.0.0 --port 5001 --workers 2

# Medium system (4-8 cores)
uvicorn app:app --host 0.0.0.0 --port 5001 --workers 4

# High-end system (16+ cores)
uvicorn app:app --host 0.0.0.0 --port 5001 --workers 12

# Development mode (single worker with reload)
uvicorn app:app --host 0.0.0.0 --port 5001 --reload
```

### Production Deployment Script

Create a `start_server.sh` script for consistent deployment:

```bash
#!/bin/bash
# start_server.sh - Optimized Docling API startup

# Calculate optimal workers (CPU count - 1 for system)
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKERS=$((CPU_COUNT - 1))
WORKERS=$((WORKERS < 1 ? 1 : WORKERS))  # Ensure at least 1 worker

echo "System CPUs: $CPU_COUNT"
echo "Using $WORKERS workers (reserving 1 CPU for system)"

# Start uvicorn with optimized settings
uvicorn app:app \
  --host 0.0.0.0 \
  --port 5001 \
  --workers $WORKERS \
  --loop asyncio \
  --access-log \
  --log-level info
```

Then run:
```bash
chmod +x start_server.sh
./start_server.sh
```

## API

### Process PDF
```bash
curl -X POST "http://localhost:5001/process" \
  -F "file=@document.pdf"
```

Response:
```json
{
  "status": "success",
  "filename": "document.pdf",
  "pages": 10,
  "output_path": "output/document.md",
  "size_kb": 45.2
}
```

The markdown file is saved to `./output/document.md`

## How it Works

1. **Multiple Uvicorn Workers**: Each worker is a separate OS process
2. **ProcessPoolExecutor**: Each worker has a pool of processes for PDF processing
3. **Total Parallelism**: `workers × max_workers` (e.g., 4 × 3 = 12 concurrent PDFs)

### Parallelism Formula

- **Uvicorn Workers**: `CPU_COUNT - 1` (reserve 1 CPU for system)
- **Process Pool per Worker**: `MAX_WORKERS = max(1, cpu_count - 1)` (set in app.py)
- **Total Concurrent PDFs**: `(CPU_COUNT - 1) × (CPU_COUNT - 1)`

Examples:
- 4-core system: 3 workers × 3 processes = 9 concurrent PDFs
- 8-core system: 7 workers × 7 processes = 49 concurrent PDFs
- 16-core system: 15 workers × 15 processes = 225 concurrent PDFs

The combination of multiple workers and process pools provides true CPU parallelism, bypassing Python's GIL for CPU-intensive PDF processing.

## Performance Tips

### For Mac (MPS)
The current configuration is optimized for Mac:
- OCR only runs on pages without text layers
- Thread oversubscription is prevented
- Use more workers for better parallelism

### For CUDA/GPU
To use GPU acceleration:
1. Install PyTorch with CUDA support
2. Docling will automatically use GPU for deep learning models
3. Consider reducing worker count since GPU handles parallelism

### OCR Performance
- Current setting: Smart OCR (only when needed)
- For faster processing: Disable OCR in `ConversionConfig`
- For better accuracy: Enable `ocr_force_full_page=True`