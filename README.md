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

## Usage

Run with multiple workers for maximum parallelism:

```bash
# Run with uvicorn (number of workers = CPU cores)
venv/bin/uvicorn app:app --host 0.0.0.0 --port 5001 --workers 4
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