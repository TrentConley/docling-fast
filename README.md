# Docling PDF Processing API

A simple FastAPI service that processes PDFs with true parallelism using Docling.

## Features

- **True Parallelism**: Uses ProcessPoolExecutor to bypass Python's GIL
- **Multiple Workers**: Each uvicorn worker has its own process pool
- **Simple**: One endpoint, minimal dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run with multiple workers for maximum parallelism:

```bash
# Run with uvicorn (number of workers = CPU cores)
uvicorn app:app --host 0.0.0.0 --port 5001 --workers 4
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
  "content": "# Document content in markdown..."
}
```

## How it Works

1. **Multiple Uvicorn Workers**: Each worker is a separate OS process
2. **ProcessPoolExecutor**: Each worker has a pool of processes for PDF processing
3. **Total Parallelism**: `workers × max_workers` (e.g., 4 × 3 = 12 concurrent PDFs)

The combination of multiple workers and process pools provides true CPU parallelism, bypassing Python's GIL for CPU-intensive PDF processing.