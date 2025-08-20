import os
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from docling.document_converter import DocumentConverter

# Settings
MAX_WORKERS = max(1, os.cpu_count() - 1)  # Leave one CPU for the system
MAX_FILE_SIZE_MB = 50

# Process pool executor
executor = None


def process_pdf_sync(pdf_content: bytes, filename: str) -> dict:
    """Process PDF in a separate process for true parallelism."""
    try:
        # Save to temporary file (docling needs file path)
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'wb') as f:
            f.write(pdf_content)
        
        # Process with docling
        converter = DocumentConverter()
        result = converter.convert(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        # Return results
        return {
            "status": "success",
            "filename": filename,
            "pages": len(result.document.pages) if hasattr(result.document, 'pages') else 0,
            "content": result.document.export_to_markdown()
        }
    except Exception as e:
        return {
            "status": "error",
            "filename": filename,
            "error": str(e)
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage process pool lifecycle."""
    global executor
    # Startup
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    yield
    # Shutdown
    executor.shutdown(wait=True)


# Create FastAPI app
app = FastAPI(
    title="Docling PDF Processor",
    description="Fast parallel PDF processing with true concurrency",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check."""
    return {
        "service": "Docling PDF Processor",
        "max_workers": MAX_WORKERS,
        "max_file_size_mb": MAX_FILE_SIZE_MB
    }


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    """Process a PDF file with parallel execution."""
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read and check file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
        )
    
    # Process in separate process for true parallelism
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor,
        process_pdf_sync,
        content,
        file.filename
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result


if __name__ == "__main__":
    import uvicorn
    
    # Run with multiple workers for additional parallelism
    # Total parallelism = workers × max_workers
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5001,
        workers=os.cpu_count() or 4
    )