import os
import asyncio
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from docling.document_converter import DocumentConverter

# Set environment for performance
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent thread oversubscription
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Settings
cpu_count = os.cpu_count() or 4  # Default to 4 if can't detect
# Reduce workers to prevent exponential process creation
# With uvicorn workers, total processes = uvicorn_workers × MAX_WORKERS
MAX_WORKERS = max(1, min(2, cpu_count // 4))  # Conservative: 1-2 processes per worker
MAX_FILE_SIZE_MB = 50

logger.info(f"System CPU count: {cpu_count}")
logger.info(f"Using {MAX_WORKERS} workers for ProcessPoolExecutor")

# Process pool executor
executor = None

# Global converter instance (loaded once per process)
_converter = None


def get_converter():
    """Get or create the DocumentConverter instance."""
    global _converter
    if _converter is None:
        logger.info("Initializing DocumentConverter (first time per process)...")
        try:
            from docling.datamodel.base_models import ConversionConfig
            config = ConversionConfig()
            _converter = DocumentConverter(config=config)
        except Exception as e:
            logger.warning(f"Failed to create configured converter: {e}")
            _converter = DocumentConverter()
        logger.info("DocumentConverter initialized")
    return _converter


def process_pdf_sync(pdf_content: bytes, filename: str) -> dict:
    """Process PDF in a separate process for true parallelism."""
    try:
        logger.info(f"Starting processing of {filename} ({len(pdf_content)} bytes)")
        
        # Sanitize filename for safe file system usage
        import re
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Save to temporary file (docling needs file path)
        temp_path = f"/tmp/{safe_filename}"
        with open(temp_path, 'wb') as f:
            f.write(pdf_content)
        logger.info(f"Saved temp file: {temp_path}")
        
        # Process with docling - reuse converter instance
        converter = get_converter()
        
        logger.info(f"Converting {filename}...")
        result = converter.convert(temp_path)
        logger.info(f"Conversion complete for {filename}")
        
        # Save markdown to file
        logger.info("Exporting to markdown...")
        markdown_content = result.document.export_to_markdown()
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save with same name but .md extension
        output_filename = f"{Path(filename).stem}.md"
        output_path = output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"Saved markdown to: {output_path}")
        
        # Clean up
        try:
            os.remove(temp_path)
            logger.info(f"Cleaned up temp file: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {temp_path}: {e}")
        
        # Return minimal results (not the full content)
        return {
            "status": "success",
            "filename": filename,
            "pages": len(result.document.pages) if hasattr(result.document, 'pages') else 0,
            "output_path": str(output_path),
            "size_kb": len(markdown_content) / 1024
        }
    except Exception as e:
        error_msg = f"Error processing {filename}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "filename": filename,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage process pool lifecycle."""
    global executor
    # Startup
    logger.info(f"Starting ProcessPoolExecutor with {MAX_WORKERS} workers")
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    logger.info("ProcessPoolExecutor started successfully")
    yield
    # Shutdown
    logger.info("Shutting down ProcessPoolExecutor...")
    executor.shutdown(wait=True)
    logger.info("ProcessPoolExecutor shutdown complete")


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
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read and check file size
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f}MB")
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"File too large: {file_size_mb:.2f}MB > {MAX_FILE_SIZE_MB}MB")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
            )
        
        # Process in separate process for true parallelism
        logger.info(f"Submitting {file.filename} to process pool...")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            process_pdf_sync,
            content,
            file.filename
        )
        
        logger.info(f"Processing complete for {file.filename}: {result['status']}")
        
        if result["status"] == "error":
            logger.error(f"Processing failed: {result['error']}")
            if "traceback" in result:
                logger.error(f"Full traceback:\n{result['traceback']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run with multiple workers for additional parallelism
    # Total parallelism = workers × max_workers
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5001,
        workers=cpu_count
    )