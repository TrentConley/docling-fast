import os
import multiprocessing as mp
import asyncio
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from docling.document_converter import DocumentConverter

# CPU-only execution: hide GPUs from downstream libraries
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set environment for performance on CPU (use all cores)
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 4)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Settings optimized for CPU-only + high CPU count
cpu_count = os.cpu_count() or 4
MAX_WORKERS = max(1, cpu_count)  # Use as many cores as available
MAX_FILE_SIZE_MB = 500
BATCH_PROCESSING_THRESHOLD = 10

# CPU-only processing
REQUIRE_GPU = False

logger.info(f"System CPU count: {cpu_count}")
logger.info(f"Using {MAX_WORKERS} workers for ProcessPoolExecutor (CPU-only)")
logger.info(f"GPU-only mode: {REQUIRE_GPU}")
logger.info(f"Max file size: {MAX_FILE_SIZE_MB}MB")
logger.info(f"Batch processing threshold: {BATCH_PROCESSING_THRESHOLD}MB")

# Process pool executor
executor = None

# Global converter instance (loaded once per process)
_converter = None


def check_gpu_available():
    """CPU-only mode: always report CPU device."""
    logger.info("CPU-only mode: GPU acceleration disabled")
    return "cpu"


def get_converter():
    """Get or create the DocumentConverter instance (forced CPU)."""
    global _converter
    if _converter is None:
        logger.info("Initializing DocumentConverter (first time per process, CPU-only)...")
        device = check_gpu_available()

        try:
            from docling.datamodel.base_models import ConversionConfig
            # Optimize for speed - matching app.py
            config = ConversionConfig(
                table_structure_model=None,    # Disable table structure processing
                ocr_force_full_page=False,     # Only OCR when needed
                do_ocr=False                   # Disable OCR by default for speed
            )
            _converter = DocumentConverter(config=config)
            
        except Exception as e:
            logger.warning(f"Failed to create configured converter: {e}")
            _converter = DocumentConverter()
        logger.info(f"DocumentConverter initialized with device: {device}")
    return _converter


def process_pdf_sync(temp_path: str, filename: str, save_markdown: bool = True) -> dict:
    """Process an already-saved PDF file in a separate process for true parallelism."""
    try:
        logger.info(f"Starting processing of {filename} from temp path: {temp_path}")

        # Process with docling - reuse converter instance
        converter = get_converter()

        logger.info(f"Converting {filename}...")
        result = converter.convert(temp_path)
        logger.info(f"Conversion complete for {filename}")

        # Optionally save markdown to file
        output_path = None
        markdown_size_kb = 0

        if save_markdown:
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
            markdown_size_kb = len(markdown_content) / 1024

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
            "output_path": str(output_path) if output_path else None,
            "size_kb": markdown_size_kb
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
    finally:
        # Ensure memory is released in the worker process
        try:
            import gc
            gc.collect()
        except Exception:
            pass


def _worker_init():
    # Preload converter once per worker to remove cold-start latency
    try:
        get_converter()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage process pool lifecycle."""
    global executor

    # Startup
    logger.info(f"Starting ProcessPoolExecutor with {MAX_WORKERS} workers (CPU-only)")

    # Always CPU in this app
    device = check_gpu_available()
    if REQUIRE_GPU and device == "cpu":
        logger.error("GPU required but not available!")
        raise RuntimeError("GPU required but not available. Cannot start server.")

    # Use spawn start method for safety in subprocesses
    executor = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_worker_init,
        mp_context=mp.get_context("spawn")
    )
    logger.info("ProcessPoolExecutor started successfully")
    logger.info(f"System can process up to {MAX_WORKERS} PDFs concurrently")
    logger.info("CPU-only processing")

    yield

    # Shutdown
    logger.info("Shutting down ProcessPoolExecutor...")
    executor.shutdown(wait=True)
    logger.info("ProcessPoolExecutor shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Docling PDF Processor (CPU-only)",
    description="Fast parallel PDF processing on CPU",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check."""
    device = check_gpu_available()
    return {
        "service": "Docling PDF Processor (CPU-only)",
        "max_workers": MAX_WORKERS,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "batch_threshold_mb": BATCH_PROCESSING_THRESHOLD,
        "gpu_available": False,
        "device": device,
        "gpu_required": REQUIRE_GPU,
        "processing_model": f"Up to {MAX_WORKERS} concurrent PDFs (CPU-only)",
        "hardware_optimized_for": "Multi-core CPU"
    }


@app.post("/process")
async def process_pdf(file: UploadFile = File(...), save_markdown: bool = False):
    """Process a PDF file with parallel execution.

    Args:
        file: The PDF file to process
        save_markdown: Whether to save markdown output (default: True).
                      Set to False for faster processing without file output.
    """
    try:
        logger.info(f"Received file: {file.filename}")

        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Stream to temp file to avoid large cross-process pickling
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f}MB")

        # Save content to a temp file and pass path to worker to avoid copying large bytes between processes
        import re
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        temp_path = f"/tmp/{safe_filename}"
        with open(temp_path, 'wb') as f:
            f.write(content)
        logger.info(f"Saved upload to temp file: {temp_path}")

        # Process in separate process for true parallelism
        logger.info(f"Submitting {file.filename} to process pool (save_markdown={save_markdown})...")
        loop = asyncio.get_running_loop()
        # Retry loop kept for parity; CPU OOM unlikely
        retry_delay = 1.0
        while True:
            result = await loop.run_in_executor(
                executor,
                process_pdf_sync,
                temp_path,
                file.filename,
                save_markdown
            )

            logger.info(f"Processing attempt complete for {file.filename}: {result['status']}")

            if result["status"] != "error":
                return result

            error_text = str(result.get("error", "")).lower()
            is_oom = ("out of memory" in error_text) or ("oom" in error_text)
            if is_oom:
                logger.warning(f"OOM detected while processing {file.filename}. Retrying in {retry_delay:.1f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

            # Non-OOM errors bubble up as 500
            logger.error(f"Processing failed: {result['error']}")
            if "traceback" in result:
                logger.error(f"Full traceback:\n{result['traceback']}")
            raise HTTPException(status_code=500, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app_cpu:app",
        host="0.0.0.0",
        port=5002,
        workers=1
    )


