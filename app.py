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

# GPU memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"  # Improve fragmentation and allow segment release

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Settings
cpu_count = os.cpu_count() or 4  # Default to 4 if can't detect
# One PDF processor per uvicorn worker for simplicity
MAX_WORKERS = 1  # Each uvicorn worker gets exactly 1 PDF processor
MAX_FILE_SIZE_MB = 100  # Increased from 50MB to handle larger PDFs

# GPU-only processing settings
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "true").lower() == "true"

logger.info(f"System CPU count: {cpu_count}")
logger.info(f"Using {MAX_WORKERS} worker for ProcessPoolExecutor (1 PDF processor per uvicorn worker)")
logger.info(f"GPU-only mode: {REQUIRE_GPU}")

# Process pool executor
executor = None

# Global converter instance (loaded once per process)
_converter = None


def check_gpu_available():
    """Check if GPU is available for acceleration."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            logger.info(f"CUDA GPU detected: {device_name}")
            logger.info(f"GPU Memory: {allocated_memory:.1f}/{total_memory:.1f} GB used")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Apple Metal GPU detected")
            return "mps"
    except ImportError:
        pass
    logger.warning("No GPU detected")
    return "cpu"


def get_converter():
    """Get or create the DocumentConverter instance."""
    global _converter
    if _converter is None:
        logger.info("Initializing DocumentConverter (first time per process)...")
        device = check_gpu_available()
        
        # Enforce GPU requirement if enabled
        if REQUIRE_GPU and device == "cpu":
            raise RuntimeError("GPU required but not available. Set REQUIRE_GPU=false to allow CPU processing.")
        
        try:
            from docling.datamodel.base_models import ConversionConfig
            # Optimize for speed
            config = ConversionConfig(
                table_structure_model="fast",  # Use fast model (139MB vs 203MB)
                ocr_force_full_page=False,     # Only OCR when needed
                do_ocr=True                    # Enable smart OCR
            )
            _converter = DocumentConverter(config=config)
            
            # Force models to GPU if available
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()  # Clear any cached memory
                
        except Exception as e:
            logger.warning(f"Failed to create configured converter: {e}")
            _converter = DocumentConverter()
        logger.info(f"DocumentConverter initialized with device: {device}")
    return _converter


def process_pdf_sync(pdf_content: bytes, filename: str, save_markdown: bool = True) -> dict:
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
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage process pool lifecycle."""
    global executor
    
    # Startup
    logger.info(f"Starting ProcessPoolExecutor with {MAX_WORKERS} worker")
    
    # Check GPU availability at startup
    device = check_gpu_available()
    if REQUIRE_GPU and device == "cpu":
        logger.error("GPU required but not available!")
        raise RuntimeError("GPU required but not available. Cannot start server.")
    
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    logger.info("ProcessPoolExecutor started successfully")
    logger.info("Each uvicorn worker processes exactly 1 PDF at a time")
    
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
    device = check_gpu_available()
    return {
        "service": "Docling PDF Processor",
        "max_workers": MAX_WORKERS,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "gpu_available": device != "cpu",
        "device": device,
        "gpu_required": REQUIRE_GPU,
        "processing_model": "1 PDF per uvicorn worker"
    }


@app.post("/process")
async def process_pdf(file: UploadFile = File(...), save_markdown: bool = True):
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
        
        # Check GPU memory before processing (optional monitoring)
        if check_gpu_available() == "cuda":
            import torch
            free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            logger.info(f"GPU free memory before processing: {free_memory:.1f} GB")
        
        # Process in separate process for true parallelism
        logger.info(f"Submitting {file.filename} to process pool (save_markdown={save_markdown})...")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            process_pdf_sync,
            content,
            file.filename,
            save_markdown
        )
        
        # Clear GPU cache after processing
        if check_gpu_available() == "cuda":
            import torch
            torch.cuda.empty_cache()
        
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