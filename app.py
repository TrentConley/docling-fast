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

# Set environment for performance
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent thread oversubscription
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# GPU memory management optimized for RTX 3090 (24GB VRAM)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:1024,roundup_power2_divisions:16"  # Optimize for large VRAM
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable async CUDA operations
os.environ["CUDA_CACHE_MAXSIZE"] = "268435456"  # 256MB CUDA cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Settings optimized for RTX 3090 (24GB VRAM) + high CPU count
cpu_count = os.cpu_count() or 4  # Default to 4 if can't detect
# Optimize for RTX 3090: Each worker needs ~2-3GB VRAM, 24GB allows ~6-8 workers
MAX_WORKERS = min(8, max(1, cpu_count // 2))  # Scale with CPU but cap at 8 for RTX 3090
MAX_FILE_SIZE_MB = 500  # Increased for high-end hardware capabilities
# Batch size for processing multiple smaller files efficiently
BATCH_PROCESSING_THRESHOLD = 10  # Files under 10MB can be batched

# GPU-only processing settings
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "true").lower() == "true"

logger.info(f"System CPU count: {cpu_count}")
logger.info(f"Using {MAX_WORKERS} workers for ProcessPoolExecutor (optimized for RTX 3090)")
logger.info(f"GPU-only mode: {REQUIRE_GPU}")
logger.info(f"Max file size: {MAX_FILE_SIZE_MB}MB")
logger.info(f"Batch processing threshold: {BATCH_PROCESSING_THRESHOLD}MB")

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
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            logger.info(f"CUDA GPU detected: {device_name}")
            logger.info(f"GPU Memory: {allocated_memory:.1f}GB used, {reserved_memory:.1f}GB reserved, {total_memory:.1f}GB total")
            logger.info(f"GPU Compute Capability: {props.major}.{props.minor}")
            logger.info(f"GPU Multiprocessors: {props.multi_processor_count}")
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
                do_ocr=False                   # Disable OCR by default for speed
            )
            _converter = DocumentConverter(config=config)
            
            # Optimize GPU utilization for RTX 3090
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()  # Clear any cached memory
                # Enable optimizations for RTX 3090
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster computation
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
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
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
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
    logger.info(f"Starting ProcessPoolExecutor with {MAX_WORKERS} workers (RTX 3090 optimized)")
    
    # Check GPU availability at startup
    device = check_gpu_available()
    if REQUIRE_GPU and device == "cpu":
        logger.error("GPU required but not available!")
        raise RuntimeError("GPU required but not available. Cannot start server.")
    
    # Use spawn start method to be CUDA-safe in subprocesses
    executor = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_worker_init,
        mp_context=mp.get_context("spawn")
    )
    logger.info("ProcessPoolExecutor started successfully")
    logger.info(f"System can process up to {MAX_WORKERS} PDFs concurrently")
    logger.info("Optimized for RTX 3090 with 24GB VRAM")
    
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
        "batch_threshold_mb": BATCH_PROCESSING_THRESHOLD,
        "gpu_available": device != "cpu",
        "device": device,
        "gpu_required": REQUIRE_GPU,
        "processing_model": f"Up to {MAX_WORKERS} concurrent PDFs (RTX 3090 optimized)",
        "hardware_optimized_for": "RTX 3090 24GB + high CPU count"
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
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"File too large: {file_size_mb:.2f}MB > {MAX_FILE_SIZE_MB}MB")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
            )
        
        # Enhanced GPU memory monitoring for RTX 3090
        if check_gpu_available() == "cuda":
            import torch
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory = total_memory - reserved_memory
            logger.info(f"GPU memory before processing: {allocated_memory:.1f}GB used, {reserved_memory:.1f}GB reserved, {free_memory:.1f}GB free")
            
            # Warn if memory usage is high
            if reserved_memory / total_memory > 0.8:
                logger.warning(f"High GPU memory usage detected: {reserved_memory/total_memory*100:.1f}%")
        
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
        result = await loop.run_in_executor(
            executor,
            process_pdf_sync,
            temp_path,
            file.filename,
            save_markdown
        )
        
        # Do not clear GPU cache in parent process; the worker handles its own cache
        
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
        workers=1
    )