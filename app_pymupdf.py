import os
import asyncio
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException

# Limit thread oversubscription for CPU-bound workloads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use all available CPUs for heavy CPU lifting
CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = max(1, CPU_COUNT)
logger.info(f"System CPU count: {CPU_COUNT}")
logger.info(f"Using {MAX_WORKERS} workers for ProcessPoolExecutor (CPU-heavy mode)")

executor = None


def process_pdf_sync(temp_path: str, filename: str, save_markdown: bool = True) -> dict:
    """CPU-heavy PDF->Markdown conversion using PyMuPDF4LLM.

    Runs in a separate process for true CPU parallelism.
    """
    try:
        logger.info(f"[worker] Processing {filename} from temp path: {temp_path}")

        # Import inside worker to keep parent light
        import pymupdf4llm
        import fitz  # PyMuPDF

        # Convert to Markdown (this is CPU intensive for larger PDFs)
        md_text = pymupdf4llm.to_markdown(temp_path)

        # Determine page count (for benchmark compatibility)
        pages = 0
        try:
            doc = fitz.open(temp_path)
            pages = doc.page_count
            doc.close()
        except Exception:
            pages = 0

        output_path = None
        size_kb = 0.0
        if save_markdown:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_filename = f"{Path(filename).stem}.md"
            output_path = output_dir / output_filename
            with open(output_path, 'wb') as f:
                f.write(md_text.encode('utf-8'))
            size_kb = len(md_text) / 1024.0

        # Best-effort cleanup of temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

        return {
            "status": "success",
            "filename": filename,
            "pages": int(pages),
            "output_path": str(output_path) if output_path else None,
            "size_kb": size_kb,
        }
    except Exception as e:
        return {
            "status": "error",
            "filename": filename,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            import gc
            gc.collect()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    logger.info(f"Starting ProcessPoolExecutor with {MAX_WORKERS} workers (CPU-only)")
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    try:
        yield
    finally:
        logger.info("Shutting down ProcessPoolExecutor...")
        executor.shutdown(wait=True)
        logger.info("ProcessPoolExecutor shutdown complete")


app = FastAPI(
    title="PyMuPDF4LLM PDF Processor",
    description="CPU-heavy PDF→Markdown conversion using PyMuPDF4LLM",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "service": "PyMuPDF4LLM PDF Processor",
        "max_workers": MAX_WORKERS,
        "cpu_count": CPU_COUNT,
        "endpoint": "/process",
        "notes": "Upload a PDF via multipart form field 'file'. Returns page count and optional markdown output path.",
    }


@app.post("/process")
async def process_pdf(file: UploadFile = File(...), save_markdown: bool = True):
    try:
        logger.info(f"Received file: {file.filename}")
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        content = await file.read()

        import re
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        temp_path = f"/tmp/{safe_filename}"
        with open(temp_path, 'wb') as f:
            f.write(content)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            process_pdf_sync,
            temp_path,
            file.filename,
            save_markdown,
        )

        if result.get("status") == "success":
            return result
        else:
            if "traceback" in result:
                logger.error(result["traceback"])
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_pymupdf:app",
        host="0.0.0.0",
        port=5002,
        workers=1,
    )


