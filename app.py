# app.py
import os
import re
import uuid
import shutil
import asyncio
import logging
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# -------------------------
# Environment & Logging
# -------------------------

# Prevent thread over-subscription (tune later if CPU-bound stages need more)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# PyTorch CUDA allocator: keep this simple & widely supported
# (expandable segments + coarser split to reduce fragmentation)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("docling_server")

# -------------------------
# Settings
# -------------------------

cpu_count = os.cpu_count() or 4
# Safer default for a single 3090: 1 GPU-heavy worker; increase only if profiled
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "1"))
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "0"))  # 0 = unlimited
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "true").lower() == "true"

# Chunking knobs (dynamic OOM-aware page window for large PDFs)
INITIAL_PAGES_PER_CHUNK = int(os.environ.get("INITIAL_PAGES_PER_CHUNK", "12"))
MIN_PAGES_PER_CHUNK = 1
MAX_CHUNK_RETRIES = 4  # retries per chunk as we shrink chunk size on OOM

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parent-side submission guard (prevents queue explosions)
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "8"))
SUBMIT_SEM = asyncio.Semaphore(MAX_INFLIGHT)

# -------------------------
# Globals set at startup
# -------------------------

executor: Optional[ProcessPoolExecutor] = None
DEVICE_INFO = {
    "device": "cpu",
    "name": None,
    "total_gb": None,
    "compute_capability": None,
    "mps": False,
}


# -------------------------
# Device helpers (cached)
# -------------------------

def _detect_device_once() -> None:
    """Detect GPU/MPS once at startup and store in DEVICE_INFO."""
    try:
        import torch

        if torch.cuda.is_available():
            dev = "cuda"
            props = torch.cuda.get_device_properties(0)
            DEVICE_INFO.update({
                "device": dev,
                "name": torch.cuda.get_device_name(0),
                "total_gb": round(props.total_memory / (1024 ** 3), 1),
                "compute_capability": f"{props.major}.{props.minor}",
                "mps": False,
            })
            logger.info(f"CUDA GPU: {DEVICE_INFO['name']} | {DEVICE_INFO['total_gb']} GB | CC {DEVICE_INFO['compute_capability']}")
            return

        # Apple Metal (not for 3090 boxes, but keep it)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            DEVICE_INFO.update({
                "device": "mps",
                "name": "Apple MPS",
                "total_gb": None,
                "compute_capability": None,
                "mps": True,
            })
            logger.info("Apple MPS detected")
            return
    except Exception as e:
        logger.warning(f"Could not probe torch device: {e}")

    DEVICE_INFO.update({"device": "cpu"})
    logger.info("No GPU detected; device=cpu")


def _gpu_mem_string() -> str:
    try:
        import torch
        if DEVICE_INFO["device"] == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
            rsv = torch.cuda.memory_reserved(0) / (1024 ** 3)
            return f"GPU mem: {alloc:.1f} GB used, {rsv:.1f} GB reserved, {total:.1f} GB total"
        if DEVICE_INFO["device"] == "mps":
            return "MPS device active"
    except Exception:
        pass
    return "No GPU"


# -------------------------
# Worker-side setup
# -------------------------

_converter = None  # one per worker process

def _worker_check_and_init_converter():
    """Initialize the DocumentConverter once per worker with speed-first config."""
    global _converter
    if _converter is not None:
        return _converter

    # Device sanity for workers (optional: rely on parent to block CPU-only startup)
    try:
        from docling.datamodel.base_models import ConversionConfig
        from docling.document_converter import DocumentConverter

        # Speed-oriented config: adjust if your docling version differs
        config = ConversionConfig(
            table_structure_model="fast",
            ocr_force_full_page=False,
            do_ocr=False,
        )
        _converter = DocumentConverter(config=config)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # TF32 and heuristics: profile if beneficial for your models
                torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = False  # conservative default
        except Exception:
            pass

        logger.info("DocumentConverter initialized in worker")
    except Exception as e:
        # Fallback to defaults if config signature changes between versions
        logger.warning(f"Falling back to default DocumentConverter: {e}")
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()

    return _converter


def _is_oom(exc: BaseException) -> bool:
    """Best-effort OOM detector across CUDA/MPS/backends."""
    s = (str(exc) or "").lower()
    if any(k in s for k in ("cuda out of memory", "mps out of memory", "out of memory", "oom")):
        return True
    try:
        import torch
        from torch.cuda import OutOfMemoryError as TorchOOM
        if isinstance(exc, TorchOOM):
            return True
    except Exception:
        pass
    return False


# -------------------------
# Worker: chunked processing
# -------------------------

def _get_pdf_page_count(pdf_path: str) -> int:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    return len(reader.pages)


def _write_pdf_chunk(src_path: str, start: int, end: int, dst_path: str) -> None:
    """Write pages [start, end) from src to dst (0-indexed)."""
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(src_path)
    writer = PdfWriter()
    for i in range(start, end):
        writer.add_page(reader.pages[i])
    with open(dst_path, "wb") as f:
        writer.write(f)


def _convert_pdf_chunk(chunk_path: str) -> str:
    """Run docling on a chunk and return markdown."""
    converter = _worker_check_and_init_converter()
    result = converter.convert(chunk_path)
    return result.document.export_to_markdown()


def process_pdf_sync_chunked(
    original_temp_path: str,
    original_filename: str,
    save_markdown: bool,
) -> dict:
    """
    Convert a potentially large PDF by splitting into page-chunks.
    Dynamically reduces chunk size on OOM to keep progress going.
    """
    import gc
    import time
    start_time = time.time()
    pages_total = 0
    out_path = None
    combined_md_fp = None

    try:
        # Count pages up-front (cheap & CPU-only)
        page_count = _get_pdf_page_count(original_temp_path)
        logger.info(f"[{original_filename}] page_count={page_count} | {_gpu_mem_string()}")

        # Prepare output writer if requested
        if save_markdown:
            out_name = f"{Path(original_filename).stem}_{uuid.uuid4().hex[:8]}.md"
            out_path = OUTPUT_DIR / out_name
            combined_md_fp = open(out_path, "w", encoding="utf-8")
            # Optional: header
            combined_md_fp.write(f"<!-- Generated by docling chunked pipeline; source={original_filename} -->\n\n")

        # Walk pages in adaptive chunks
        pages_per_chunk = max(MIN_PAGES_PER_CHUNK, INITIAL_PAGES_PER_CHUNK)
        i = 0
        while i < page_count:
            # plan this chunk
            j = min(i + pages_per_chunk, page_count)
            chunk_file = Path("/tmp") / f"chunk_{uuid.uuid4().hex}_{i}_{j}.pdf"

            try:
                _write_pdf_chunk(original_temp_path, i, j, str(chunk_file))
                # Attempt conversion
                md = _convert_pdf_chunk(str(chunk_file))

                # Success: write/accumulate
                if save_markdown and combined_md_fp:
                    if i > 0:
                        combined_md_fp.write("\n\n---\n\n")  # simple separator
                    combined_md_fp.write(md)

                pages_converted = j - i
                pages_total += pages_converted
                logger.info(f"[{original_filename}] Converted pages [{i}:{j}) OK; total={pages_total}/{page_count}")

                # Heuristic: if we just succeeded after downsizing, consider ramping back up slightly
                if pages_per_chunk < INITIAL_PAGES_PER_CHUNK:
                    pages_per_chunk = min(INITIAL_PAGES_PER_CHUNK, pages_per_chunk * 2)

                i = j  # move to next window

            except Exception as e:
                # OOM? shrink window and retry
                if _is_oom(e):
                    logger.warning(f"[{original_filename}] OOM on [{i}:{j}); shrinking chunk. {e}")
                    # Shrink chunk; retry up to MAX_CHUNK_RETRIES at min size
                    retries = 0
                    success = False
                    # first shrink once (half), then eventually to 1 page if needed
                    pages_per_chunk = max(MIN_PAGES_PER_CHUNK, max(1, pages_per_chunk // 2))
                    while retries < MAX_CHUNK_RETRIES:
                        try:
                            j2 = min(i + pages_per_chunk, page_count)
                            # rewrite a smaller chunk
                            try:
                                if chunk_file.exists():
                                    chunk_file.unlink(missing_ok=True)
                            except Exception:
                                pass
                            chunk_file = Path("/tmp") / f"chunk_{uuid.uuid4().hex}_{i}_{j2}.pdf"
                            _write_pdf_chunk(original_temp_path, i, j2, str(chunk_file))
                            md = _convert_pdf_chunk(str(chunk_file))
                            # write accumulated
                            if save_markdown and combined_md_fp:
                                if i > 0:
                                    combined_md_fp.write("\n\n---\n\n")
                                combined_md_fp.write(md)
                            pages_converted = j2 - i
                            pages_total += pages_converted
                            logger.info(f"[{original_filename}] Recovered OOM: converted [{i}:{j2}); total={pages_total}/{page_count}")
                            # recovery: keep small for next step; we may ramp up later
                            i = j2
                            success = True
                            break
                        except Exception as e2:
                            if not _is_oom(e2):
                                raise  # different failure; bubble up
                            retries += 1
                            logger.warning(f"[{original_filename}] OOM again (retry {retries}/{MAX_CHUNK_RETRIES}) on size={pages_per_chunk}: {e2}")
                            if pages_per_chunk > MIN_PAGES_PER_CHUNK:
                                pages_per_chunk = max(MIN_PAGES_PER_CHUNK, pages_per_chunk // 2)
                            else:
                                # already at min; keep retrying a few times in case of transient allocator state
                                pass
                            # Try to free memory pressure before next attempt
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            gc.collect()

                    if not success:
                        raise HTTPException(
                            status_code=503,
                            detail=f"GPU ran out of memory at pages [{i}:{j}). Converted {pages_total}/{page_count} pages.",
                        )
                else:
                    # Not OOM: bubble up with context
                    tb = traceback.format_exc()
                    logger.error(f"[{original_filename}] Non-OOM failure on pages [{i}:{j}): {e}\n{tb}")
                    raise

            finally:
                # Cleanup chunk file
                try:
                    if chunk_file.exists():
                        chunk_file.unlink(missing_ok=True)
                except Exception:
                    pass

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"[{original_filename}] COMPLETE: pages={pages_total}/{page_count} in {elapsed}s | {_gpu_mem_string()}")

        return {
            "status": "success",
            "filename": original_filename,
            "pages": pages_total,
            "output_path": str(out_path) if save_markdown else None,
            "elapsed_sec": elapsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[{original_filename}] ERROR: {e}\n{tb}")
        return {
            "status": "error",
            "filename": original_filename,
            "error": str(e),
            "traceback": tb,
        }
    finally:
        # Best-effort memory cleanup in worker
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(__import__("torch"), "mps"):
                # torch.mps.empty_cache() exists in some builds
                try:
                    import torch
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass


def _worker_init():
    # Preload converter once per worker to avoid cold start during first chunk
    try:
        _worker_check_and_init_converter()
    except Exception as e:
        logger.warning(f"Worker init failed: {e}")


# -------------------------
# FastAPI app & schemas
# -------------------------

class ProcessResponse(BaseModel):
    status: str
    filename: str
    pages: Optional[int] = 0
    output_path: Optional[str] = None
    elapsed_sec: Optional[float] = None
    error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor

    _detect_device_once()
    if REQUIRE_GPU and DEVICE_INFO["device"] == "cpu":
        logger.error("GPU required but not available. Set REQUIRE_GPU=false to allow CPU fallback.")
        raise RuntimeError("GPU required but not available")

    logger.info(f"Starting ProcessPoolExecutor: MAX_WORKERS={MAX_WORKERS}, device={DEVICE_INFO['device']}")
    executor = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_worker_init,
        mp_context=mp.get_context("spawn"),
    )

    try:
        yield
    finally:
        logger.info("Shutting down ProcessPoolExecutor…")
        executor.shutdown(wait=True)
        logger.info("Executor shutdown complete.")


app = FastAPI(
    title="Docling PDF Processor",
    description="Fast, chunked PDF→Markdown with OOM-aware GPU usage",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "service": "Docling PDF Processor",
        "device": DEVICE_INFO,
        "max_workers": MAX_WORKERS,
        "max_inflight": MAX_INFLIGHT,
        "output_dir": str(OUTPUT_DIR),
    }


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", stem)
    return safe or "upload"


def _looks_like_pdf_first_bytes(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            hdr = f.read(5)
        return hdr == b"%PDF-"
    except Exception:
        return False


@app.post("/process", response_model=ProcessResponse)
async def process_pdf(file: UploadFile = File(...), save_markdown: bool = True):
    """
    Process a PDF. For large PDFs, we split into page chunks and adapt chunk size
    on GPU OOM to keep progressing. Returns a single merged Markdown path if requested.
    """
    # Validate filename extension (weak), then magic bytes (strong)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported (must end with .pdf)")

    # Stream to unique temp file (no giant read(); avoids cross-process pickling)
    safe_base = _safe_stem(file.filename)
    tmp = NamedTemporaryFile(
        prefix=f"{safe_base}_{uuid.uuid4().hex}_",
        suffix=".pdf",
        dir="/tmp",
        delete=False,
    )
    temp_path = tmp.name
    try:
        # UploadFile.file is a SpooledTemporaryFile; stream to our named file
        await file.seek(0)
        shutil.copyfileobj(file.file, tmp)
    finally:
        tmp.close()

    # Size check (optional)
    try:
        fsz = Path(temp_path).stat().st_size / (1024 * 1024)
    except Exception:
        fsz = None

    if MAX_FILE_SIZE_MB and fsz and fsz > MAX_FILE_SIZE_MB:
        # Parent owns cleanup
        try:
            os.remove(temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=413, detail=f"File too large; limit={MAX_FILE_SIZE_MB} MB")

    # Magic bytes quick check
    if not _looks_like_pdf_first_bytes(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Invalid PDF file (bad header)")

    logger.info(f"Accepted upload: name={file.filename} size_mb={f'{fsz:.2f}' if fsz else 'unknown'} | {_gpu_mem_string()}")

    # Submit bounded
    async with SUBMIT_SEM:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                executor,
                process_pdf_sync_chunked,
                temp_path,
                file.filename,
                save_markdown,
            )
        finally:
            # Parent always cleans original temp file
            try:
                os.remove(temp_path)
            except Exception:
                pass

    # Normalize worker errors
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    return result


if __name__ == "__main__":
    import uvicorn
    # Keep a single uvicorn worker; GPU workers are managed by our ProcessPoolExecutor
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5001,
        workers=1,
    )