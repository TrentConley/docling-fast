#!/usr/bin/env python3
"""
Test script to send 10 PDFs concurrently to the Docling API
"""

import asyncio
import httpx
import time
from pathlib import Path
import glob

API_URL = "http://localhost:5001/process"


async def process_pdf(client: httpx.AsyncClient, pdf_path: Path) -> dict:
    """Send a single PDF to the API."""
    print(f"Processing: {pdf_path.name}")
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path.name, f, 'application/pdf')}
            response = await client.post(API_URL, files=files)
            response.raise_for_status()
            result = response.json()
            print(f"✓ Completed: {pdf_path.name} - {result['pages']} pages - Saved to: {result['output_path']}")
            return result
    except Exception as e:
        print(f"✗ Failed: {pdf_path.name} - {str(e)}")
        return {"status": "error", "filename": pdf_path.name, "error": str(e)}


async def main():
    # Get first 10 PDFs from Pdf directory
    pdf_files = glob.glob("Pdf/*.pdf")[:10]
    
    if not pdf_files:
        print("No PDF files found in Pdf directory")
        return
    
    print(f"Found {len(pdf_files)} PDFs to process")
    print("=" * 50)
    
    # Create HTTP client with timeout
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Check if server is running
        try:
            response = await client.get("http://localhost:5001/")
            print(f"Server status: {response.json()}")
            print("=" * 50)
        except:
            print("Error: Server is not running at http://localhost:5001")
            return
        
        # Process all PDFs concurrently
        start_time = time.time()
        
        tasks = [process_pdf(client, Path(pdf)) for pdf in pdf_files]
        results = await asyncio.gather(*tasks)
        
        # Summary
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get("status") != "error")
        total_pages = sum(r.get("pages", 0) for r in results if r.get("status") != "error")
        
        print("=" * 50)
        print(f"Processed {len(pdf_files)} PDFs in {total_time:.2f} seconds")
        print(f"Successful: {successful}/{len(pdf_files)}")
        print(f"Total pages: {total_pages}")
        print(f"Average time per PDF: {total_time/len(pdf_files):.2f}s")
        print(f"Throughput: {len(pdf_files)/total_time:.2f} PDFs/second")
        print(f"\nMarkdown files saved in: ./output/")


if __name__ == "__main__":
    asyncio.run(main())
