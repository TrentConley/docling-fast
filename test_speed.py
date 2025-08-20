#!/usr/bin/env python3
"""
Quick speed test to compare processing with and without markdown export
"""

import asyncio
import httpx
import time
from pathlib import Path
import glob

API_URL = "http://localhost:5001/process"


async def process_pdf(client: httpx.AsyncClient, pdf_path: Path, save_markdown: bool = True) -> dict:
    """Send a single PDF to the API."""
    start_time = time.time()
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path.name, f, 'application/pdf')}
            params = {'save_markdown': save_markdown}
            response = await client.post(API_URL, files=files, params=params)
            response.raise_for_status()
            result = response.json()
            process_time = time.time() - start_time
            result['process_time'] = process_time
            return result
    except Exception as e:
        return {"status": "error", "filename": pdf_path.name, "error": str(e)}


async def main():
    # Get first PDF from uploaded directory
    pdf_files = glob.glob("uploaded/*.pdf")[:1]
    
    if not pdf_files:
        # Try Pdf directory
        pdf_files = glob.glob("Pdf/*.pdf")[:1]
    
    if not pdf_files:
        print("No PDF files found in uploaded/ or Pdf/ directory")
        return
    
    pdf_path = Path(pdf_files[0])
    print(f"Testing with: {pdf_path.name}")
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
        
        # Test 1: With markdown export (default)
        print("Test 1: Processing WITH markdown export...")
        result1 = await process_pdf(client, pdf_path, save_markdown=True)
        print(f"✓ Completed in {result1.get('process_time', 0):.2f}s")
        print(f"  Pages: {result1.get('pages', 0)}")
        print(f"  Output: {result1.get('output_path', 'N/A')}")
        
        # Test 2: Without markdown export
        print("\nTest 2: Processing WITHOUT markdown export...")
        result2 = await process_pdf(client, pdf_path, save_markdown=False)
        print(f"✓ Completed in {result2.get('process_time', 0):.2f}s")
        print(f"  Pages: {result2.get('pages', 0)}")
        print(f"  Output: {result2.get('output_path', 'None')}")
        
        # Compare
        if result1.get('status') == 'success' and result2.get('status') == 'success':
            time1 = result1.get('process_time', 0)
            time2 = result2.get('process_time', 0)
            speedup = ((time1 - time2) / time1) * 100 if time1 > 0 else 0
            
            print("\n" + "=" * 50)
            print("Performance Summary:")
            print(f"With markdown:    {time1:.2f}s")
            print(f"Without markdown: {time2:.2f}s")
            print(f"Speed improvement: {speedup:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
