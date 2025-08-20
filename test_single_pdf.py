#!/usr/bin/env python3
"""
Quick test script to verify API and benchmark are working correctly
"""

import asyncio
import aiohttp
from pathlib import Path

async def test_single_pdf():
    """Test a single PDF upload to verify the API works."""
    # Find first PDF file
    pdf_files = list(Path("Pdf").glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found!")
        return
    
    pdf_path = pdf_files[0]
    print(f"Testing with: {pdf_path.name}")
    
    # Read file
    with open(pdf_path, 'rb') as f:
        file_content = f.read()
    
    print(f"File size: {len(file_content) / 1024:.2f} KB")
    
    # Create session and send request
    async with aiohttp.ClientSession() as session:
        # Create FormData
        data = aiohttp.FormData()
        data.add_field('file',
                      file_content,
                      filename=pdf_path.name,
                      content_type='application/pdf')
        
        # Send request
        url = "http://localhost:5001/process"
        try:
            async with session.post(url, data=data) as response:
                print(f"Response status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    print(f"Success! Result: {result}")
                else:
                    error = await response.text()
                    print(f"Error: {error}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    print("Testing single PDF upload...")
    asyncio.run(test_single_pdf())
