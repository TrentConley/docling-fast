#!/usr/bin/env python3
"""
Analyze PDF files to get page count statistics

This script scans all PDFs in the Pdf directory and provides:
- Page count for each PDF
- Total pages across all PDFs
- Average pages per PDF
- Distribution statistics
"""

import os
from pathlib import Path
from typing import List, Tuple
import PyPDF2
from tqdm import tqdm
import statistics
import csv

def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return len(pdf_reader.pages)
    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")
        return -1

def analyze_pdf_directory(pdf_dir: Path = Path("Pdf")) -> List[Tuple[str, int]]:
    """Analyze all PDFs in the directory and return page counts."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found!")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to analyze...")
    
    results = []
    for pdf_path in tqdm(pdf_files, desc="Analyzing PDFs"):
        page_count = get_pdf_page_count(pdf_path)
        if page_count >= 0:
            results.append((pdf_path.name, page_count))
    
    return results

def print_statistics(results: List[Tuple[str, int]]):
    """Print comprehensive statistics about the PDFs."""
    if not results:
        print("No valid PDFs to analyze!")
        return
    
    page_counts = [count for _, count in results]
    
    print("\n" + "="*80)
    print("PDF ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total PDFs analyzed: {len(results)}")
    print(f"   Total pages: {sum(page_counts):,}")
    print(f"   Average pages per PDF: {statistics.mean(page_counts):.2f}")
    print(f"   Median pages per PDF: {statistics.median(page_counts):.0f}")
    
    if len(page_counts) > 1:
        print(f"   Standard deviation: {statistics.stdev(page_counts):.2f}")
    
    print(f"\n📈 DISTRIBUTION:")
    print(f"   Minimum pages: {min(page_counts)}")
    print(f"   Maximum pages: {max(page_counts)}")
    print(f"   25th percentile: {statistics.quantiles(page_counts, n=4)[0]:.0f}")
    print(f"   75th percentile: {statistics.quantiles(page_counts, n=4)[2]:.0f}")
    
    # Page range distribution
    ranges = {
        "1-10 pages": 0,
        "11-50 pages": 0,
        "51-100 pages": 0,
        "101-500 pages": 0,
        "500+ pages": 0
    }
    
    for count in page_counts:
        if count <= 10:
            ranges["1-10 pages"] += 1
        elif count <= 50:
            ranges["11-50 pages"] += 1
        elif count <= 100:
            ranges["51-100 pages"] += 1
        elif count <= 500:
            ranges["101-500 pages"] += 1
        else:
            ranges["500+ pages"] += 1
    
    print(f"\n📚 PAGE RANGE DISTRIBUTION:")
    for range_name, count in ranges.items():
        percentage = (count / len(results)) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {range_name:15} {count:4d} ({percentage:5.1f}%) {bar}")
    
    # Find extremes
    results_sorted = sorted(results, key=lambda x: x[1])
    
    print(f"\n📄 SMALLEST PDFS:")
    for name, pages in results_sorted[:5]:
        print(f"   {name}: {pages} pages")
    
    print(f"\n📚 LARGEST PDFS:")
    for name, pages in results_sorted[-5:]:
        print(f"   {name}: {pages} pages")
    
    print("\n" + "="*80)

def save_to_csv(results: List[Tuple[str, int]], filename: str = "pdf_analysis.csv"):
    """Save the results to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Page Count'])
        writer.writerows(results)
    print(f"\nResults saved to {filename}")

def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze PDF page counts")
    parser.add_argument('--dir', default='Pdf', help='Directory containing PDFs')
    parser.add_argument('--csv', action='store_true', help='Save results to CSV')
    parser.add_argument('--verbose', action='store_true', help='Show all PDFs')
    
    args = parser.parse_args()
    
    pdf_dir = Path(args.dir)
    if not pdf_dir.exists():
        print(f"Directory {pdf_dir} not found!")
        return
    
    # Analyze PDFs
    results = analyze_pdf_directory(pdf_dir)
    
    # Print statistics
    print_statistics(results)
    
    # Save to CSV if requested
    if args.csv:
        save_to_csv(results)
    
    # Show all PDFs if verbose
    if args.verbose:
        print("\n📋 ALL PDFS:")
        for name, pages in sorted(results, key=lambda x: x[1]):
            print(f"   {name}: {pages} pages")

if __name__ == "__main__":
    main()
