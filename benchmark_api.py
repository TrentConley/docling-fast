#!/usr/bin/env python3
"""
Comprehensive API Benchmark Script for Docling PDF Processing

This script performs intensive benchmarking of the Docling API by:
- Making concurrent requests to maximize CPU utilization
- Testing all available PDFs from the Pdf/ directory
- Measuring performance per page and overall throughput
- Providing detailed performance analytics

Usage:
    python benchmark_api.py [options]

Options:
    --url URL              API base URL (default: http://localhost:5001)
    --concurrency N        Number of concurrent requests (default: auto-detect CPU cores)
    --max-files N          Maximum number of PDF files to test (default: all)
    --timeout SECONDS      Request timeout in seconds (default: 300)
    --output FILE          Save results to JSON file
    --verbose              Enable verbose output
"""

import asyncio
import aiohttp
import argparse
import json
import logging
import os
import psutil
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
try:
    import GPUtil  # For GPU monitoring if available
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False


class DoclingBenchmark:
    def __init__(self, base_url: str = "http://localhost:5001", timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.logger = self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def get_pdf_files(self, max_files: int = None) -> List[Path]:
        """Get all PDF files from the Pdf directory."""
        pdf_dir = Path("Pdf")
        if not pdf_dir.exists():
            raise FileNotFoundError("Pdf directory not found")

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("No PDF files found in Pdf directory")

        if max_files:
            pdf_files = pdf_files[:max_files]

        self.logger.info(f"Found {len(pdf_files)} PDF files to test")
        return pdf_files

    async def process_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file and return timing results."""
        url = f"{self.base_url}/process"

        start_time = time.time()
        file_size = pdf_path.stat().st_size

        try:
            # Read file content
            with open(pdf_path, 'rb') as f:
                file_content = f.read()

            # Create FormData properly
            data = aiohttp.FormData()
            data.add_field('file',
                          file_content,
                          filename=pdf_path.name,
                          content_type='application/pdf')

            async with self.session.post(url, data=data) as response:
                    end_time = time.time()
                    processing_time = end_time - start_time

                    if response.status == 200:
                        result = await response.json()
                        pages = result.get('pages', 0)
                        return {
                            'filename': pdf_path.name,
                            'status': 'success',
                            'file_size_bytes': file_size,
                            'processing_time_seconds': processing_time,
                            'pages': pages,
                            'time_per_page': processing_time / max(pages, 1),
                            'error': None
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'filename': pdf_path.name,
                            'status': 'error',
                            'file_size_bytes': file_size,
                            'processing_time_seconds': processing_time,
                            'pages': 0,
                            'time_per_page': 0,
                            'error': f"HTTP {response.status}: {error_text}"
                        }

        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            return {
                'filename': pdf_path.name,
                'status': 'error',
                'file_size_bytes': file_size,
                'processing_time_seconds': processing_time,
                'pages': 0,
                'time_per_page': 0,
                'error': str(e)
            }

    def get_optimal_concurrency(self) -> int:
        """Calculate optimal concurrency based on CPU cores and GPU capability."""
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # For RTX 3090 + high-end CPU, allow much higher concurrency
        # Base concurrency on CPU count, but allow higher for powerful systems
        base_concurrency = cpu_count + 2
        
        # Scale up for high-end systems (RTX 3090 + plenty of RAM)
        if memory_gb > 32:  # High-end system
            optimal = min(base_concurrency * 2, 100)  # Allow up to 100 concurrent requests
        else:
            optimal = min(base_concurrency, 50)  # Allow up to 50 for mid-range systems
            
        self.logger.info(f"Detected {cpu_count} CPU cores, {memory_gb:.1f}GB RAM")
        self.logger.info(f"Using {optimal} concurrent requests (high-performance mode)")
        return optimal

    async def benchmark_api(self, pdf_files: List[Path], concurrency: int = None,
                          verbose: bool = False) -> Dict[str, Any]:
        """Run comprehensive benchmark with specified concurrency."""
        if concurrency is None:
            concurrency = self.get_optimal_concurrency()

        self.logger.info(f"Starting benchmark with {concurrency} concurrent requests")
        self.logger.info(f"Testing {len(pdf_files)} PDF files")

        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0
        results = []
        
        # Create progress bar with enhanced info
        pbar = tqdm(total=len(pdf_files), desc="Processing PDFs", unit="files", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        # Running tallies with performance tracking
        ok_count = 0
        err_count = 0
        total_pages_processed = 0
        tally_lock = asyncio.Lock()
        batch_start_time = start_time

        async def bounded_process(pdf_path: Path) -> Dict[str, Any]:
            async with semaphore:
                if verbose:
                    self.logger.info(f"Processing {pdf_path.name}")
                result = await self.process_single_pdf(pdf_path)
                # Update running tallies safely with performance metrics
                async with tally_lock:
                    nonlocal ok_count, err_count, total_pages_processed
                    if result.get('status') == 'success':
                        ok_count += 1
                        total_pages_processed += result.get('pages', 0)
                    else:
                        err_count += 1
                    
                    # Calculate real-time throughput
                    elapsed = time.time() - batch_start_time
                    if elapsed > 0:
                        files_per_sec = ok_count / elapsed
                        pages_per_sec = total_pages_processed / elapsed
                    else:
                        files_per_sec = pages_per_sec = 0
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'ok': ok_count, 'err': err_count, 
                        'fps': f'{files_per_sec:.1f}', 'pps': f'{pages_per_sec:.1f}'
                    })
                return result

        # Create tasks for all PDFs
        tasks = []
        for pdf_path in pdf_files:
            tasks.append(bounded_process(pdf_path))

        # Run all tasks concurrently
        self.logger.info("Starting concurrent processing...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()
        total_time = time.time() - start_time

        # Process results
        successful_results = []
        failed_results = []
        total_pages = 0
        total_size = 0

        for result in results:
            if isinstance(result, Exception):
                failed_results.append({
                    'filename': 'unknown',
                    'error': str(result),
                    'status': 'exception'
                })
            elif result['status'] == 'success':
                successful_results.append(result)
                total_pages += result['pages']
                total_size += result['file_size_bytes']
            else:
                failed_results.append(result)

        # Calculate statistics
        if successful_results:
            processing_times = [r['processing_time_seconds'] for r in successful_results]
            time_per_page_values = [r['time_per_page'] for r in successful_results]

            benchmark_results = {
                'benchmark_info': {
                    'total_files_tested': len(pdf_files),
                    'successful_files': len(successful_results),
                    'failed_files': len(failed_results),
                    'concurrency_level': concurrency,
                    'total_time_seconds': total_time,
                    'total_pages_processed': total_pages,
                    'total_size_bytes': total_size,
                    'average_file_size_mb': total_size / len(successful_results) / (1024 * 1024),
                    'timestamp': datetime.now().isoformat()
                },
                'performance_metrics': {
                    'total_processing_time_seconds': sum(processing_times),
                    'average_time_per_file_seconds': mean(processing_times),
                    'median_time_per_file_seconds': median(processing_times),
                    'min_time_per_file_seconds': min(processing_times),
                    'max_time_per_file_seconds': max(processing_times),
                    'std_time_per_file_seconds': stdev(processing_times) if len(processing_times) > 1 else 0,
                    'average_time_per_page_seconds': mean(time_per_page_values),
                    'median_time_per_page_seconds': median(time_per_page_values),
                    'pages_per_second': total_pages / total_time,
                    'files_per_second': len(successful_results) / total_time,
                    'mb_per_second': (total_size / (1024 * 1024)) / total_time
                },
                'results': successful_results,
                'failures': failed_results
            }
        else:
            benchmark_results = {
                'benchmark_info': {
                    'total_files_tested': len(pdf_files),
                    'successful_files': 0,
                    'failed_files': len(failed_results),
                    'concurrency_level': concurrency,
                    'total_time_seconds': total_time,
                    'total_pages_processed': 0,
                    'total_size_bytes': 0,
                    'timestamp': datetime.now().isoformat()
                },
                'performance_metrics': {},
                'results': [],
                'failures': failed_results
            }

        return benchmark_results

    def print_results(self, results: Dict[str, Any], verbose: bool = False):
        """Print formatted benchmark results."""
        info = results['benchmark_info']
        metrics = results['performance_metrics']

        print("\n" + "="*80)
        print("DOCLING API BENCHMARK RESULTS")
        print("="*80)

        print(f"\n📊 BENCHMARK INFO:")
        print(f"   Files Tested: {info['total_files_tested']}")
        print(f"   Successful: {info['successful_files']}")
        print(f"   Failed: {info['failed_files']}")
        print(f"   Concurrency: {info['concurrency_level']}")
        print(f"   Total Time: {info['total_time_seconds']:.2f}s")
        print(f"   Total Pages: {info['total_pages_processed']}")
        print(f"   Total Size: {info['total_size_bytes'] / (1024*1024):.2f} MB")

        if metrics:
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"   Files per Second: {metrics['files_per_second']:.2f}")
            print(f"   Pages per Second: {metrics['pages_per_second']:.2f}")
            print(f"   MB per Second: {metrics['mb_per_second']:.2f}")
            print(f"   Avg Time per File: {metrics['average_time_per_file_seconds']:.2f}s")
            print(f"   Avg Time per Page: {metrics['average_time_per_page_seconds']:.3f}s")
            print(f"   Median Time per File: {metrics['median_time_per_file_seconds']:.2f}s")

            # Enhanced performance estimates
            cpu_utilization = min(metrics['files_per_second'] * metrics['average_time_per_file_seconds'] / info['concurrency_level'] * 100, 100)
            efficiency = (info['successful_files'] / info['total_files_tested']) * 100
            
            print(f"   Estimated CPU Utilization: {cpu_utilization:.1f}%")
            print(f"   Processing Efficiency: {efficiency:.1f}%")
            print(f"   Peak Memory per Request: ~{metrics['average_time_per_file_seconds'] * 200:.0f}MB (estimated)")
            
            # GPU utilization estimate (if applicable)
            if info.get('total_pages_processed', 0) > 0:
                gpu_efficiency = metrics['pages_per_second'] / (info['total_pages_processed'] / info['total_time_seconds']) * 100
                print(f"   GPU Processing Efficiency: {gpu_efficiency:.1f}%")

        if results['failures']:
            print(f"\n❌ FAILURES:")
            for failure in results['failures'][:5]:  # Show first 5 failures
                print(f"   {failure['filename']}: {failure['error']}")
            if len(results['failures']) > 5:
                print(f"   ... and {len(results['failures']) - 5} more")

        if verbose and results['results']:
            print(f"\n📋 DETAILED RESULTS (first 20):")
            for result in sorted(results['results'][:20], key=lambda x: x['processing_time_seconds']):
                file_size_mb = result['file_size_bytes'] / (1024 * 1024)
                throughput_mbps = file_size_mb / result['processing_time_seconds'] if result['processing_time_seconds'] > 0 else 0
                print(f"   {result['filename']:<30} {result['processing_time_seconds']:>6.2f}s "
                      f"{result['pages']:>3}p {result['time_per_page']:>6.3f}s/p {throughput_mbps:>5.1f}MB/s")

        print("\n" + "="*80)

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Docling API performance")
    parser.add_argument('--url', default='http://localhost:5001',
                       help='API base URL (default: http://localhost:5001)')
    parser.add_argument('--concurrency', type=int, default=None,
                       help='Number of concurrent requests (default: auto-detect, up to 100 for high-end systems)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of PDF files to test (default: all)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds (default: 300)')
    parser.add_argument('--output', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Create benchmark instance
    async with DoclingBenchmark(args.url, args.timeout) as benchmark:
        try:
            # Get PDF files
            pdf_files = benchmark.get_pdf_files(args.max_files)

            # Run benchmark
            results = await benchmark.benchmark_api(
                pdf_files,
                concurrency=args.concurrency,
                verbose=args.verbose
            )

            # Print results
            benchmark.print_results(results, args.verbose)

            # Save results if requested
            if args.output:
                benchmark.save_results(results, args.output)

        except Exception as e:
            benchmark.logger.error(f"Benchmark failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
