"""
Profiling utilities for benchmarking and performance analysis
=============================================================

Provides tools for:
- CUDA event timing
- Memory profiling
- Bandwidth measurement
- Token/s calculation
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class ProfilingResult:
    """Container for profiling results"""
    name: str
    elapsed_ms: float
    throughput: Optional[float] = None  # operations/second
    memory_used_mb: Optional[float] = None
    flops: Optional[float] = None
    bandwidth_gb_s: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class CUDATimer:
    """High-precision CUDA event timer"""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_event = None
        self.end_event = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.cpu_start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self.cpu_start) * 1000
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return self.elapsed_ms
    
    def get_elapsed_s(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed_ms / 1000.0


class MemoryProfiler:
    """Track GPU memory usage"""
    
    def __init__(self, device: int = 0):
        self.device = device
        self.initial_memory = 0
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
            self.initial_memory = torch.cuda.memory_allocated(self.device)
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_memory = torch.cuda.max_memory_allocated(self.device)
            self.current_memory = torch.cuda.memory_allocated(self.device)
    
    def get_memory_usage_mb(self) -> Dict[str, float]:
        """Get memory usage in MB"""
        return {
            'initial_mb': self.initial_memory / 1024**2,
            'current_mb': self.current_memory / 1024**2,
            'peak_mb': self.peak_memory / 1024**2,
            'allocated_mb': (self.current_memory - self.initial_memory) / 1024**2
        }


class ThroughputCalculator:
    """Calculate throughput metrics"""
    
    @staticmethod
    def tokens_per_second(num_tokens: int, elapsed_s: float) -> float:
        """Calculate token generation throughput"""
        return num_tokens / elapsed_s if elapsed_s > 0 else 0
    
    @staticmethod
    def flops(operations: int, elapsed_s: float) -> float:
        """Calculate FLOPS (floating point operations per second)"""
        return operations / elapsed_s if elapsed_s > 0 else 0
    
    @staticmethod
    def bandwidth(bytes_transferred: int, elapsed_s: float) -> float:
        """Calculate memory bandwidth in GB/s"""
        return (bytes_transferred / elapsed_s) / 1e9 if elapsed_s > 0 else 0
    
    @staticmethod
    def matmul_flops(m: int, n: int, k: int) -> int:
        """Calculate FLOPs for matrix multiplication C = A @ B
        where A is (m, k), B is (k, n), C is (m, n)
        """
        return 2 * m * n * k


class Benchmarker:
    """Comprehensive benchmarking tool"""
    
    def __init__(self, warmup_iterations: int = 10, num_iterations: int = 100):
        self.warmup_iterations = warmup_iterations
        self.num_iterations = num_iterations
        self.results: List[ProfilingResult] = []
    
    def benchmark_function(self,
                          func: Callable,
                          name: str,
                          *args,
                          track_memory: bool = True,
                          **kwargs) -> ProfilingResult:
        """
        Benchmark a function with warmup
        
        Args:
            func: Function to benchmark
            name: Name for this benchmark
            track_memory: Whether to track memory usage
            *args, **kwargs: Arguments to pass to func
        
        Returns:
            ProfilingResult
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        memory_results = []
        
        for _ in range(self.num_iterations):
            with CUDATimer() as timer:
                if track_memory and torch.cuda.is_available():
                    with MemoryProfiler() as mem_prof:
                        result = func(*args, **kwargs)
                    memory_results.append(mem_prof.get_memory_usage_mb())
                else:
                    result = func(*args, **kwargs)
            
            times.append(timer.get_elapsed_ms())
        
        # Calculate statistics
        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        
        memory_mb = None
        if memory_results:
            memory_mb = np.mean([m['allocated_mb'] for m in memory_results])
        
        profile_result = ProfilingResult(
            name=name,
            elapsed_ms=avg_time_ms,
            memory_used_mb=memory_mb,
            metadata={
                'std_ms': std_time_ms,
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'num_iterations': self.num_iterations
            }
        )
        
        self.results.append(profile_result)
        return profile_result
    
    def benchmark_matmul(self,
                        m: int, n: int, k: int,
                        dtype: torch.dtype = torch.float16,
                        device: str = "cuda") -> ProfilingResult:
        """
        Benchmark matrix multiplication
        
        Args:
            m, n, k: Matrix dimensions
            dtype: Data type
            device: Device to run on
        
        Returns:
            ProfilingResult with FLOPS
        """
        A = torch.randn(m, k, dtype=dtype, device=device)
        B = torch.randn(k, n, dtype=dtype, device=device)
        
        def matmul_fn():
            return torch.matmul(A, B)
        
        result = self.benchmark_function(
            matmul_fn,
            name=f"matmul_{m}x{n}x{k}_{dtype}",
            track_memory=True
        )
        
        # Calculate FLOPS
        total_flops = ThroughputCalculator.matmul_flops(m, n, k)
        result.flops = ThroughputCalculator.flops(
            total_flops,
            result.elapsed_ms / 1000.0
        ) / 1e12  # Convert to TFLOPS
        
        # Estimate bandwidth (simplified)
        bytes_transferred = (m * k + k * n + m * n) * (2 if dtype == torch.float16 else 4)
        result.bandwidth_gb_s = ThroughputCalculator.bandwidth(
            bytes_transferred,
            result.elapsed_ms / 1000.0
        )
        
        return result
    
    def print_results(self):
        """Print all benchmark results in a formatted table"""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 100)
        print(f"{'Operation':<30} {'Avg Time (ms)':<15} {'TFLOPS':<12} {'BW (GB/s)':<12} {'Memory (MB)':<12}")
        print("=" * 100)
        
        for result in self.results:
            flops_str = f"{result.flops:.2f}" if result.flops else "N/A"
            bw_str = f"{result.bandwidth_gb_s:.2f}" if result.bandwidth_gb_s else "N/A"
            mem_str = f"{result.memory_used_mb:.2f}" if result.memory_used_mb else "N/A"
            
            print(f"{result.name:<30} {result.elapsed_ms:>14.3f} {flops_str:>11} {bw_str:>11} {mem_str:>11}")
        
        print("=" * 100 + "\n")
    
    def get_summary(self) -> Dict:
        """Get summary statistics of all benchmarks"""
        if not self.results:
            return {}
        
        return {
            'num_benchmarks': len(self.results),
            'total_time_ms': sum(r.elapsed_ms for r in self.results),
            'avg_time_ms': np.mean([r.elapsed_ms for r in self.results]),
            'total_memory_mb': sum(r.memory_used_mb for r in self.results if r.memory_used_mb),
            'avg_flops': np.mean([r.flops for r in self.results if r.flops]) if any(r.flops for r in self.results) else None
        }


@contextmanager
def profile_scope(name: str, print_result: bool = True):
    """
    Context manager for quick profiling
    
    Usage:
        with profile_scope("my_operation"):
            # code to profile
            ...
    """
    timer = CUDATimer(name)
    mem_prof = MemoryProfiler() if torch.cuda.is_available() else None
    
    start_time = time.time()
    
    if mem_prof:
        with mem_prof:
            with timer:
                yield
    else:
        with timer:
            yield
    
    elapsed = timer.get_elapsed_ms()
    
    if print_result:
        print(f"\n[{name}] Time: {elapsed:.3f} ms")
        if mem_prof:
            mem_info = mem_prof.get_memory_usage_mb()
            print(f"[{name}] Memory: {mem_info['allocated_mb']:.2f} MB allocated, "
                  f"{mem_info['peak_mb']:.2f} MB peak")


def compare_implementations(implementations: Dict[str, Callable],
                           args_list: List = None,
                           kwargs_list: List[Dict] = None,
                           num_iterations: int = 100):
    """
    Compare multiple implementations of the same operation
    
    Args:
        implementations: Dict of {name: function}
        args_list: List of args to pass to each function
        kwargs_list: List of kwargs to pass to each function
        num_iterations: Number of iterations per benchmark
    
    Returns:
        Dict of ProfilingResults
    """
    if args_list is None:
        args_list = [() for _ in implementations]
    if kwargs_list is None:
        kwargs_list = [{} for _ in implementations]
    
    benchmarker = Benchmarker(warmup_iterations=10, num_iterations=num_iterations)
    results = {}
    
    for (name, func), args, kwargs in zip(implementations.items(), args_list, kwargs_list):
        print(f"Benchmarking {name}...")
        result = benchmarker.benchmark_function(func, name, *args, **kwargs)
        results[name] = result
    
    benchmarker.print_results()
    
    return results
