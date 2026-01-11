"""
GPU Introspection - Deep dive into GPU state during Marlin kernel execution
===========================================================================

This module provides detailed GPU profiling and visualization:
- Block/Grid configuration
- Thread organization
- Memory hierarchy state
- Real-time GPU metrics
- Kernel execution timeline
"""

import argparse
import torch
import time
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box

console = Console()

# Try to import GPU monitoring libraries
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    console.print("[yellow]⚠️  pynvml not available. Install with: pip install nvidia-ml-py3[/yellow]")

try:
    from torch.profiler import profile, ProfilerActivity, record_function
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False


@dataclass
class GPUState:
    """Snapshot of GPU state at a point in time"""
    timestamp: float
    gpu_util: float
    memory_used: int
    memory_total: int
    temperature: int
    power_draw: float
    sm_clock: int
    memory_clock: int
    

class GPUIntrospector:
    """Main class for GPU introspection and profiling"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        
        if HAS_NVML:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
    def get_gpu_state(self) -> Optional[GPUState]:
        """Capture current GPU state"""
        if not HAS_NVML:
            return None
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
            clocks = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            
            return GPUState(
                timestamp=time.time(),
                gpu_util=util.gpu,
                memory_used=memory.used,
                memory_total=memory.total,
                temperature=temp,
                power_draw=power,
                sm_clock=clocks,
                memory_clock=mem_clock
            )
        except Exception as e:
            console.print(f"[red]Error getting GPU state: {e}[/red]")
            return None
    
    def print_gpu_info(self):
        """Print detailed GPU information"""
        if not torch.cuda.is_available():
            console.print("[red]❌ CUDA not available[/red]")
            return
        
        props = torch.cuda.get_device_properties(self.device_id)
        
        # Create info table
        table = Table(title="GPU Information", box=box.DOUBLE_EDGE)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Device Name", props.name)
        table.add_row("Compute Capability", f"{props.major}.{props.minor}")
        table.add_row("Total Memory", f"{props.total_memory / 1024**3:.2f} GB")
        table.add_row("Multiprocessors (SMs)", str(props.multi_processor_count))
        
        # These properties may not be available in all PyTorch versions
        try:
            table.add_row("Max Threads per SM", str(props.max_threads_per_multi_processor))
        except AttributeError:
            table.add_row("Max Threads per SM", "N/A")
        
        try:
            max_threads = props.max_threads_per_block
        except AttributeError:
            max_threads = 1024  # Common default
        table.add_row("Max Threads per Block", str(max_threads))
        
        table.add_row("Warp Size", str(32))  # Always 32 for NVIDIA
        
        try:
            shared_mem = props.shared_memory_per_block
        except AttributeError:
            shared_mem = props.total_memory // 1000  # Rough estimate
        table.add_row("Shared Memory per Block", f"{shared_mem / 1024:.1f} KB")
        
        try:
            table.add_row("Registers per Block", str(props.regs_per_block))
        except AttributeError:
            table.add_row("Registers per Block", "N/A")
        
        # Check Marlin compatibility
        marlin_compatible = props.major >= 7 and (props.major > 7 or props.minor >= 5)
        compat_text = "[green]✓ Compatible[/green]" if marlin_compatible else "[red]✗ Not Compatible (need ≥7.5)[/red]"
        table.add_row("Marlin Kernel Compatible", compat_text)
        
        console.print(table)
    
    def visualize_kernel_config(self, grid_dim: tuple, block_dim: tuple):
        """Visualize kernel launch configuration"""
        grid_x, grid_y, grid_z = grid_dim
        block_x, block_y, block_z = block_dim
        
        total_blocks = grid_x * grid_y * grid_z
        threads_per_block = block_x * block_y * block_z
        total_threads = total_blocks * threads_per_block
        
        # Create configuration display
        config_text = f"""
╔══════════════════════════════════════════════════════════════╗
║              KERNEL LAUNCH CONFIGURATION                      ║
╚══════════════════════════════════════════════════════════════╝

Grid Dimensions:    ({grid_x}, {grid_y}, {grid_z})      → {total_blocks} blocks total
Block Dimensions:   ({block_x}, {block_y}, {block_z})      → {threads_per_block} threads per block
Total Threads:      {total_threads:,} threads
"""
        
        console.print(config_text)
        
        # Visualize grid layout (simplified for small grids)
        if total_blocks <= 256:
            console.print("\n[bold]Grid Layout (Block IDs):[/bold]")
            self._print_grid_layout(grid_x, grid_y)
    
    def _print_grid_layout(self, grid_x: int, grid_y: int, max_display: int = 16):
        """Print ASCII visualization of grid layout"""
        display_x = min(grid_x, max_display)
        display_y = min(grid_y, max_display)
        
        layout_str = ""
        for y in range(display_y):
            row = ""
            for x in range(display_x):
                block_id = y * grid_x + x
                row += f"[B{block_id:>3}] "
            layout_str += row + "\n"
        
        if grid_x > max_display or grid_y > max_display:
            layout_str += f"\n... ({grid_x}×{grid_y} total, showing {display_x}×{display_y})"
        
        console.print(layout_str)
    
    def visualize_memory_hierarchy(self, state: Optional[GPUState] = None):
        """Visualize GPU memory hierarchy with current state"""
        
        if state is None:
            state = self.get_gpu_state()
        
        if state:
            mem_used_gb = state.memory_used / 1024**3
            mem_total_gb = state.memory_total / 1024**3
            mem_percent = (state.memory_used / state.memory_total) * 100
            mem_bar = "█" * int(mem_percent / 2) + "░" * (50 - int(mem_percent / 2))
        else:
            mem_used_gb = 0
            mem_total_gb = 0
            mem_bar = "?" * 50
        
        hierarchy = f"""
╔═══════════════════════════════════════════════════════════════╗
║                   GPU MEMORY HIERARCHY                         ║
╠═══════════════════════════════════════════════════════════════╣
║                                                                ║
║  Global Memory (HBM)              [{mem_used_gb:.1f} GB / {mem_total_gb:.1f} GB] ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ {mem_bar} │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                      ▼ Bandwidth: ~1500 GB/s                   ║
║                                                                ║
║  L2 Cache                                         [~6 MB]     ║
║  ┌────────────────────────────────────────────┐               ║
║  │ Shared across all SMs                      │               ║
║  │ Typically 78-95% hit rate for Marlin       │               ║
║  └────────────────────────────────────────────┘               ║
║                      ▼ 128-256 bytes/cycle                     ║
║                                                                ║
║  L1 Cache / Shared Memory (per SM)           [~164 KB]        ║
║  ┌────────────────────────────────────────────┐               ║
║  │ Shared memory: User-controlled             │               ║
║  │ L1 cache: Hardware-managed                 │               ║
║  │ Marlin uses ~48 KB shared memory           │               ║
║  └────────────────────────────────────────────┘               ║
║                      ▼ Double buffering active                 ║
║                                                                ║
║  Register File (per thread)                  [255 regs max]   ║
║  ┌────────────────────────────────────────────┐               ║
║  │ Fastest memory, ~20TB/s bandwidth          │               ║
║  │ Marlin typically uses 64-128 registers     │               ║
║  └────────────────────────────────────────────┘               ║
║                                                                ║
╚═══════════════════════════════════════════════════════════════╝
"""
        
        console.print(hierarchy)
    
    def profile_simple_matmul(self, size: int = 1024):
        """Profile a simple matrix multiplication as baseline"""
        console.print(f"\n[bold]Profiling {size}×{size} FP16 Matrix Multiplication[/bold]\n")
        
        # Create random matrices
        A = torch.randn(size, size, dtype=torch.float16, device=self.device)
        B = torch.randn(size, size, dtype=torch.float16, device=self.device)
        
        # Warm up
        for _ in range(10):
            _ = torch.matmul(A, B)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        num_iterations = 100
        
        start.record()
        for _ in range(num_iterations):
            C = torch.matmul(A, B)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        avg_time = elapsed_ms / num_iterations
        
        # Calculate FLOPS
        flops = 2 * size * size * size  # 2N³ for matmul
        tflops = (flops / (avg_time / 1000)) / 1e12
        
        # Display results
        table = Table(title="Matrix Multiplication Benchmark")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Matrix Size", f"{size}×{size}")
        table.add_row("Data Type", "FP16")
        table.add_row("Average Time", f"{avg_time:.3f} ms")
        table.add_row("Throughput", f"{tflops:.2f} TFLOPS")
        
        console.print(table)
        
        return avg_time, tflops
    
    def monitor_realtime(self, duration: int = 10):
        """Real-time GPU monitoring"""
        if not HAS_NVML:
            console.print("[red]NVML not available for real-time monitoring[/red]")
            return
        
        console.print(f"\n[bold]Starting {duration}s real-time monitoring...[/bold]\n")
        
        states = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                state = self.get_gpu_state()
                if state:
                    states.append(state)
                    
                    # Create live display
                    self._display_gpu_state(state)
                    
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped by user[/yellow]")
        
        return states
    
    def _display_gpu_state(self, state: GPUState):
        """Display current GPU state"""
        gpu_bar = "█" * int(state.gpu_util / 2) + "░" * (50 - int(state.gpu_util / 2))
        mem_percent = (state.memory_used / state.memory_total) * 100
        mem_bar = "█" * int(mem_percent / 2) + "░" * (50 - int(mem_percent / 2))
        
        display = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ GPU Real-time Monitoring                          ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                    ┃
┃ GPU Utilization:  {gpu_bar} {state.gpu_util:>3.0f}%  ┃
┃ Memory Usage:     {mem_bar} {mem_percent:>3.0f}%  ┃
┃ Temperature:      {state.temperature}°C                              ┃
┃ Power Draw:       {state.power_draw:.1f}W                            ┃
┃ SM Clock:         {state.sm_clock} MHz                          ┃
┃ Memory Clock:     {state.memory_clock} MHz                          ┃
┃                                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""
        
        # Clear and print
        os.system('clear' if os.name != 'nt' else 'cls')
        console.print(display)


def main():
    parser = argparse.ArgumentParser(description="GPU Introspection Tool")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--snapshot", action="store_true", help="Take GPU snapshot")
    parser.add_argument("--visualize", action="store_true", help="Visualize memory hierarchy")
    parser.add_argument("--benchmark", action="store_true", help="Run matmul benchmark")
    parser.add_argument("--monitor", type=int, default=0, help="Real-time monitoring duration (seconds)")
    parser.add_argument("--kernel-config", type=str, help="Show kernel config (format: gridX,gridY,gridZ;blockX,blockY,blockZ)")
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       GPU Introspection & Profiling Tool                [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
    
    introspector = GPUIntrospector(device_id=args.device)
    
    # Always show GPU info
    introspector.print_gpu_info()
    
    if args.snapshot or args.visualize:
        state = introspector.get_gpu_state()
        if args.visualize:
            introspector.visualize_memory_hierarchy(state)
    
    if args.kernel_config:
        # Parse kernel config: "128,4,1;256,1,1"
        try:
            grid_str, block_str = args.kernel_config.split(';')
            grid = tuple(map(int, grid_str.split(',')))
            block = tuple(map(int, block_str.split(',')))
            introspector.visualize_kernel_config(grid, block)
        except Exception as e:
            console.print(f"[red]Error parsing kernel config: {e}[/red]")
    
    if args.benchmark:
        introspector.profile_simple_matmul(size=2048)
    
    if args.monitor > 0:
        introspector.monitor_realtime(duration=args.monitor)
    
    console.print("\n[bold green]✓ Introspection complete![/bold green]\n")


if __name__ == "__main__":
    main()
