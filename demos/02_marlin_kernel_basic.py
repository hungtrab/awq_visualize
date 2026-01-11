"""
Marlin Kernel Basic Demo
========================

Demonstrates basic Marlin kernel usage:
- FP16×INT4 matrix multiplication
- Memory layout and data packing
- Performance comparison with standard matmul
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.profiling import Benchmarker, CUDATimer, profile_scope
from utils.visualization import VisualizationHelper


def pack_int4_weights(weights_fp16: torch.Tensor) -> tuple:
    """
    Pack FP16 weights into INT4 format (Marlin-compatible)
    
    Args:
        weights_fp16: FP16 weight tensor [M, K]
    
    Returns:
        Tuple of (packed_weights, scales)
    """
    # Quantize to INT4 (-8 to 7 for signed 4-bit)
    w_abs_max = weights_fp16.abs().max(dim=-1, keepdim=True)[0]
    scale = w_abs_max / 7.0  # Max value for 4-bit signed
    
    weights_quantized = torch.clamp(
        torch.round(weights_fp16 / (scale + 1e-8)),
        min=-8,
        max=7
    ).to(torch.int8)
    
    # Pack two INT4 values into one byte
    # weights_quantized shape: [M, K]
    M, K = weights_quantized.shape
    assert K % 2 == 0, "K must be even for INT4 packing"
    
    # Pack pairs: [w0, w1] -> byte
    packed = torch.zeros(M, K // 2, dtype=torch.uint8, device=weights_fp16.device)
    
    for i in range(M):
        for j in range(0, K, 2):
            # Get two 4-bit values
            val0 = weights_quantized[i, j].item() & 0x0F  # Lower 4 bits
            val1 = weights_quantized[i, j+1].item() & 0x0F  # Lower 4 bits
            
            # Pack into single byte: val1 in upper 4 bits, val0 in lower 4 bits
            packed[i, j//2] = (val1 << 4) | val0
    
    return packed, scale


def unpack_int4_weights(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Unpack INT4 weights to FP16 (dequantization)
    
    Args:
        packed: Packed INT4 weights [M, K//2]
        scale: Scaling factors [M, 1]
    
    Returns:
        Unpacked FP16 weights [M, K]
    """
    M, K_half = packed.shape
    K = K_half * 2
    
    unpacked = torch.zeros(M, K, dtype=torch.float16, device=packed.device)
    
    for i in range(M):
        for j in range(K_half):
            byte_val = packed[i, j].item()
            
            # Extract two 4-bit values
            val0 = (byte_val & 0x0F)  # Lower 4 bits
            val1 = (byte_val >> 4) & 0x0F  # Upper 4 bits
            
            # Convert from unsigned to signed 4-bit
            if val0 >= 8:
                val0 = val0 - 16
            if val1 >= 8:
                val1 = val1 - 16
            
            # Dequantize
            unpacked[i, j*2] = val0 * scale[i, 0]
            unpacked[i, j*2 + 1] = val1 * scale[i, 0]
    
    return unpacked


def simulate_marlin_matmul(A_fp16: torch.Tensor, 
                          B_packed: torch.Tensor, 
                          B_scales: torch.Tensor) -> torch.Tensor:
    """
    Simulate Marlin kernel behavior:
    1. Unpack INT4 weights to FP16
    2. Perform FP16 matmul
    
    In real Marlin kernel, unpacking happens on-the-fly in shared memory
    
    Args:
        A_fp16: Activation tensor [batch, K]
        B_packed: Packed INT4 weights [N, K//2]
        B_scales: Weight scales [N, 1]
    
    Returns:
        Output tensor [batch, N]
    """
    # Unpack weights
    B_fp16 = unpack_int4_weights(B_packed, B_scales)
    
    # Transpose for matmul: [N, K] -> [K, N]
    B_fp16_T = B_fp16.T
    
    # FP16 matmul: [batch, K] @ [K, N] = [batch, N]
    C = torch.matmul(A_fp16, B_fp16_T)
    
    return C


def compare_matmul_methods(M: int, N: int, K: int, device: str = "cuda"):
    """
    Compare different matrix multiplication methods
    
    Args:
        M: Batch size / number of rows in A
        N: Output dimension / number of rows in B
        K: Hidden dimension
        device: Device to run on
    """
    print(f"\n{'='*70}")
    print(f"  Matrix Multiplication Comparison")
    print(f"  Dimensions: A=[{M}, {K}] × B=[{N}, {K}]^T = C=[{M}, {N}]")
    print(f"{'='*70}\n")
    
    # Create input matrices
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)
    
    print("📦 Packing weights to INT4...")
    B_packed, B_scales = pack_int4_weights(B_fp16)
    
    # Calculate memory reduction
    fp16_size = B_fp16.numel() * 2  # 2 bytes per FP16
    int4_size = B_packed.numel() * 1 + B_scales.numel() * 2  # 1 byte per packed INT4 + scales
    reduction = fp16_size / int4_size
    
    print(f"   FP16 size: {fp16_size / 1024:.2f} KB")
    print(f"   INT4 size: {int4_size / 1024:.2f} KB")
    print(f"   Reduction: {reduction:.2f}×\n")
    
    # Benchmark methods
    benchmarker = Benchmarker(warmup_iterations=20, num_iterations=100)
    
    # Method 1: Standard FP16×FP16
    print("⚡ Benchmarking FP16×FP16...")
    def fp16_matmul():
        return torch.matmul(A_fp16, B_fp16.T)
    
    result_fp16 = benchmarker.benchmark_function(fp16_matmul, "FP16×FP16")
    
    # Method 2: Simulated Marlin (unpack + matmul)
    print("⚡ Benchmarking Simulated Marlin (FP16×INT4)...")
    def marlin_matmul():
        return simulate_marlin_matmul(A_fp16, B_packed, B_scales)
    
    result_marlin = benchmarker.benchmark_function(marlin_matmul, "Simulated Marlin")
    
    # Verify correctness
    print("\n🔍 Verifying correctness...")
    C_fp16 = fp16_matmul()
    C_marlin = marlin_matmul()
    
    mse = torch.mean((C_fp16 - C_marlin) ** 2).item()
    relative_error = mse / torch.mean(C_fp16 ** 2).item()
    
    print(f"   MSE: {mse:.6f}")
    print(f"   Relative Error: {relative_error * 100:.2f}%")
    
    if relative_error < 0.01:  # Less than 1% error
        print("   ✅ Results match within tolerance")
    else:
        print("   ⚠️  Warning: High error detected")
    
    # Print results
    benchmarker.print_results()
    
    # Speedup analysis
    speedup = result_fp16.elapsed_ms / result_marlin.elapsed_ms
    print(f"\n📊 Analysis:")
    print(f"   Speedup: {speedup:.2f}×")
    print(f"   Memory saved: {reduction:.2f}×")
    
    print(f"\n💡 Note: Real Marlin kernel is much faster because it:")
    print(f"   • Unpacks INT4 on-the-fly in shared memory")
    print(f"   • Uses optimized tensor core operations")
    print(f"   • Employs advanced memory pipelining")
    print(f"   • Expected speedup: 2-4× vs FP16 baseline\n")
    
    return {
        'fp16_time_ms': result_fp16.elapsed_ms,
        'marlin_time_ms': result_marlin.elapsed_ms,
        'memory_reduction': reduction,
        'accuracy_error': relative_error
    }


def visualize_int4_packing():
    """Visualize how INT4 packing works"""
    print("\n" + "="*70)
    print("  INT4 Weight Packing Visualization")
    print("="*70 + "\n")
    
    # Create small example
    example_weights = torch.tensor([
        [0.5, -0.3, 0.8, -0.1, 0.6, -0.4, 0.2, 0.0],
        [-0.7, 0.4, -0.2, 0.9, -0.5, 0.3, -0.1, 0.7]
    ], dtype=torch.float16, device="cpu")
    
    print("Original FP16 weights (2×8):")
    print(example_weights)
    print()
    
    # Pack
    packed, scales = pack_int4_weights(example_weights)
    
    print("Scales:")
    print(scales)
    print()
    
    print("Packed INT4 (2×4 bytes, each byte contains 2 INT4 values):")
    for i in range(packed.shape[0]):
        row_str = "Row " + str(i) + ": "
        for j in range(packed.shape[1]):
            byte_val = packed[i, j].item()
            val0 = (byte_val & 0x0F)
            val1 = (byte_val >> 4) & 0x0F
            row_str += f"[{val0:X}{val1:X}] "
        print(row_str)
    print()
    
    # Unpack
    unpacked = unpack_int4_weights(packed, scales)
    
    print("Unpacked back to FP16:")
    print(unpacked)
    print()
    
    # Error
    error = torch.abs(example_weights - unpacked)
    print("Quantization error:")
    print(error)
    print(f"Max error: {error.max().item():.4f}")
    print(f"Mean error: {error.mean().item():.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Marlin Kernel Basic Demo")
    parser.add_argument("--m", type=int, default=128, help="Batch size")
    parser.add_argument("--n", type=int, default=512, help="Output dimension")
    parser.add_argument("--k", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--verify", action="store_true", help="Run verification tests")
    parser.add_argument("--visualize-packing", action="store_true", help="Visualize INT4 packing")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Running on CPU (will be slow).")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}\n")
    
    if args.visualize_packing:
        visualize_int4_packing()
    
    if args.verify or not args.visualize_packing:
        results = compare_matmul_methods(args.m, args.n, args.k, device=device)
    
    print("="*70)
    print("  Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
