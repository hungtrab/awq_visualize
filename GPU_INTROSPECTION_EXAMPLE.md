# GPU Introspection Example Output

Đây là ví dụ về những gì bạn sẽ thấy khi chạy GPU introspection tool.

## 1. Kernel Launch Configuration

```
╔══════════════════════════════════════════════════════════════╗
║              MARLIN KERNEL CONFIGURATION                      ║
╚══════════════════════════════════════════════════════════════╝

Grid Dimensions:    (128, 4, 1)      → 512 blocks total
Block Dimensions:   (256, 1, 1)      → 256 threads per block
Total Threads:      131,072 threads
Shared Memory:      48 KB per block
Registers:          64 per thread
Theoretical Occupancy: 75%
Achieved Occupancy:    68%
```

## 2. Thread Block Layout Visualization

```
GPU Grid Layout (Top View of Streaming Multiprocessors)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SM 0:  [B0 ][B4 ][B8 ][B12]  ← Block IDs
SM 1:  [B1 ][B5 ][B9 ][B13]
SM 2:  [B2 ][B6 ][B10][B14]
SM 3:  [B3 ][B7 ][B11][B15]
...
SM 80: [B320][B324][B328][B332]

Legend: B# = Block ID | █ = Active | ░ = Idle
```

## 3. Memory Hierarchy Snapshot

```
╔═══════════════════════════════════════════════════════════════╗
║                   GPU MEMORY HIERARCHY                         ║
╠═══════════════════════════════════════════════════════════════╣
║                                                                ║
║  Global Memory (HBM)                          [12 GB / 24 GB] ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Weights (INT4): 3.2 GB      ████████████░░░░░░░░░       │ ║
║  │ Activations:    1.8 GB      ███████░░░░░░░░░░░░░░       │ ║
║  │ Intermediate:   0.5 GB      ██░░░░░░░░░░░░░░░░░░░       │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                      ▼ Bandwidth: 875 GB/s (87% peak)          ║
║                                                                ║
║  L2 Cache                                     [4.5 MB / 6 MB] ║
║  ┌────────────────────────────────────────────┐               ║
║  │ Weight tiles cached    ███████████████░░░ │               ║
║  │ Hit rate: 78%                              │               ║
║  └────────────────────────────────────────────┘               ║
║                      ▼ 128 bytes/cycle                         ║
║                                                                ║
║  L1 Cache / Shared Memory (per SM)           [128 KB / 164 KB]║
║  ┌────────────────────────────────────────────┐               ║
║  │ Block 0:                                   │               ║
║  │  ┌─ Weight Buffer  (32 KB) ████████████   │               ║
║  │  ┌─ Act Buffer     (16 KB) ██████░░░░░░   │               ║
║  │  └─ Output Buffer  (8 KB)  ████░░░░░░░░   │               ║
║  └────────────────────────────────────────────┘               ║
║                      ▼ Double buffering active                 ║
║                                                                ║
║  Register File (per thread)                  [64 registers]   ║
║  ┌────────────────────────────────────────────┐               ║
║  │ Thread 0: [r0-r15]: Weight accumulators   │               ║
║  │          [r16-r31]: Activation data        │               ║
║  │          [r32-r47]: Intermediate results   │               ║
║  │          [r48-r63]: Loop counters          │               ║
║  └────────────────────────────────────────────┘               ║
║                                                                ║
╚═══════════════════════════════════════════════════════════════╝
```

## 4. Thread-Level Execution View

```
Block [0,0] - Warp 0 (Threads 0-31)
═══════════════════════════════════════════════════════════

Time  │ Activity                          │ Stall Reason
──────┼───────────────────────────────────┼─────────────────
0μs   │ ████ Load weights (async)        │ -
2μs   │ ████ Compute FP16xINT4 matmul    │ -
4μs   │ ░░░░ Wait for data                │ Memory latency
6μs   │ ████ Dequantize INT4→FP16        │ -
8μs   │ ████ Accumulate results          │ -
10μs  │ ████ Write to global memory      │ -

Occupancy: ████████████████░░░░░░░░░░░░░░░░ 52%
           ^Active          ^Stalled
```

## 5. Memory Transaction Details

```
Block 0 Memory Transactions (First 32 cycles)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cycle  Operation         Source → Dest           Size    Coalesced
──────────────────────────────────────────────────────────────────
   1   Load Weight       Global → Shared         128B    ✓ Yes
   2   Load Activation   Global → Shared         256B    ✓ Yes
   5   Dequantize        Shared → Register       64B     -
   7   FP16×INT4 Compute Registers → Registers   -       -
  10   Store Result      Register → Shared       128B    -
  12   Write Back        Shared → Global         256B    ✓ Yes

Total Transactions:  342
Coalesced:          298 (87.1%)
Wasted Bandwidth:   12.9%
```

## 6. Data Packing Visualization (Marlin Format)

```
INT4 Weight Packing in Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Original FP16 weights: [w0, w1, w2, w3, w4, w5, w6, w7]
                         ▼  Quantize to INT4
Packed INT4 format (8 weights → 32 bits):

Byte 0: [w1:4bit][w0:4bit]    0x3A
Byte 1: [w3:4bit][w2:4bit]    0x7C
Byte 2: [w5:4bit][w4:4bit]    0x1F
Byte 3: [w7:4bit][w6:4bit]    0x8B
        └────────┴────────┘
        32-bit word = 0x8B1F7C3A

Reordering for Marlin kernel (optimized memory access):
┌─────────┬─────────┬─────────┬─────────┐
│ Tile 0  │ Tile 1  │ Tile 2  │ Tile 3  │
│ [w0,w4] │ [w1,w5] │ [w2,w6] │ [w3,w7] │
└─────────┴─────────┴─────────┴─────────┘
   ▲         ▲         ▲         ▲
   └─────────┴─────────┴─────────┴─── Loaded by different warps
```

## 7. Tensor Core Activity (Ampere/Hopper)

```
Tensor Core Utilization (per SM)
═════════════════════════════════════════════

SM 0:  TC0 ████████████████ 95%  ← Very high utilization
       TC1 ███████████████░ 91%
       TC2 ████████████░░░░ 78%
       TC3 ███████████░░░░░ 72%

Operations: 8192 FP16×INT4 MMA instructions
Throughput: 312 TFLOPS (theoretical: 330 TFLOPS)
Efficiency: 94.5%
```

## 8. Kernel Execution Timeline

```
Marlin Kernel Execution Timeline (Block 0)
═══════════════════════════════════════════════════════════════

Phase               Duration    │ Visualization
────────────────────────────────┼──────────────────────────────
Setup & prefetch    2.1 μs      │ ██░░░░░░░░░░░░░░░░░░░░░░░░░░
Main loop           45.3 μs     │ ░░████████████████████████░░
  ├─ Load weights   12.1 μs     │   ███░░░░░░░░░░░░░░░░░░░░░░░
  ├─ Dequantize     8.7 μs      │   ░░░██░░░░░░░░░░░░░░░░░░░░░
  ├─ Compute        18.2 μs     │   ░░░░░█████████░░░░░░░░░░░░
  └─ Accumulate     6.3 μs      │   ░░░░░░░░░░░░░██░░░░░░░░░░░
Write back          1.8 μs      │ ░░░░░░░░░░░░░░░░░░░░░░░░░░██
────────────────────────────────┼──────────────────────────────
Total               49.2 μs     │ ████████████████████████████

Pipeline stages overlapped: 68% (good!)
```

## 9. Real-time GPU Stats

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ NVIDIA A100 - Real-time Monitoring                 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                     ┃
┃ GPU Utilization:  ████████████████████░░░░░░ 87%   ┃
┃ Memory Usage:     ███████████░░░░░░░░░░░░░░ 52%   ┃
┃ Temperature:      65°C  (Safe)                     ┃
┃ Power:            245W / 400W                      ┃
┃                                                     ┃
┃ SM Activity:                                        ┃
┃   Active SMs:     104 / 108                        ┃
┃   Avg Occupancy:  68.2%                            ┃
┃                                                     ┃
┃ Memory Bandwidth:                                   ┃
┃   Current:        875 GB/s                         ┃
┃   Peak (HBM2):    1555 GB/s                        ┃
┃   Utilization:    ████████████████░░░░░░░░░░ 56%   ┃
┃                                                     ┃
┃ Kernel Stats:                                       ┃
┃   Active Kernels: 1 (marlin_kernel_fp16_int4)      ┃
┃   Launched:       512 blocks                       ┃
┃   Completed:      347 blocks (67.8%)               ┃
┃                                                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## 10. Profiling Summary

```
═══════════════════════════════════════════════════════════════
                   MARLIN KERNEL PROFILING REPORT
═══════════════════════════════════════════════════════════════

Kernel Name:       marlin_fp16_int4_matmul
Grid Size:         (128, 4, 1)
Block Size:        (256, 1, 1)
Total Duration:    3.42 ms

PERFORMANCE METRICS
───────────────────────────────────────────────────────────────
Throughput:                2847 tokens/second
FLOPS:                     187.3 TFLOPS
Memory Bandwidth:          875 GB/s (56% of peak)
Kernel Efficiency:         72.4%

BOTTLENECK ANALYSIS
───────────────────────────────────────────────────────────────
Primary Bottleneck:        Memory Bandwidth ⚠
Compute Utilization:       94%  ✓ Excellent
Memory Utilization:        56%  △ Moderate
Occupancy:                 68%  ✓ Good

RECOMMENDATIONS
───────────────────────────────────────────────────────────────
✓ Tensor Cores well utilized
⚠ Consider increasing data reuse in shared memory
✓ Coalesced memory access pattern good (87%)
✓ No significant warp divergence detected

═══════════════════════════════════════════════════════════════
```

---

## Các Tool Được Sử Dụng

1. **NSight Compute**: Kernel-level profiling
2. **NSight Systems**: Timeline visualization
3. **PyTorch Profiler**: Integration với training loop
4. **Custom CUDA printf**: In-kernel debugging
5. **py3nvml**: Real-time GPU monitoring
6. **Rich library**: Terminal visualization

Tất cả visualizations trên sẽ được generate tự động khi bạn chạy `gpu_introspection.py`!
