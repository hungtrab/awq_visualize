# AWQ + Marlin Kernel Demo

🚀 **Comprehensive demonstrations** của cách AWQ quantization và Marlin kernel hoạt động cùng nhau để tăng tốc LLM inference.

## 📋 Tổng Quan

Project này cung cấp:
- ✅ **AWQ Quantization Demo**: Hiểu cách AWQ quantize weights từ FP16 → INT4
- ✅ **Marlin Kernel Demo**: Xem cách Marlin kernel thực hiện FP16×INT4 matmul
- ✅ **GPU Introspection**: "X-ray view" của GPU state trong khi chạy kernel
- ✅ **Performance Profiling**: So sánh hiệu năng across batch sizes
- ✅ **Visualization Tools**: Charts, plots, và diagrams

## 🎯 Quick Start

### 1. Setup

```bash
# Clone repositories (Marlin, AutoAWQ)
./setup_repos.sh

# Install dependencies
pip install -r requirements.txt

# (Optional) Build Marlin kernel
cd external/marlin
python setup.py install
cd ../..
```

### 2. Run Demos

```bash
# Demo 1: AWQ Quantization
python demos/01_awq_quantization_demo.py --model facebook/opt-125m

# Demo 2: Marlin Kernel Basics  
python demos/02_marlin_kernel_basic.py --visualize-packing --verify

# Demo 3: GPU Introspection
python demos/05_gpu_introspection.py --snapshot --visualize --benchmark

# Demo 4: Performance Profiling
python demos/04_performance_profiler.py --batch-sizes 1 2 4 8 16 32
```

## 📚 Demos Chi Tiết

### 01 - AWQ Quantization Demo

Demonstrates AWQ quantization process với visualizations:

```bash
python demos/01_awq_quantization_demo.py \
    --model facebook/opt-125m \
    --quick-test
```

**Output:**
- Weight distribution plots (before/after quantization)
- Quantization error analysis
- Memory reduction metrics
- Inference test

**Học được gì:**
- AWQ bảo vệ important weights với scaling factors
- 4× memory reduction với <1% accuracy loss
- Activation-aware quantization vs naive quantization

---

### 02 - Marlin Kernel Basics

Xem cách Marlin kernel pack/unpack INT4 weights:

```bash
python demos/02_marlin_kernel_basic.py \
    --m 128 --n 512 --k 512 \
    --visualize-packing \
    --verify
```

**Output:**
- INT4 packing visualization
- FP16×FP16 vs FP16×INT4 comparison
- Memory layout diagrams
- Performance benchmarks

**Học được gì:**
- INT4 weight packing format
- On-the-fly dequantization
- Memory bandwidth savings

---

### 05 - GPU Introspection

Deep dive vào GPU state:

```bash
# Quick snapshot
python demos/05_gpu_introspection.py --snapshot --visualize

# With benchmarking
python demos/05_gpu_introspection.py --benchmark

# Real-time monitoring (10 seconds)
python demos/05_gpu_introspection.py --monitor 10

# Show kernel config
python demos/05_gpu_introspection.py --kernel-config "128,4,1;256,1,1"
```

**Output:**
- GPU info (compute capability, memory, SMs)
- Memory hierarchy visualization
- Kernel launch configuration
- Real-time GPU metrics (utilization, temperature, power)
- Matrix multiplication benchmarks

**Học được gì:**
- GPU memory hierarchy (Global → L2 → L1/Shared → Registers)
- Block/thread organization
- Memory bandwidth utilization
- SM occupancy

---

### 04 - Performance Profiler

Comprehensive performance analysis:

```bash
python demos/04_performance_profiler.py \
    --model-dim 4096 \
    --batch-sizes 1 2 4 8 16 32 64 \
    --output-dir ./results
```

**Output:**
- `performance_results.csv`: All benchmark data
- `throughput_scaling.png`: Throughput vs batch size
- `latency_comparison.png`: FP16 vs INT4 latency
- Terminal summary table

**Metrics:**
- Latency (ms)
- Throughput (tokens/s)
- TFLOPS
- Speedup
- Memory saved

---

## 🖼️ Visualizations

Example outputs được tạo:

| File | Description |
|------|-------------|
| `awq_quantization_*.png` | Weight distributions, errors |
| `throughput_scaling.png` | Performance vs batch size |
| `latency_comparison.png` | FP16 vs INT4 comparison |
| `performance_results.csv` | Raw benchmark data |

## 🔧 Utilities

### Visualization (`utils/visualization.py`)

```python
from utils.visualization import VisualizationHelper

viz = VisualizationHelper(output_dir="./results")

# Plot weight distributions
viz.plot_weight_distribution({
    'original': weights_fp16,
    'quantized': weights_int4
}, filename='weights.png')

# Compare performance
viz.plot_performance_comparison(results, metric='latency_ms')
```

### Profiling (`utils/profiling.py`)

```python
from utils.profiling import CUDATimer, Benchmarker, profile_scope

# Quick timing
with profile_scope("my_operation"):
    result = my_function()

# Detailed benchmarking
benchmarker = Benchmarker(num_iterations=100)
result = benchmarker.benchmark_function(my_function, "test")
benchmarker.print_results()
```

## 📊 Expected Performance

Trên NVIDIA A100 GPU:

| Method | Latency | Throughput | Memory | Speedup |
|--------|---------|------------|--------|---------|
| FP16 Baseline | 10.5 ms | 95 tok/s | 16 GB | 1.0× |
| AWQ + Marlin | 3.7 ms | 270 tok/s | 4 GB | 2.8× |

**Note:** Actual Marlin kernel có thể faster hơn simulation trong demo.

## 🎓 Educational Materials

### Example GPU State Visualization

See [`GPU_INTROSPECTION_EXAMPLE.md`](GPU_INTROSPECTION_EXAMPLE.md) để xem detailed examples của:
- Kernel configuration visualization
- Memory hierarchy snapshots
- Thread block layouts
- Real-time monitoring displays

### Key Concepts

**AWQ Quantization:**
- Activation-aware scaling
- Outlier preservation
- Per-channel quantization
- Minimal accuracy loss

**Marlin Kernel:**
- FP16×INT4 mixed-precision
- Asynchronous memory operations
- Pipelining & double buffering
- Shared memory optimization
- Tensor Core utilization

## 💻 Requirements

### Hardware
- NVIDIA GPU với CUDA support
- Compute Capability ≥ 7.5 (recommended cho Marlin)
- Minimum 8GB VRAM

### Software
- Python 3.8+
- PyTorch 2.0+ với CUDA
- CUDA Toolkit 11.8+ (cho Marlin kernel compilation)

### Python Packages
See [`requirements.txt`](requirements.txt)

## 🐛 Troubleshooting

**CUDA not available:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Import errors:**
```bash
# Ensure utils are in path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Marlin kernel compilation issues:**
```bash
# Check compute capability
python -c "import torch; print(torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor)"

# May need newer nvcc
# Follow instructions at: https://github.com/IST-DASLab/marlin
```

## 📖 Further Reading

- **AWQ Paper**: [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- **Marlin Paper**: [Marlin: Fast INT4 Inference](https://arxiv.org/abs/2408.11743)
- **Marlin GitHub**: https://github.com/IST-DASLab/marlin
- **AutoAWQ GitHub**: https://github.com/casper-hansen/AutoAWQ

## 🤝 Contributing

Suggestions và improvements welcome! This is an educational demo project.

## 📄 License

This demo project is for educational purposes. Please check licenses of:
- Marlin kernel: IST-DASLab
- AutoAWQ: casper-hansen
- Individual model licenses from HuggingFace

## 🙏 Acknowledgments

- **IST-DASLab** for Marlin kernel
- **MIT Han Lab** for AWQ algorithm
- **casper-hansen** for AutoAWQ implementation
- **HuggingFace** for model hosting

---

**Author**: Demo created for understanding AWQ + Marlin internals  
**Last Updated**: January 2026

Happy learning! 🎉
