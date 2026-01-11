# 🚀 Quick Start Guide

## Để bắt đầu ngay:

```bash
cd /home/hungchan/Work/quantize_engineering

# 1. Git clone repositories cần thiết
./setup_repos.sh

# 2. Install dependencies (optional - có thể skip nếu chưa có GPU)
pip install -r requirements.txt

# 3. Chạy demos!
```

## Demo Commands:

### 1. Xem INT4 Packing (không cần GPU, chạy ngay!)
```bash
python demos/02_marlin_kernel_basic.py --visualize-packing
```

### 2. AWQ Quantization Demo (cần download model nhỏ)
```bash
python demos/01_awq_quantization_demo.py --model facebook/opt-125m
```

### 3. GPU Introspection (cần CUDA)
```bash
# Xem thông tin GPU
python demos/05_gpu_introspection.py --snapshot

# Với visualization
python demos/05_gpu_introspection.py --snapshot --visualize
```

### 4. Performance Profiling
```bash
python demos/04_performance_profiler.py --batch-sizes 1 2 4 8
```

## File Quan Trọng:

- **README.md**: Full documentation
- **GPU_INTROSPECTION_EXAMPLE.md**: Ví dụ output của GPU profiling
- **demos/**: 4 demo scripts
- **utils/**: Profiling và visualization tools

Enjoy! 🎉
