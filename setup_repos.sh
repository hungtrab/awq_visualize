#!/bin/bash
# Setup script for AWQ + Marlin Kernel Demo
# This script clones necessary repositories and sets up the environment

set -e  # Exit on error

echo "=================================================="
echo "  AWQ + Marlin Kernel Demo Setup"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: Git is not installed. Please install git first.${NC}"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}Warning: nvcc not found. CUDA toolkit may not be installed.${NC}"
    echo -e "${YELLOW}Marlin kernel requires CUDA. Install CUDA Toolkit 11.8+ for full functionality.${NC}"
fi

# Create external directory for repositories
EXTERNAL_DIR="external"
mkdir -p "$EXTERNAL_DIR"
cd "$EXTERNAL_DIR"

echo ""
echo -e "${GREEN}[1/3] Cloning Marlin Kernel Repository...${NC}"
if [ -d "marlin" ]; then
    echo "  ↳ Marlin already exists, pulling latest changes..."
    cd marlin && git pull && cd ..
else
    git clone https://github.com/IST-DASLab/marlin.git
    echo -e "${GREEN}  ✓ Marlin cloned successfully${NC}"
fi

echo ""
echo -e "${GREEN}[2/3] Cloning AutoAWQ Repository (optional)...${NC}"
if [ -d "AutoAWQ" ]; then
    echo "  ↳ AutoAWQ already exists, pulling latest changes..."
    cd AutoAWQ && git pull && cd ..
else
    git clone https://github.com/casper-hansen/AutoAWQ.git
    echo -e "${GREEN}  ✓ AutoAWQ cloned successfully${NC}"
fi

echo ""
echo -e "${GREEN}[3/3] Checking GPU information...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
    echo ""
    
    # Check compute capability
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
    MINOR=$(echo $COMPUTE_CAP | cut -d. -f2)
    
    if [ "$MAJOR" -ge 7 ] && [ "$MINOR" -ge 5 ]; then
        echo -e "${GREEN}  ✓ GPU compute capability $COMPUTE_CAP is compatible with Marlin${NC}"
    else
        echo -e "${YELLOW}  ⚠ GPU compute capability $COMPUTE_CAP may not be fully compatible${NC}"
        echo -e "${YELLOW}  ⚠ Marlin requires compute capability >= 7.5${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ nvidia-smi not found. Cannot detect GPU.${NC}"
fi

cd ..

echo ""
echo -e "${GREEN}=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Install Python dependencies: pip install -r requirements.txt"
echo "  2. Build Marlin kernel: cd external/marlin && python setup.py install"
echo "  3. Run demos: python demos/01_awq_quantization_demo.py"
echo ""
echo "Repositories cloned to: ${PWD}/${EXTERNAL_DIR}/"
echo ""
