# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Quick installation with auto-detection (recommended)
./install.sh

# Two-step installation
./install.sh deps     # Install dependencies only
./install.sh build    # Build and install

# Development installation
pip install -e .

# Custom build with environment variables
export CPUINFER_CPU_INSTRUCT=AVX512    # NATIVE|FANCY|AVX512|AVX2
export CPUINFER_ENABLE_AMX=ON          # Enable AMX
export CPUINFER_BUILD_TYPE=Release     # Release|Debug|RelWithDebInfo
./install.sh build --manual
```

## Test Commands

```bash
# Run test suite
pytest test/per_commit/
pytest test/per_commit/ -m cpu         # CPU backend tests only
pytest test/per_commit/ -m slow        # Run slow tests

# Run custom test suite
python test/run_suite.py --hw cpu --suite default
```

## Lint/Format Commands

```bash
# Format C++ code (requires clang-format 18+)
cmake -B build && cd build && make format

# Format Python code
black .
```

## Architecture Overview

### Core Components

- **`/operators/`** - Kernel implementations
  - `amx/` - Intel AMX backend (moe.hpp, k2-moe.hpp, awq-moe.hpp)
  - `moe-tp.hpp` - Tensor Parallel MoE wrapper that distributes work across NUMA nodes
  - `llamafile/` - Universal CPU backend
  - `moe_kernel/` - AMD BLIS and ARM KML implementations
  - `common.hpp` - Shared types (GeneralMOEConfig, WorkerPoolConfig, etc.)

- **`/cpu_backend/`** - CPU execution infrastructure
  - `worker_pool.h/cpp` - NUMA-aware thread pool with subpool configuration
  - `cpuinfer.h` - CPUInfer class for task submission and synchronization

- **`/python/`** - Python package
  - `experts.py` - KTMoEWrapper public API
  - `experts_base.py` - BaseMoEWrapper base class with CPUInfer singleton management
  - `utils/amx.py` - AMXMoEWrapper and RAWAMXMoEWrapper implementations

- **`/ext_bindings.cpp`** - pybind11 bindings exposing C++ classes to Python

### Key Patterns

**MoE TP Distribution**: The `TP_MOE` class in `moe-tp.hpp` distributes intermediate_size across NUMA nodes. Each TP instance handles a slice of gate/up projections (row-major) and down projection (column-major).

**Worker Pool Configuration**: `WorkerPoolConfig` in `worker_pool.h` configures:
- `subpool_count` - Number of NUMA subpools
- `subpool_numa_map` - NUMA node ID for each subpool
- `subpool_thread_count` - Threads per subpool
- `subpool_weight_ratios` - Weight ratios for TP distribution (e.g., [1, 1, 4])

**AMX Alignment Requirements**: AMX kernels require:
- K_STEP = 64 alignment for input dimension (hidden_size)
- N_STEP = 32 alignment for output dimension (intermediate_size per TP)

## Recent Modifications: Uneven TP Distribution for CXL Support

Added support for heterogeneous NUMA configurations where nodes have different memory capacities (e.g., 2 physical NUMA + 1 CXL memory expansion node).

### Problem
When `intermediate_size` is not evenly divisible by `tp_count` (NUMA nodes), the original code threw:
```
RuntimeError: For TP, intermediate_size must be a multiple of NUMA node count
```

### Solution
Modified `moe-tp.hpp` to support:
1. **Uneven distribution**: Distributes K_STEP-aligned blocks across TPs with remainder handling
2. **Weighted distribution**: Uses `subpool_weight_ratios` to control how much weight each NUMA node stores

### Files Modified

1. **`operators/moe-tp.hpp`**
   - Added `tp_intermediate_offsets` vector to track byte offsets for each TP
   - Implemented K_STEP (64) aligned block distribution
   - Added weighted distribution when `subpool_weight_ratios` is configured

2. **`operators/amx/k2-moe.hpp`** and **`operators/amx/moe.hpp`**
   - Updated weight loading to use `tp_intermediate_offsets` for proper slicing
   - Fixed memcpy operations for both per-expert pointers and contiguous memory paths

3. **`cpu_backend/worker_pool.h`**
   - Added `subpool_weight_ratios` field to `WorkerPoolConfig`

4. **`python/experts_base.py`**
   - Added `set_subpool_weight_ratios()` class method
   - Added environment variable support: `KT_SUBPOOL_WEIGHT_RATIOS=1:1:4`

### Usage

```python
# Option 1: Environment variable (before running)
# export KT_SUBPOOL_WEIGHT_RATIOS=1:1:4

# Option 2: Python API (before creating MoE layers)
from kt_kernel.python.experts_base import BaseMoEWrapper
BaseMoEWrapper.set_subpool_weight_ratios([1, 1, 4])
```

With `intermediate_size=2048` and ratio `1:1:4`:
- 32 K_STEP blocks (2048 / 64)
- TP 0: 5 blocks = 320
- TP 1: 5 blocks = 320
- TP 2: 22 blocks = 1408

### Test Script

```bash
python examples/test_weighted_tp.py
```

Expected output shows weighted distribution:
```
AMX TP splitting (weighted): intermediate_size=2048, tp_count=3, K_STEP=64
  TP 0: intermediate_size=320, offset=0, blocks=5
  TP 1: intermediate_size=320, offset=320, blocks=5
  TP 2: intermediate_size=1408, offset=640, blocks=22
```
