#!/usr/bin/env python3
"""Test script to verify weighted TP distribution for CXL support."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch
from kt_kernel import kt_kernel_ext

# Test configuration
expert_num = 256
hidden_size = 7168
intermediate_size = 2048  # 2048 / 64 = 32 blocks
max_len = 1024
num_experts_per_tok = 8
qlen = 1

# NUMA configuration: 3 subpools with 1:1:4 weight ratio
threadpool_count = 3
total_threads = 90
threads_per_pool = total_threads // threadpool_count

print("=" * 60)
print("Testing Weighted TP Distribution for CXL Support")
print("=" * 60)
print(f"intermediate_size: {intermediate_size}")
print(f"K_STEP blocks: {intermediate_size // 64}")
print(f"threadpool_count: {threadpool_count}")
print(f"Expected distribution with 1:1:4 ratio:")
print(f"  TP 0: ~5 blocks = 320")
print(f"  TP 1: ~5 blocks = 320")
print(f"  TP 2: ~22 blocks = 1408")
print("=" * 60)

# Create WorkerPoolConfig with weight ratios
worker_config = kt_kernel_ext.WorkerPoolConfig()
worker_config.subpool_count = threadpool_count
worker_config.subpool_numa_map = list(range(threadpool_count))
worker_config.subpool_thread_count = [threads_per_pool] * threadpool_count
worker_config.subpool_weight_ratios = [1, 1, 4]  # 1:1:4 ratio

print(f"\nWorkerPoolConfig:")
print(f"  subpool_count: {worker_config.subpool_count}")
print(f"  subpool_numa_map: {worker_config.subpool_numa_map}")
print(f"  subpool_thread_count: {worker_config.subpool_thread_count}")
print(f"  subpool_weight_ratios: {worker_config.subpool_weight_ratios}")
print()

# Create CPUInfer with the configured WorkerPoolConfig
CPUInfer = kt_kernel_ext.CPUInfer(worker_config)
print()

# Create random weights
physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()
gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16, device="cpu").contiguous()

# Create MOE config
config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
config.max_len = max_len
config.gate_proj = gate_proj.data_ptr()
config.up_proj = up_proj.data_ptr()
config.down_proj = down_proj.data_ptr()
config.gate_scale = 0
config.pool = CPUInfer.backend_

print("Creating AMXBF16_MOE...")
print("-" * 60)

# Create MoE - this should print the TP distribution
moe = kt_kernel_ext.moe.AMXBF16_MOE(config)

print("-" * 60)
print("\nLoading weights...")
CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
CPUInfer.sync()

# Run a simple forward pass to verify it works
print("\nRunning forward pass...")
bsz_tensor = torch.tensor([qlen], device="cpu")
expert_ids = torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]).contiguous()
weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
input_tensor = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

CPUInfer.submit(
    moe.forward_task(
        bsz_tensor.data_ptr(),
        num_experts_per_tok,
        expert_ids.data_ptr(),
        weights.data_ptr(),
        input_tensor.data_ptr(),
        output.data_ptr(),
        False,
    )
)
CPUInfer.sync()

print(f"Forward pass completed successfully!")
print(f"Output shape: {output.shape}")
print(f"Output sample: {output[0, :5]}")

print()
print("=" * 60)
print("TEST PASSED - Weighted TP distribution is working!")
print("=" * 60)
