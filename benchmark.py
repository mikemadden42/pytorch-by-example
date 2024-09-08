#!/usr/bin/env python3

import time

import torch


# Function to perform the benchmark
def benchmark(device, size=5000, warmup_iterations=5):
    # Generate random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up iterations
    for _ in range(warmup_iterations):
        _ = torch.matmul(a, b)

    # Perform matrix multiplication and measure time
    start_time = time.time()
    _ = torch.matmul(a, b)
    elapsed_time = time.time() - start_time

    return elapsed_time


# Matrix size (adjustable)
matrix_size = 5000  # Adjust the size for significant benchmarks

# Number of warm-up iterations
warmup_iterations = 5

# Benchmark on CPU
cpu_time = benchmark(
    device="cpu", size=matrix_size, warmup_iterations=warmup_iterations
)
print(f"CPU time: {cpu_time:.6f} seconds")

# Check if CUDA is available and benchmark on GPU
if torch.cuda.is_available():
    gpu_time = benchmark(
        device="cuda", size=matrix_size, warmup_iterations=warmup_iterations
    )
    print(f"GPU (CUDA) time: {gpu_time:.6f} seconds")
else:
    print("CUDA is not available on this device.")

# Check if MPS is available and benchmark on MPS
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    mps_time = benchmark(
        device="mps", size=matrix_size, warmup_iterations=warmup_iterations
    )
    print(f"GPU (MPS) time: {mps_time:.6f} seconds")
else:
    print("MPS is not available on this device.")
