#!/usr/bin/env python3

import sys

import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
if not cuda_available:
    sys.exit("CUDA not available")

# Get the number of GPUs
device_count = torch.cuda.device_count()
print(f"Number of GPUs: {device_count}")

# List GPU names and detailed properties
for i in range(device_count):
    device_name = torch.cuda.get_device_name(i)
    device_properties = torch.cuda.get_device_properties(i)

    print(f"\nDevice {i}: {device_name}")
    print(f"  - Name: {device_properties.name}")
    print(f"  - Major: {device_properties.major}")
    print(f"  - Minor: {device_properties.minor}")
    print(f"  - Total Memory: {device_properties.total_memory / 1024 ** 3:.2f} GB")
    print(f"  - Multi Processor Count: {device_properties.multi_processor_count}")
    print(
        f"  - CUDA Capability (Major.Minor): {device_properties.major}.{device_properties.minor}"
    )

    if hasattr(device_properties, "max_threads_per_block"):
        print(f"  - Max Threads per Block: {device_properties.max_threads_per_block}")
    if hasattr(device_properties, "max_threads_per_multiprocessor"):
        print(
            f"  - Max Threads per Multi Processor: {device_properties.max_threads_per_multiprocessor}"
        )
    if hasattr(device_properties, "max_grid_size"):
        print(f"  - Max Grid Size: {device_properties.max_grid_size}")
    if hasattr(device_properties, "max_block_dim"):
        print(f"  - Max Block Dimensions: {device_properties.max_block_dim}")
    if hasattr(device_properties, "warp_size"):
        print(f"  - Warp Size: {device_properties.warp_size}")
    if hasattr(device_properties, "memory_clock_rate"):
        print(
            f"  - Memory Clock Rate: {device_properties.memory_clock_rate / 1e3:.2f} MHz"
        )
    if hasattr(device_properties, "memory_bus_width"):
        print(f"  - Memory Bus Width: {device_properties.memory_bus_width} bits")
    if hasattr(device_properties, "l2_cache_size"):
        print(f"  - L2 Cache Size: {device_properties.l2_cache_size / 1024:.2f} KB")
    if hasattr(device_properties, "shared_memory_per_block"):
        print(
            f"  - Shared Memory per Block: {device_properties.shared_memory_per_block / 1024:.2f} KB"
        )
    if hasattr(device_properties, "shared_memory_per_multiprocessor"):
        print(
            f"  - Shared Memory per Multi Processor: {device_properties.shared_memory_per_multiprocessor / 1024:.2f} KB"
        )
    if hasattr(device_properties, "regs_per_block"):
        print(f"  - Registers per Block: {device_properties.regs_per_block}")
    if hasattr(device_properties, "regs_per_multiprocessor"):
        print(
            f"  - Registers per Multi Processor: {device_properties.regs_per_multiprocessor}"
        )

# Get the current GPU
if cuda_available:
    current_device = torch.cuda.current_device()
    print(f"\nCurrent GPU Device: {current_device}")
    print(f"Current GPU Device Name: {torch.cuda.get_device_name(current_device)}")
