import numpy as np
from numba import cuda
from tqdm import tqdm
import time

# CUDA kernel
@cuda.jit
def add_kernel(a, b, result):
    i = cuda.grid(1)
    if i < a.shape[0]:
        # Accessing blockIdx.x, blockDim.x, and threadIdx.x
        block_idx = cuda.blockIdx.x
        block_dim = cuda.blockDim.x
        thread_idx = cuda.threadIdx.x
        global_idx = block_idx * block_dim + thread_idx

        result[i] = a[i] + b[i] + global_idx  # Add global index for demonstration

def cpu_vector_addition(a, b):
    return a + b

def main():
    # Set the size of the arrays
    array_size = 100000000  # Increased the array size for a longer calculation

    # Host (CPU) arrays
    a_host = np.random.random(array_size)
    b_host = np.random.random(array_size)

    # Measure CPU performance
    start_time_cpu = time.time()
    result_cpu = cpu_vector_addition(a_host, b_host)
    end_time_cpu = time.time()
    cpu_execution_time = end_time_cpu - start_time_cpu

    # Device (GPU) arrays
    a_device = cuda.to_device(a_host)
    b_device = cuda.to_device(b_host)
    result_device = cuda.device_array_like(a_device)

    # Set up the grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (array_size + threads_per_block - 1) // threads_per_block

    # Print CUDA grid and block information
    print(f"GridDim: {blocks_per_grid}, BlockDim: {threads_per_block}")

    # Measure CUDA performance
    start_time_cuda = time.time()
    with tqdm(total=blocks_per_grid) as progress_bar:
        add_kernel[blocks_per_grid, threads_per_block](a_device, b_device, result_device)
        progress_bar.update(blocks_per_grid)
    end_time_cuda = time.time()
    cuda_execution_time = end_time_cuda - start_time_cuda

    # Copy the result back to the host
    result_cpu_cuda = result_device.copy_to_host()

    # Verify the result
    expected_result = a_host + b_host + np.arange(array_size)
    assert np.allclose(result_cpu_cuda, expected_result)

    # Print performance comparison
    print(f"CPU Execution Time: {cpu_execution_time:.6f} seconds")
    print(f"CUDA Execution Time: {cuda_execution_time:.6f} seconds")

    # Calculate the performance difference
    performance_difference = cpu_execution_time / cuda_execution_time
    print(f"CUDA is {performance_difference:.2f} times faster than CPU.")

    # Print results for comparison
    print("\nResults for Comparison:")
    print("CPU Result:")
    print(result_cpu[:10])  # Print the first 10 elements
    print("\nCUDA Result:")
    print(result_cpu_cuda[:10])  # Print the first 10 elements

if __name__ == "__main__":
    main()
