import numpy as np      # https://numba.readthedocs.io/en/stable/cuda/examples.html
from numba import cuda
from tqdm import tqdm
import time

@cuda.jit
def f(a, b, c):
    # like threadIdx.x + (blockIdx.x * blockDim.x)
    tid = cuda.grid(1)
    # if tid % 100 == 0:
    #    print('tid=', tid)
    size = len(c)
    if tid < size:
        c[tid] = a[tid] + b[tid]

N = 10000000
a = cuda.to_device(np.random.random(N))
b = cuda.to_device(np.random.random(N))
c = cuda.device_array_like(a)

start_time_cpu = time.time()
f.forall(len(a))(a, b, c)
end_time_cpu = time.time()
cpu_execution_time = end_time_cpu - start_time_cpu
print(f'CPU len={len(a)}, ', c.copy_to_host())

# input()

start_time_cuda = time.time()
nthreads = 256  # Enough threads per block for several warps per block
nblocks = (len(a) // nthreads) + 1  # Enough blocks to cover the entire vector depending on its length
f[nblocks, nthreads](a, b, c)
end_time_cuda = time.time()
cuda_execution_time = end_time_cuda - start_time_cuda

print(f'CUDA len={len(a)}, ', c.copy_to_host())
print(f"CUDA threads: {nthreads}, blocks: {nblocks}")

print(f"CPU Execution Time: {cpu_execution_time:.6f} seconds")
print(f"CUDA Execution Time: {cuda_execution_time:.6f} seconds")