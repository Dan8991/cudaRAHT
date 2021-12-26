from numba import cuda, vectorize
from time import time
import numpy as np

if __name__ == "__main__":
    a = np.random.rand(int(1e8)).astype(np.float64)

    gpu = cuda.get_current_device()
    max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
    max_blocks_per_grid = gpu.MAX_GRID_DIM_X
    print("Max threads per block:", max_threads_per_block)
    print("Max blocks per grid:", max_blocks_per_grid)
    threadsperblock = 1024
    blockspergrid = min((a.size + (threadsperblock - 1)) // threadsperblock, max_blocks_per_grid)
    print("threads per block:", threadsperblock)
    print("blocks per grid:", blockspergrid)



