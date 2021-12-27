from numba import cuda, vectorize
from time import time
import numpy as np
from raht import raht
from PCutils import read_ply_files, voxelize_PC
import matplotlib.pyplot as plt

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

    data = read_ply_files("../dataset/long.ply", only_geom=False)
    data = voxelize_PC(data).astype(np.uint8)
    res = np.zeros((1024, 1024, 1024, 4), dtype=np.uint8)

    res[tuple(data[:, :3].T)] = np.concatenate([np.ones((len(data), 1)), data[:, 3:]], axis = 1)
    now = time()
    weight, lf, hf = raht(res)
    print(time() - now)
    hf = np.array(hf).reshape(-1)
    plt.hist(hf, bins=100)
    plt.yscale("log")
    plt.show()



