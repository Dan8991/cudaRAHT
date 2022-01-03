from numba import cuda, vectorize
from time import time
import numpy as np
from raht import raht, flatten_cubes, unflatten_cubes, parallelized_raht
from PCutils import read_ply_files, voxelize_PC
import matplotlib.pyplot as plt

if __name__ == "__main__":
    a = np.random.rand(int(1e8)).astype(np.float64)
    grid_size = 1024

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
    data = voxelize_PC(data, n_voxels=grid_size).astype(np.uint8)
    res = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.uint8)

    res[tuple(data[:, :3].T)] = np.concatenate([np.ones((len(data), 1)), data[:, 3:]], axis = 1)
    now = time()
    weight, lf, hf = parallelized_raht(res)
    print("time elapsed:", time() - now)
    print(np.sum(res[..., 0]), weight, lf, hf.shape)
    now = time()
    weight, lf, hf = raht(res)
    print("time elapsed:", time() - now)
    hf = np.concatenate(hf, axis=0).reshape(-1, 3)
    print(np.sum(res[..., 0]), weight, lf, hf.shape)
    now = time()
    weight, lf, hf = parallelized_raht(
        res,
        cuda=True,
        threadsperblock=threadsperblock,
        blockspergrid=blockspergrid
    )
    print("time elapsed:", time() - now)
    print(np.sum(res[..., 0]), weight, lf, hf.shape)


