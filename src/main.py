from numba import cuda, vectorize
from time import time
import numpy as np
from raht import raht, flatten_cubes, unflatten_cubes, parallelized_raht, full_cuda_raht
from PCutils import read_ply_files, voxelize_PC
import matplotlib.pyplot as plt
import pandas as pd
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop_level",
        type=int,
        default=3,
        help="number of levels of detail processed using gpu"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="resolution of the grid"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="fullcuda",
        help="fullcuda=all execution is carried out in cuda for the first stop_level levels \n"
        "numpy=full numpy execution\n"
        "sequential=the whole computation is carried out sequentially"
        "partialsequential=most of the computation is carried out sequentially except for the operation to check if a region is empty"
    )
    parser.add_argument(
        "--shared_memory",
        action="store_true",
        help="If true shared memory is used in gpu block, otherwise shared memory isn't used"
    )

    FLAGS = parser.parse_args()
    stop_level = FLAGS.stop_level
    grid_size = FLAGS.resolution
    execution_type = FLAGS.type
    shared_memory = FLAGS.shared_memory

    gpu = cuda.get_current_device()
    max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
    max_blocks_per_grid = gpu.MAX_GRID_DIM_X
    print("Max threads per block:", max_threads_per_block)
    print("Max blocks per grid:", max_blocks_per_grid)
    print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
    print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
    print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
    print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
    print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
    print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
    print("maxSharedMemoryPerBlock = %s bytes" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK / 4))

    # reading the pc
    data = read_ply_files("../dataset/long.ply", only_geom=False)
    # quantizing the pc to obtain the correct resolution
    data = voxelize_PC(data, n_voxels=grid_size)
    #removing duplicate points by averaging the color
    df = pd.DataFrame(data, columns=["x", "y", "z", "r", "g", "b"])
    data = df.groupby(["x", "y", "z"], as_index=False).mean().to_numpy().astype(np.int32)

    #executes the sequential code
    if execution_type == "sequential" or execution_type == "partialsequential":
        weight, lf, hf = raht(
            data,
            grid_size,
            slightly_parallelized=execution_type == "partialsequential"
        )
        hf = np.concatenate(hf, axis=0).reshape(-1, 3)
        print(len(data), weight, lf, hf.shape)
    #executes the code optimized with numpy
    elif execution_type == "numpy":
        weight, lf, hf = parallelized_raht(
            data,
            grid_size = grid_size
        )
        print(len(data), weight, lf, hf.shape)
    #executes the code in cuda
    elif execution_type == "fullcuda":
        weight, lf, hf = full_cuda_raht(
            data,
            (grid_size, grid_size, grid_size, 4),
            max_num_iter=stop_level,
            shared_memory=shared_memory
        )
        print(len(data), weight, lf, hf.shape)

