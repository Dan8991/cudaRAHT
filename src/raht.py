import numpy as np
from numba import cuda, vectorize, float32

def flatten_cubes(vol, nb):
    '''
    flattens subcubes in cube of size (nbx, nby, nbz) i.e. values from the same
    subcube are disposed one after the other
    Parameters:
        vol (np.ndarray): volume to flatten of size (nvxx, nvxy, nvxz, c) where
                          c is the number of channels
        nb (int): subcube size along the various dimensions, if int then it is supposed cubic
    Return
        vol (np.ndarray): flattened volume of size 
                          (nvxx/nbx, nvxy/nby, nvxz/nbz, nbx * nby * nbz, c)
    '''
    nvxx, nvxy, nvxz = vol.shape[:3]

    if isinstance(nb, tuple):
        nbx, nby, nbz = nb
    else:
        nbx = nby = nbz = nb

    nblx = nvxx // nbx
    nbly = nvxy // nby
    nblz = nvxz // nbz
    # divides each of the dimensions in nbl parts of dimension nb
    vol = vol.reshape(nblx, nbx, nbly, nby, nblz, nbz, -1)
    # places all nearby the axis relative to the nbs and in the correct order
    # so reshaping actually yields the correct vector
    vol = vol.swapaxes(1, 4).swapaxes(3, 4).swapaxes(1, 2).reshape(
        nblx, nbly, nblz, nbx * nby * nbz, -1
    )
    return vol

# reshapes vol so it goes back to how it was before flatten_cubes


def unflatten_cubes(vol, nb):
    '''
    Inverse operation of flatten cubes
    Parameters:
        vol (np.ndarray): volume to unflatten of size 
                          (nblx, nbly, nblz, nbx * nby * nbz, c) where
                          c is the number of channels
        nb (int): subcube size along one dimension
    Return
        vol (np.ndarray): reshaped volume of size
                          (nvxx, nvxy, nvxz, c)
    '''

    nblx, nbly, nblz = vol.shape[:3]

    if isinstance(nb, tuple):
        nbx, nby, nbz = nb
    else:
        nbx = nby = nbz = nb

    nvxx = nblx * nbx
    nvxy = nbly * nby
    nvxz = nblz * nbz
    vol = vol.reshape(nblx, nbly, nblz, nbx, nby, nbz, -1)
    vol = vol.swapaxes(1, 2).swapaxes(3, 4).swapaxes(1, 4).reshape(
                nvxx, nvxy, nvxz, -1
    )
    return vol

def one_level_raht(block: np.ndarray, axis: int) -> np.ndarray:

    '''
    performs a slightly different raht transform
    Parameters:
        block: array of couples of lf coefficients and weights to be transformed
        axis: axis along which raht should be computed
    Returns:
        the transformed 2x2x2 block
    '''

    w = block[..., :1]
    lf = block[..., 1:]

    final = []
    w_sqrt = np.sqrt(w) 
    w_sqrt_inv = np.concatenate([
        - w_sqrt[(slice(None),) * (axis + 1) + (slice(1, 2),)],
        w_sqrt[(slice(None),) * (axis + 1) + (slice(0, 1),)],
    ], axis = axis + 1)
    w_coeffs = np.concatenate([w_sqrt, w_sqrt_inv], axis=-1)
    w_coeffs = np.swapaxes(w_coeffs, axis+1, -2).swapaxes(-1, -2)
    
    w_div = np.maximum(np.sum(w_sqrt, axis = axis+1, keepdims=True), 1)
    w = np.sum(w, axis = axis+1, keepdims=True)

    coeffs = w_coeffs @ np.swapaxes(lf, axis + 1, -2)
    coeffs = np.swapaxes(coeffs, axis+1, -2) / w_div
    hf = coeffs[(slice(None),)*(axis + 1) + (slice(1, 2),)]
    lf = coeffs[(slice(None),)*(axis + 1) + (slice(0, 1),)]

    return np.concatenate([w, lf], axis = -1).reshape(-1, block.shape[-1]), hf

@cuda.jit
def one_level_raht_gpu(block: np.ndarray, axis: int, lf: np.ndarray, hf: np.ndarray):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    if pos < block.shape[0]:
        w1 = block[pos, 0, 0]
        w2 = block[pos, 1, 0]
        sw1 = w1 ** 0.5
        sw2 = w2 ** 0.5
        for i in range(3):
            l1 = block[pos, 0, i + 1]
            l2 = block[pos, 1, i + 1]
            hf[pos, i] = (- sw2 * l1 + sw1 * l2)/(sw1 + sw2)
            lf[pos, i + 1] = (sw1 * l1 + sw2 * l2)/(sw1 + sw2)
        lf[pos, 0] = w1 + w2


@cuda.jit
def one_level_raht_full_gpu(
    block: np.ndarray,
    axis: int,
    axis_mult: list
):

    dx = axis_mult[0]
    dy = axis_mult[1]
    dz = axis_mult[2]

    tz = cuda.threadIdx.x * dz
    tx = cuda.blockIdx.x * dx
    ty = cuda.blockIdx.y * dy

    tx2 = tx
    ty2 = ty
    tz2 = tz

    if axis == 0:
        tx2 = tx2 + dx // 2
    elif axis == 1:
        ty2 = ty2 + dy // 2
    else:
        tz2 = tz2 + dz // 2

    w1 = block[tx, ty, tz, 0]
    w2 = block[tx2, ty2, tz2, 0]
    sw1 = w1 ** 0.5
    sw2 = w2 ** 0.5
    for i in range(3):
        l1 = block[tx, ty, tz, i + 1]
        l2 = block[tx2, ty2, tz2, i + 1]
        if (w1 == 0) or (w2 == 0):
            block[tx2, ty2, tz2, i + 1] = 0
            block[tx, ty, tz, i + 1] = l1 + l2
            block[tx2, ty2, tz2, 0] = 0
        else:
            #hf components
            block[tx2, ty2, tz2, i + 1] = (- sw2 * l1 + sw1 * l2)/(sw1 + sw2)
            #lf components
            block[tx, ty, tz, i + 1] = (sw1 * l1 + sw2 * l2)/(sw1 + sw2)
            block[tx2, ty2, tz2, 0] = -1
    block[tx, ty, tz, 0] = w1 + w2

@cuda.jit
def one_level_raht_full_gpu_shared_memory(
    block: np.ndarray,
    n_levels: int,
):

    shared_arr = cuda.shared.array(shape=(8, 8, 32, 4), dtype=float32)
    cuda.syncthreads()
    curr_level = 1
    dx = 2 
    dy = 1
    dz = 1

    threadIdx = cuda.threadIdx.x
    x_start = cuda.blockIdx.x * 8
    y_start = cuda.blockIdx.y * 8
    z_start = cuda.blockIdx.z * 32
    tz = threadIdx % 32
    ty = (threadIdx // 32) % 8
    tx = ((threadIdx // 256) % 4) * 2

    tx2 = tx + 1
    ty2 = ty
    tz2 = tz

    w1 = block[x_start + tx, y_start + ty, z_start + tz, 0]
    w2 = block[x_start + tx2, y_start + ty2, z_start + tz2, 0]
    sw1 = w1 ** 0.5
    sw2 = w2 ** 0.5
    for i in range(3):
        l1 = block[x_start + tx, y_start + ty, z_start + tz, i + 1]
        l2 = block[x_start + tx2, y_start + ty2, z_start + tz2, i + 1]
        if (w1 == 0) or (w2 == 0):
            shared_arr[tx2, ty2, tz2, i + 1] = 0
            shared_arr[tx, ty, tz, i + 1] = l1 + l2
            shared_arr[tx2, ty2, tz2, 0] = 0
        else:
            #hf components
            shared_arr[tx2, ty2, tz2, i + 1] = (- sw2 * l1 + sw1 * l2)/(sw1 + sw2)
            #lf components
            shared_arr[tx, ty, tz, i + 1] = (sw1 * l1 + sw2 * l2)/(sw1 + sw2)
            shared_arr[tx2, ty2, tz2, 0] = -1
    shared_arr[tx, ty, tz, 0] = w1 + w2
    cuda.syncthreads()


    for step in range(1, n_levels*3):
        axis = step % 3
        if axis == 0:
            tx *= 2
            dx *= 2
            tx2 = tx + dx // 2
            ty2 = ty
            tz2 = tz 
        elif axis == 1:
            ty *= 2
            dy *= 2
            ty2 = ty + dy // 2
            tx2 = tx
            tz2 = tz 
        else:
            tz *= 2
            dz *= 2
            tz2 = tz + dz // 2
            tx2 = tx
            ty2 = ty 

        if tx < 8 and ty < 8 and tz < 32:
            w1 = shared_arr[tx, ty, tz, 0]
            w2 = shared_arr[tx2, ty2, tz2, 0]
            sw1 = w1 ** 0.5
            sw2 = w2 ** 0.5
            for i in range(3):
                l1 = shared_arr[tx, ty, tz, i + 1]
                l2 = shared_arr[tx2, ty2, tz2, i + 1]
                if (w1 == 0) or (w2 == 0):
                    shared_arr[tx2, ty2, tz2, i + 1] = 0
                    shared_arr[tx, ty, tz, i + 1] = l1 + l2
                    shared_arr[tx2, ty2, tz2, 0] = 0
                else:
                    #hf components
                    shared_arr[tx2, ty2, tz2, i + 1] = (- sw2 * l1 + sw1 * l2)/(sw1 + sw2)
                    #lf components
                    shared_arr[tx, ty, tz, i + 1] = (sw1 * l1 + sw2 * l2)/(sw1 + sw2)
                    shared_arr[tx2, ty2, tz2, 0] = -1
            shared_arr[tx, ty, tz, 0] = w1 + w2
        cuda.syncthreads()

    tz = threadIdx % 32
    ty = (threadIdx // 32) % 8
    tx = ((threadIdx // 256) % 4) * 2
    for i in range(2):
        for c in range(4):
            block[x_start+tx+i, y_start+ty, z_start+tz, c] = shared_arr[tx + i, ty, tz, c]
    cuda.syncthreads()

@cuda.jit
def initialize_empty_array(arr):
    tz = cuda.threadIdx.x // 2
    tx = cuda.blockIdx.x * 2 + (cuda.threadIdx.x % 2)
    ty = cuda.blockIdx.y
    for i in range(arr.shape[-1]):
        arr[tx, ty, tz, i] = 0

@cuda.jit
def set_pc_in_array(arr, pc, posx, posy, posz):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    x = pc[pos, 0]
    y = pc[pos, 1]
    z = pc[pos, 2]
    if pos < len(pc):
        if (posx <= x < (posx + 512)) and (posy <= y < (posy + 512)) and (posz <= z < (posz + 512)):
            x = x - posx
            y = y - posy
            z = z - posz
            for i in range(arr.shape[-1] - 1):
                arr[x, y, z, i + 1] = float(pc[pos, i + 3])
            arr[x, y, z, 0] = 1.0

def get_hf_components(block, n_level):
    return block[np.where(block[..., 0] == -1)][:, 1:]

# @profile
def full_cuda_raht(
    pc,
    shape,
    max_num_iter = 3,
    shared_memory = True
):
    hf = []
    if shape[0] < 1024:
        vol = partial_cuda_raht(pc, shape, max_num_iter, shared_memory=shared_memory)
    else:
        size = 1024 // (2 ** max_num_iter)
        dx = size // 2
        vol = np.zeros((size, size, size, shape[-1]))
        for posx in range(2):
            for posy in range(2):
                for posz in range(2):
                    vol_temp = partial_cuda_raht(
                        pc,
                        (512, 512, 512, 4),
                        max_num_iter,
                        posx * 512,
                        posy * 512,
                        posz * 512,
                        shared_memory = shared_memory
                    )
                    x_start = posx * dx
                    x_end = (posx + 1) * dx
                    y_start = posy * dx
                    y_end = (posy + 1) * dx
                    z_start = posz * dx
                    z_end = (posz + 1) * dx
                    step = 2 ** max_num_iter
                    hf.append(get_hf_components(vol_temp, max_num_iter))
                    vol[x_start:x_end, y_start:y_end, z_start:z_end] = vol_temp[
                        ::step,
                        ::step,
                        ::step
                    ]

    weight, lf, final_hf = compute_parallel_raht(vol)
    hf.append(final_hf)

    return weight, lf, np.concatenate(hf, axis=0)

@profile
def partial_cuda_raht( 
    pc,
    shape,
    max_num_iter = 4,
    posx=0,
    posy=0,
    posz=0,
    shared_memory=True
):
    # in this case the block is saved in memory right away
    # block = np.zeros(shape, dtype=np.uint8)
    # block[tuple(pc[:, :3].T)] = np.concatenate([np.ones((len(pc), 1)), pc[:, 3:]], axis = 1)
    # cuda_vol = cuda.to_device(block.astype(np.float32))
    #here the block is created in the gpu memory
    cuda_vol = cuda.device_array(shape, np.float32)
    initialize_empty_array[(shape[0]//2, shape[0]), shape[0]*2](cuda_vol)
    threadsperblock = 1024
    blockspergrid = (len(pc) + (threadsperblock - 1)) // threadsperblock
    set_pc_in_array[threadsperblock, blockspergrid](cuda_vol, pc, posx, posy, posz)

    axis = 0
    final_hf = []
    coord_scale = np.array([1, 1, 1])
    block_size = len(cuda_vol)
    max_num_iter = 3 * max_num_iter
    i = 0
    if shared_memory:
        block_x = block_size // 8
        block_y = block_size // 8
        block_z = block_size // 32
        one_level_raht_full_gpu_shared_memory[(block_x, block_y, block_z), 1024](
            cuda_vol,
            max_num_iter // 3,
        )
    else:
        while np.product(coord_scale) != np.product(cuda_vol.shape[:3]) and i < max_num_iter:
            i += 1
            coord_scale[axis] *= 2
            block_x = block_size // coord_scale[0]
            block_y = block_size // coord_scale[1]
            block_z = block_size // coord_scale[2]
            one_level_raht_full_gpu[(block_x, block_y), block_z](cuda_vol, axis, coord_scale)
            axis = (axis + 1) % 3
            cuda.synchronize()

    vol = cuda_vol.copy_to_host()
    return vol

# @profile
def parallelized_raht(
    data: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    # repeat until the shape of the block becomes (1, 1, 1, c)
    block = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.uint8)
    block[tuple(data[:, :3].T)] = np.concatenate([np.ones((len(data), 1)), data[:, 3:]], axis = 1)
    return compute_parallel_raht(block)

@cuda.jit
def collapse_flattened_volume(vol, res):
    '''
    parallelizes the collapsing of the weights
    the number of threads is equal to the number of samples in the x axis
    the number of blocks is shape_y * shape_z
    Parameters:
        vol (np.ndarray): grid representation of the PC
    '''
    tz = cuda.threadIdx.x
    tx = cuda.blockIdx.x
    ty = cuda.blockIdx.y
    for i in range(vol.shape[-1]):
        res[tx, ty, tz, i] = vol[tx, ty, tz, 0, i] + vol[tx, ty, tz, 1, i]

# @profile
def compute_parallel_raht(
    block: np.ndarray
):
    '''
    performs a slightly different raht transform
    Parameters:
        block: np array of size of nxnxnx4 to be transformed where the first channel represent
               geometric  information while the remaining ones represent color
    Returns:
        the weight, and the hf and lf transformed coefficients 
    '''
    axis = 0
    final_hf = []
    channels = block.shape[-1]
    level = 0
    while np.prod(block.shape) > block.shape[-1]:
        nb = (1,) * axis + (2,) + (1,) * (2 - axis) 
        flattened_block = flatten_cubes(block, nb)
        new_lf_idxs = np.where(np.min(flattened_block[..., 0], axis=-1) > 0)
        block = np.sum(flattened_block, axis=-2).astype(np.float32)
        if len(new_lf_idxs[0]) > 0:

            blocks_to_process = flattened_block[new_lf_idxs].reshape(
                (-1, *nb, channels)
            ).astype(np.float32)
            #needed to save some ram memory

            lf, hf = one_level_raht(
                blocks_to_process, 
                axis=axis
            )
            block[new_lf_idxs] = lf
            final_hf.append(hf.reshape(-1, 3))
        axis = (axis + 1) % 3

    final_hf = np.concatenate(final_hf, axis=0).reshape((-1, channels - 1))
    return block[0, 0, 0, 0], block[0, 0, 0, 1:], final_hf



# @profile
def raht(block: np.ndarray, slightly_parallelized: bool = True) -> np.ndarray:

    '''
    performs a slightly different raht transform
    Parameters:
        block: np array of size of nxnxnx4 to be transformed where the first channel represent
               geometric  information while the remaining ones represent color
    Returns:
        the weight, and the hf and lf transformed coefficients 
    '''
    return _raht(block, axis = 2, slightly_parallelized = slightly_parallelized)

def is_empty(block):
    x, y, z = block.shape[:3]
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if block[i, j, k, 0]:
                    return False
    return True

def _raht(block, axis, slightly_parallelized = True):

    if block.shape == (1, 1, 1, 4):
        return block[0, 0, 0, 0], block[0, 0, 0, 1:], None
    weights = []
    hf = []
    lf = []
    slices_val = [
        (slice(block.shape[axis] // 2),),
        (slice(block.shape[axis] // 2, block.shape[axis]),)
    ]
    for i in range(2):
        wanted = (slice(None),) * axis +  slices_val[i]
        sub_block = block[wanted].astype(np.float32) 
        condition = np.any(sub_block[..., 0]) if slightly_parallelized else not is_empty(sub_block)
        if condition:
            results = _raht(sub_block, axis=(axis - 1) % 3)
            if results[2]:
                hf += results[2]
            weights += [results[0]]
            lf += [results[1]]
        else:
            weights += [0]
            lf += [0]

    if np.min(weights) == 0:
        return np.max(weights), lf[np.argmax(weights, axis=0)], hf
    else:
        w1, w2 = weights
        sw1 = w1 ** 0.5
        sw2 = w2 ** 0.5
        l1, l2 = lf
        hf.append((- sw2 * l1 + sw1 * l2)/(sw1 + sw2))
        new_lf = (sw1 * l1 + sw2 * l2)/(sw1 + sw2)
        return w1 + w2, new_lf, hf

def one_level_inverse_raht(
    coeffs: np.ndarray,
    geometry: np.ndarray
) -> np.ndarray:

    '''
    performs the invers operation of one_level_raht
    Parameters:
        coeffs: array of raht coefficients
        geometry: geometry information used to compute the coefficients
    Returns:
        the original blocks
    '''

    lf = coeffs[:, :1].reshape((-1, 1, 1, 1, 3))
    for i in range(3):

        w = geometry
        for j in range(3 - i - 1):
            w = np.sum(w, axis = j + 1, keepdims=True)

        w_sqrt = np.sqrt(w) 
        w_sqrt_inv = np.concatenate([
            - w_sqrt[(slice(None),) * (3-i) + (slice(1, 2),)],
            w_sqrt[(slice(None),) * (3-i) + (slice(0, 1),)],
        ], axis = 3 - i)
        w_coeffs = np.concatenate([w_sqrt, w_sqrt_inv], axis=-1)
        w_coeffs = np.swapaxes(w_coeffs, 3-i, -2).swapaxes(-1, -2)
        zero_mat = np.abs(w_coeffs).reshape(w_coeffs.shape[:-2] + (4,))
        zero_mat = np.where(zero_mat.sum(axis=-1) == 0)
        w_coeffs[zero_mat] = np.eye(2).reshape((1, 1, 1, 2, 2))
        w_coeffs = np.linalg.inv(w_coeffs)
        w_coeffs[zero_mat] = 0
        
        w_div = np.sum(w_sqrt, axis = 3-i, keepdims=True)

        hf = coeffs[:, 2**i:2**(i+1)] 
        mixed_coeffs = np.concatenate([
            lf,
            hf.reshape(lf.shape) 
        ], axis=3-i).swapaxes(3-i, -2)
        lf = (w_coeffs @ mixed_coeffs).swapaxes(3-i, -2) * w_div

    return lf
