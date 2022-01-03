import numpy as np

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

def np_parallelized_raht(block: np.ndarray) -> np.ndarray:
    '''
    performs a slightly different raht transform
    Parameters:
        block: np array of size of nxnxnx4 to be transformed where the first channel represent
               geometric  information while the remaining ones represent color
    Returns:
        the weight, and the hf and lf transformed coefficients 
    '''
    # repeat until the shape of the block becomes (1, 1, 1, c)
    axis = 0
    final_hf = []
    channels = block.shape[-1]
    while np.prod(block.shape) > block.shape[-1]:
        nb = (1,) * axis + (2,) + (1,) * (2 - axis) 
        flattened_block = flatten_cubes(block, nb)
        new_lf_idxs = np.where(np.min(flattened_block[..., 0], axis=-1) > 0)
        block = np.sum(flattened_block, axis=-2).astype(np.float32)
        if len(new_lf_idxs[0]) > 0:
            lf, hf = one_level_raht(
                flattened_block[new_lf_idxs].reshape((-1, *nb, channels)).astype(np.float32),
                axis=axis
            )
            block[new_lf_idxs] = lf
            final_hf.append(hf)
        axis = (axis + 1) % 3

    final_hf = np.concatenate(final_hf, axis=0).reshape((-1, channels - 1))
    return block[0, 0, 0, 0], block[0, 0, 0, 1:], final_hf


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
