import numpy as np

def one_level_raht(block: np.ndarray) -> np.ndarray:

    '''
    performs a slightly different raht transform
    Parameters:
        block: array of 2x2x2 blocks to be transformed
    Returns:
        the transformed 2x2x2 block
    '''

    geom = block[..., :1]
    lf = block[..., 1:] * geom

    w = geom
    final = []
    for i in range(3):

        w_sqrt = np.sqrt(w) 
        w_sqrt_inv = np.concatenate([
            - w_sqrt[(slice(None),) * (i + 1) + (slice(1, 2),)],
            w_sqrt[(slice(None),) * (i + 1) + (slice(0, 1),)],
        ], axis = i + 1)
        w_coeffs = np.concatenate([w_sqrt, w_sqrt_inv], axis=-1)
        w_coeffs = np.swapaxes(w_coeffs, i+1, -2).swapaxes(-1, -2)
        
        w_div = np.maximum(np.sum(w_sqrt, axis = i+1, keepdims=True), 1)
        w = np.sum(w, axis = i+1, keepdims=True)

        coeffs = w_coeffs @ np.swapaxes(lf, i + 1, -2)
        coeffs = np.swapaxes(coeffs, i+1, -2) / w_div
        hf = coeffs[(slice(None),)*(i + 1) + (slice(1, 2),)]
        lf = coeffs[(slice(None),)*(i + 1) + (slice(0, 1),)]
        final = [hf.reshape((
            block.shape[0],
            -1,
            3
        ))] + final
    final = [lf.reshape((
        block.shape[0],
        -1,
        3
    ))] + final

    return np.concatenate(final, axis=1)

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
