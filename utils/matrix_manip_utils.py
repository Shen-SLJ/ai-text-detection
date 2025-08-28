import numpy as np
from typing import List
from scipy.sparse import hstack, spmatrix


def combine_spmatrix_with_1d_nparrays(
    sparse_matrix: spmatrix, nparrays: List[np.ndarray]
) -> spmatrix:
    """
    Combines a sparse matrix with a dense np array by horizontally stacking them.
    """
    combined = sparse_matrix

    for nparray in nparrays:
        combined = hstack([sparse_matrix, nparray.reshape(-1, 1)])

    return combined
