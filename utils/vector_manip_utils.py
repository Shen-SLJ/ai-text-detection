import numpy as np
from typing import List, Optional, Callable
from scipy.sparse import hstack, spmatrix


def combine_spmatrix_with_1d_nparrays(
    sparse_matrix: spmatrix, nparrays: List[np.ndarray]
) -> spmatrix:
    """
    Combines a sparse matrix with a dense np array by horizontally stacking them.
    """
    combined = sparse_matrix

    for nparray in nparrays:
        combined = hstack([combined, nparray.reshape(-1, 1)])

    return combined


def combine_spmatrix(
    sparse_matrix1: Optional[spmatrix], sparse_matrix2: Optional[spmatrix]
) -> spmatrix:
    """
    Combines two sparse matrices by horizontally stacking them. At least one of the
    sparse matrices must be provided.
    """
    if sparse_matrix1 is None and sparse_matrix2 is None:
        raise ValueError(
            "At least one of sparse_matrix1 or sparse_matrix2 must be provided."
        )

    h_stack_param = ([sparse_matrix1] if sparse_matrix1 is not None else []) + (
        [sparse_matrix2] if sparse_matrix2 is not None else []
    )
    combined = hstack(h_stack_param)

    return combined
