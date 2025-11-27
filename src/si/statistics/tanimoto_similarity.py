import numpy as np

def tanimoto_similarity(X: np.ndarray , Y:np.ndarray): 
    """
    calculates the Tanimoto distances between x and each sample in y

    Parameters
    ----------
    X: A single binary sample (1D array or list).

    Y: Multiple binary samples (2D array or list of lists), where each row is a sample

    Returns
    -------
    np.ndarray: containing the Tanimoto distances between x and each sample in y.
    """
    return np.sum(X & Y, axis=1) / np.sum(X | Y, axis=1)