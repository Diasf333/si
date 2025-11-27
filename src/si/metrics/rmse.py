import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error between true and predicted values.
    
    Parameters
    ----------
    y_true: np.ndarray
        real values of y.
    y_pred: np.ndarray
        predicted values of y.
        
    Returns
    -------
    float
        float corresponding to the error between y_true and y_pred.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
