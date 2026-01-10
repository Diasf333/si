import numpy as np
from itertools import product
from typing import Callable, Dict, Any, List

from si.base.model import Model
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(
    model: Model,
    dataset: Dataset,
    hyperparameter_grid: Dict[str, np.ndarray],
    scoring: Callable[[Model, Dataset], float] | None = None,
    cv: int = 3,
    n_iter: int = 10,
    random_state: int | None = None
) -> Dict[str, Any]:
    """
    Randomized hyperparameter search with cross-validation.

    Parameters
    ----------
    model : Model
        Model to validate.
    dataset : Dataset
        Validation dataset.
    hyperparameter_grid : dict
        Dictionary with hyperparameter names as keys and iterable
        of search values as values.
    scoring : callable, optional
        Function to score the model. If None, uses model.score.
    cv : int, default=3
        Number of folds for k-fold cross-validation.
    n_iter : int, default=10
        Number of random hyperparameter combinations to test.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary with:
        - 'hyperparameters': list of tested hyperparameter combinations (dicts)
        - 'scores': list of mean scores for each combination
        - 'best_hyperparameters': best combination found
        - 'best_score': best mean score
    """
    # Validate hyperparameter names
    for param_name in hyperparameter_grid.keys():
        if not hasattr(model, param_name):
            raise ValueError(f"Model does not have hyperparameter '{param_name}'")

    # Build all possible combinations
    param_names: List[str] = list(hyperparameter_grid.keys())
    param_values: List[np.ndarray] = [np.array(v) for v in hyperparameter_grid.values()]
    all_combinations = list(product(*param_values))

    # Handle n_iter > total combinations
    n_total = len(all_combinations)
    n_iter = min(n_iter, n_total)

    # Random choice of indices
    rng = np.random.default_rng(random_state)
    chosen_indices = rng.choice(n_total, size=n_iter, replace=False)

    combinations_to_test = [all_combinations[i] for i in chosen_indices]

    # Results storage
    tested_hyperparams: List[Dict[str, Any]] = []
    scores: List[float] = []

    # Loop over random combinations
    for combo in combinations_to_test:
        # Build dict of current hyperparameters
        current_params = {name: value for name, value in zip(param_names, combo)}

        # Set model hyperparameters
        for name, value in current_params.items():
            setattr(model, name, value)

        # Cross-validation
        cv_scores = k_fold_cross_validation(
            model=model,
            dataset=dataset,
            scoring=scoring,
            cv=cv
        )

        mean_score = float(np.mean(cv_scores))

        tested_hyperparams.append(current_params)
        scores.append(mean_score)

    # Find best
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_hyperparams = tested_hyperparams[best_idx]

    # Return dictionary
    return {
        "hyperparameters": tested_hyperparams,
        "scores": scores,
        "best_hyperparameters": best_hyperparams,
        "best_score": best_score,
    }
