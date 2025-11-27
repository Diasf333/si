from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float =0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:

    """
    Split the dataset into stratified training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)

    # get unique class labels
    classes = np.unique(dataset.y)

    # initialize empty lists for train and test indices
    train_indices = []
    test_indices = []

    # loop through unique labels;
    for cls in classes:
        # all indices for this class
        cls_idx = np.where(dataset.y == cls)[0]

        # shuffle indices for the current class
        cls_idx = np.random.permutation(cls_idx)

        # calculate the number of test samples for the current class;
        n_test = int(round(test_size * len(cls_idx)))

        # select indices for the current class and add them to the test indices;
        test_indices.append(cls_idx[:n_test])

        # add the remaining indices to the train indices;
        train_indices.append(cls_idx[n_test:])

    # create arrays of indices
    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)

    # create training and testing datasets;
    X_train, y_train = dataset.X[train_indices], dataset.y[train_indices]
    X_test, y_test = dataset.X[test_indices], dataset.y[test_indices]

    train = Dataset(X_train, y_train, features=dataset.features, label=dataset.label)
    test = Dataset(X_test, y_test, features=dataset.features, label=dataset.label)

    # return the training and testing datasets.
    return train, test