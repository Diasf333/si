from unittest import TestCase
import numpy as np

from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split


class TestStratifiedTrainTestSplit(TestCase):

    def setUp(self):
        # standard iris: 150 samples, 3 classes, 50 each
        self.iris = read_csv(
            "../datasets/iris/iris.csv",
            sep=",",
            features=True,
            label=True
        )

    def test_split_sizes(self):
        test_size = 0.2
        train_ds, test_ds = stratified_train_test_split(
            self.iris, test_size=test_size, random_state=42
        )

        n_samples = self.iris.X.shape[0]
        expected_test = int(round(test_size * n_samples))
        expected_train = n_samples - expected_test

        self.assertEqual(train_ds.X.shape[0], expected_train)
        self.assertEqual(test_ds.X.shape[0], expected_test)

    def test_stratification_preserves_class_proportions(self):
        test_size = 0.2
        train_ds, test_ds = stratified_train_test_split(
            self.iris, test_size=test_size, random_state=42
        )

        # class distribution in original, train, and test
        unique, counts_full = np.unique(self.iris.y, return_counts=True)
        _, counts_train = np.unique(train_ds.y, return_counts=True)
        _, counts_test = np.unique(test_ds.y, return_counts=True)

        # same number of classes
        self.assertEqual(len(unique), len(counts_train))
        self.assertEqual(len(unique), len(counts_test))

        # check per-class counts are close to the desired proportion
        for c_full, c_train, c_test in zip(counts_full, counts_train, counts_test):
            expected_test = int(round(test_size * c_full))
            expected_train = c_full - expected_test
            self.assertEqual(c_test, expected_test)
            self.assertEqual(c_train, expected_train)
