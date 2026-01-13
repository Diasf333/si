from unittest import TestCase
from datasets import DATASETS_PATH
import os
import numpy as np
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split
from si.models.knn_classifier import KNNClassifier
from si.metrics.accuracy import accuracy





class TestSelectPercentile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        select_percentile = SelectPercentile(score_function=f_classification, percentile=50)
        
        select_percentile.fit(self.dataset)
        self.assertTrue(select_percentile.F.shape[0] > 0)
        self.assertTrue(select_percentile.p.shape[0] > 0)

    def test_transform(self):
        selector = SelectPercentile(percentile=50)
        selector.fit(self.dataset)

        X_new = selector.transform(self.dataset)  

        # number of features reduced
        self.assertEqual(X_new.shape[1],
                        int(np.ceil(self.dataset.X.shape[1] * 0.5)))

        # still same number of samples
        self.assertEqual(X_new.shape[0], self.dataset.X.shape[0])


class TestSelectPercentileKNN(TestCase):
    def setUp(self) -> None:
        iris_path = os.path.join(DATASETS_PATH, "iris", "iris.csv")
        self.dataset = read_csv(
            filename=iris_path,
            sep=",",
            features=True,
            label=True
        )
        self.train_ds, self.test_ds = train_test_split(
            self.dataset,
            test_size=0.3,
            random_state=42
        )

    def test_select_percentile_with_knn(self) -> None:
        # 3) Fit SelectPercentile on the training set
        selector = SelectPercentile(score_function=f_classification, percentile=50)
        selector.fit(self.train_ds)

        # 4) Transform X and build reduced Dataset objects
        X_train_sel = selector.transform(self.train_ds)
        X_test_sel = selector.transform(self.test_ds)

        train_reduced = Dataset(
            X=X_train_sel,
            y=self.train_ds.y,
            features=None,
            label=self.train_ds.label
        )
        test_reduced = Dataset(
            X=X_test_sel,
            y=self.test_ds.y,
            features=None,
            label=self.test_ds.label
        )

        # 5) Train KNN on reduced dataset
        knn = KNNClassifier(k=5)
        knn.fit(train_reduced)

        # 6) Evaluate
        y_pred = knn.predict(test_reduced)
        score = accuracy(test_reduced.y, y_pred)

        # basic sanity checks
        self.assertIsInstance(score, (float, np.floating))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


