from unittest import TestCase
from datasets import DATASETS_PATH
import os
import numpy as np
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification

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
