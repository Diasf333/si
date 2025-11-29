from unittest import TestCase
from datasets import DATASETS_PATH
import os
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
        # With 4 features, percentile=50 should select 2 features
        select_percentile = SelectPercentile(score_function=f_classification, percentile=50)
        
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)
        
        self.assertLess(len(new_dataset.features), len(self.dataset.features))
        self.assertLess(new_dataset.X.shape[1], self.dataset.X.shape[1])
        # Specifically for iris (4 features) at 50%, we expect 2 features
        self.assertEqual(new_dataset.X.shape[1], 2)
