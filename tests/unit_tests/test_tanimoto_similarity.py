from unittest import TestCase
import numpy as np

from si.statistics.tanimoto_similarity import tanimoto_similarity 


class TestTanimotoSimilarity(TestCase):

    def setUp(self):
        self.x = np.array([1, 0, 1, 0], dtype=bool)
        self.y = np.array([
            [1, 0, 1, 0],  
            [1, 1, 0, 0],  
            [0, 0, 0, 0],  
        ], dtype=bool)

    def test_tanimoto_basic_values(self):
        result = tanimoto_similarity(self.x, self.y)
        expected = np.array([
            1.0,        
            1/3,        
            0.0,       
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_tanimoto_shape(self):
        result = tanimoto_similarity(self.x, self.y)
        self.assertEqual(result.shape, (self.y.shape[0],))



