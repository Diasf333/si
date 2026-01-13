import unittest
import numpy as np

from si.io.csv_file import read_csv
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search_cv import randomized_search_cv


class TestRandomizedSearchCV(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = read_csv(
            "datasets/breast_bin/breast-bin.csv",
            sep=",",
            features=True,
            label=True
        )
        
        self.model = LogisticRegression(
            l2_penalty=1.0,
            alpha=0.001,
            max_iter=1000,
            patience=100,
            scale=True
        )

        # hyperparameter distributions
        self.hyperparameter_grid = {
            "l2_penalty": np.linspace(1, 10, 10),
            "alpha": np.linspace(0.001, 0.0001, 100),
            "max_iter": np.linspace(1000, 2000, 200, dtype=int)
        }

    def test_randomized_search_cv(self) -> None:
        results = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_grid=self.hyperparameter_grid,
            scoring=None,
            cv=3,
            n_iter=10,
            random_state=42
        )

        # basic structure checks
        self.assertIn("hyperparameters", results)
        self.assertIn("scores", results)
        self.assertIn("best_hyperparameters", results)
        self.assertIn("best_score", results)

        # length check: n_iter combinations evaluated
        self.assertEqual(len(results["hyperparameters"]), 10)
        self.assertEqual(len(results["scores"]), 10)

        # score sanity: must be between 0 and 1
        for s in results["scores"]:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

        self.assertGreaterEqual(results["best_score"], 0.0)
        self.assertLessEqual(results["best_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
