import unittest

from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.ensemble.stacking_classifier import StackingClassifier


class TestStackingClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = read_csv(
            "datasets/breast_bin/breast-bin.csv",
            sep=",",
            features=True,
            label=True
        )
        self.train, self.test = train_test_split(
            self.dataset,
            test_size=0.2,
            random_state=42
        )

    def test_stacking_classifier(self) -> None:
        # base models
        knn1 = KNNClassifier(k=5)
        log_reg = LogisticRegression(
            l2_penalty=1.0,
            alpha=0.001,
            max_iter=1000,
            patience=100,
            scale=True
        )
        tree = DecisionTreeClassifier(
            min_samples_split=2,
            max_depth=10,
            mode="gini"
        )
        # final model
        knn2 = KNNClassifier(k=3)

        # stacking classifier
        stacking = StackingClassifier(
            models=[knn1, log_reg, tree],
            final_model=knn2
        )

        stacking.fit(self.train)
        score = stacking.score(self.test)

        # basic sanity check: score must be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
