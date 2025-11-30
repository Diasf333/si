from unittest import TestCase
import numpy as np

from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier


class TestRandomForestClassifier(TestCase):

    def setUp(self):
        # Load iris dataset
        # Adjust the path if your datasets folder is different
        self.iris = read_csv("C:\\Users\\UTILIZADOR\\Documents\\GitHub\\si\\datasets\\iris\\iris.csv", sep=",", features=True, label=True)

    def test_fit_creates_trees(self):
        train_ds, _ = train_test_split(self.iris, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(
            nestimators=10,
            maxfeatures=None,
            minsamplesplit=2,
            maxdepth=5,
            mode="gini",
            seed=42
        )

        rf.fit(train_ds)

        # Check that the correct number of trees was created
        self.assertEqual(len(rf.trees), rf.nestimators)

        # Each entry should be (feature_indices, DecisionTreeClassifier)
        feature_indices, tree = rf.trees[0]
        self.assertIsInstance(feature_indices, np.ndarray)
        self.assertTrue(hasattr(tree, "predict"))

    def test_predict_shape(self):
        train_ds, test_ds = train_test_split(self.iris, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(
            nestimators=10,
            maxfeatures=None,
            minsamplesplit=2,
            maxdepth=5,
            mode="gini",
            seed=42
        )

        rf.fit(train_ds)
        y_pred = rf.predict(test_ds)

        # Same number of predictions as test samples
        self.assertEqual(y_pred.shape[0], test_ds.X.shape[0])

    def test_score_reasonable(self):
        train_ds, test_ds = train_test_split(self.iris, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(
            nestimators=50,
            maxfeatures=None,
            minsamplesplit=2,
            maxdepth=10,
            mode="gini",
            seed=42
        )

        rf.fit(train_ds)
        score = rf.score(test_ds)

        # Accuracy should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # On iris, a reasonable random forest should be quite accurate
        self.assertGreater(score, 0.8)
