import unittest
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from src.clustering import save_model, train_kmeans
from src.insights import profile_segments
from src.preprocessing import get_feature_columns, prepare_dataset


class CustomerSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "CustomerID": range(1, 13),
                "Gender": ["Female", "Male"] * 6,
                "Age": [20, 21, 22, 23, 55, 56, 57, 58, 34, 36, 44, 46],
                "Annual Income (k$)": [20, 22, 21, 23, 90, 92, 91, 93, 45, 47, 60, 62],
                "Spending Score (1-100)": [80, 78, 82, 79, 20, 18, 22, 19, 51, 49, 42, 44],
            }
        )

    def test_preprocessing_drops_duplicates_and_detects_common_columns(self):
        data, report, detected = prepare_dataset(pd.concat([self.data, self.data.iloc[[0]]]))
        self.assertEqual(report["duplicate_rows_removed"], 1)
        self.assertEqual(len(data), 12)
        self.assertEqual(detected["income"], "Annual Income (k$)")
        self.assertEqual(detected["spending"], "Spending Score (1-100)")
        self.assertNotIn("CustomerID", get_feature_columns(data, detected))

    def test_training_profiles_and_prediction_pipeline(self):
        data, _, detected = prepare_dataset(self.data)
        features = [detected["age"], detected["income"], detected["spending"]]
        result = train_kmeans(data, features, 3)
        profiles = profile_segments(data, result["labels"], detected)
        self.assertEqual(len(set(result["labels"])), 3)
        self.assertTrue(-1 <= result["silhouette"] <= 1)
        self.assertEqual(int(result["evaluation"].iloc[0]["clusters"]), 1)
        self.assertEqual(int(profiles["customers"].sum()), len(data))
        prediction = result["pipeline"].predict(data[features].iloc[[0]])
        self.assertIn(int(prediction[0]), profiles["cluster"].tolist())

    def test_missing_values_are_imputed_by_model_pipeline(self):
        data, _, detected = prepare_dataset(self.data)
        data.loc[0, "Age"] = None
        features = [detected["age"], detected["income"], detected["spending"]]
        result = train_kmeans(data, features, 3)
        self.assertEqual(len(result["labels"]), len(data))
        self.assertTrue(
            result["pipeline"].predict(data[features].iloc[[0]]).size == 1
        )

    def test_categorical_features_are_encoded_and_missing_values_imputed(self):
        data, _, detected = prepare_dataset(self.data)
        data.loc[0, "Gender"] = None
        features = [detected["age"], detected["gender"]]
        result = train_kmeans(data, features, 3)
        self.assertEqual(len(result["labels"]), len(data))
        self.assertEqual(
            result["pipeline"].predict(data[features].iloc[[0]]).size, 1
        )

    def test_infinite_numeric_values_are_reported_and_imputed(self):
        data, _, detected = prepare_dataset(self.data)
        income_column = detected["income"]
        data[income_column] = data[income_column].astype(float)
        data.loc[0, income_column] = float("inf")
        cleaned, report, _ = prepare_dataset(data)
        self.assertEqual(report["non_finite_values_handled"], 1)
        self.assertTrue(pd.isna(cleaned.loc[0, income_column]))

    def test_trained_pipeline_persists_with_joblib(self):
        data, _, detected = prepare_dataset(self.data)
        features = [detected["age"], detected["income"], detected["spending"]]
        result = train_kmeans(data, features, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = save_model(result, str(Path(directory) / "customer_model.joblib"))
            saved = joblib.load(path)
            self.assertEqual(saved["features"], features)
            self.assertEqual(saved["pipeline"].predict(data[features].iloc[[0]]).size, 1)

    def test_constant_features_report_unavailable_cluster_count(self):
        data = pd.DataFrame({"first": [1] * 8, "second": [2] * 8})
        with self.assertRaisesRegex(ValueError, "fewer distinct customer profiles"):
            train_kmeans(data, ["first", "second"], 2)


if __name__ == "__main__":
    unittest.main()
