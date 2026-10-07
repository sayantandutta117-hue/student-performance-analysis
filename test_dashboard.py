import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashboard_helpers
import prediction_helpers
import analytics


class TestComputeKpis(unittest.TestCase):
    def test_empty_dataframe(self):
        result = dashboard_helpers.compute_kpis(pd.DataFrame())
        self.assertEqual(result["total_students"], 0)
        self.assertIsNone(result["average_marks"])
        self.assertEqual(result["pass_count"], 0)
        self.assertIsNone(result["pass_rate"])
        self.assertEqual(result["fail_count"], 0)
        self.assertIsNone(result["average_attendance"])

    def test_valid_dataframe(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 35, 70],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60],
        })
        result = dashboard_helpers.compute_kpis(student)
        self.assertEqual(result["total_students"], 3)
        self.assertAlmostEqual(result["average_marks"], 63.33, places=1)
        self.assertEqual(result["pass_count"], 2)
        self.assertAlmostEqual(result["pass_rate"], 66.67, places=1)
        self.assertEqual(result["fail_count"], 1)
        self.assertAlmostEqual(result["average_attendance"], 78.33, places=1)


class TestGetGradeDistributionDf(unittest.TestCase):
    def test_empty_dataframe(self):
        result = dashboard_helpers.get_grade_distribution_df(pd.DataFrame())
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["Grade", "Count"])

    def test_valid_dataframe(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 35, 70],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60],
        })
        result = dashboard_helpers.get_grade_distribution_df(student)
        self.assertFalse(result.empty)
        self.assertIn("A", result["Grade"].values)
        self.assertIn("F", result["Grade"].values)
        self.assertEqual(result.loc[result["Grade"] == "A", "Count"].iloc[0], 1)


class TestGetAtRiskDf(unittest.TestCase):
    def test_empty_dataframe(self):
        result = dashboard_helpers.get_at_risk_df(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_no_attendance_column(self):
        student = pd.DataFrame({
            "name": ["Alice"],
            "roll": [1],
            "marks": [85],
            "phone": ["1234567890"],
        })
        result = dashboard_helpers.get_at_risk_df(student)
        self.assertTrue(result.empty)

    def test_valid_dataframe(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "roll": [1, 2, 3, 4],
            "marks": [85, 35, 39, 78],
            "phone": ["1234567890", "1234567891", "1234567892", "1234567893"],
            "attendance": [95, 80, 60, 70],
        })
        result = dashboard_helpers.get_at_risk_df(student)
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)


class TestGetRankedDf(unittest.TestCase):
    def test_empty_dataframe(self):
        result = dashboard_helpers.get_ranked_df(pd.DataFrame())
        self.assertTrue(result.empty)
        self.assertIn("Rank", result.columns)

    def test_valid_dataframe(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 92, 35],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60],
        })
        result = dashboard_helpers.get_ranked_df(student)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]["Rank"], 1)
        self.assertEqual(result.iloc[0]["name"], "Bob")
        self.assertEqual(result.iloc[1]["Rank"], 2)


class TestSearchStudents(unittest.TestCase):
    def test_empty_dataframe(self):
        result = dashboard_helpers.search_students(pd.DataFrame(), "Alice")
        self.assertTrue(result.empty)

    def test_valid_dataframe_search_by_roll(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [85, 90],
            "phone": ["1234567890", "1234567891"],
            "attendance": [95, 80],
        })
        result = dashboard_helpers.search_students(student, "1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

    def test_valid_dataframe_search_by_name(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [85, 90],
            "phone": ["1234567890", "1234567891"],
            "attendance": [95, 80],
        })
        result = dashboard_helpers.search_students(student, "ali")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")


class TestFilterByGrade(unittest.TestCase):
    def test_empty_dataframe(self):
        result = dashboard_helpers.filter_by_grade(pd.DataFrame(), "A")
        self.assertTrue(result.empty)

    def test_valid_dataframe(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 92, 35],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60],
        })
        result = dashboard_helpers.filter_by_grade(student, "A")
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 2)


class TestFilterByAttendanceCategory(unittest.TestCase):
    def test_empty_dataframe(self):
        result = dashboard_helpers.filter_by_attendance_category(pd.DataFrame(), "Excellent")
        self.assertTrue(result.empty)

    def test_valid_dataframe(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 92, 35],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60],
        })
        result = dashboard_helpers.filter_by_attendance_category(student, "Excellent")
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")


class TestGetDescriptiveStats(unittest.TestCase):
    def test_empty_dataframe(self):
        result = dashboard_helpers.get_descriptive_stats(pd.DataFrame())
        self.assertEqual(result, {})

    def test_valid_dataframe(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "roll": [1, 2, 3, 4, 5],
            "marks": [85, 92, 78, 35, 60],
            "phone": ["1234567890", "1234567891", "1234567892", "1234567893", "1234567894"],
            "attendance": [95, 80, 60, 70, 88],
        })
        result = dashboard_helpers.get_descriptive_stats(student)
        self.assertEqual(result["count"], 5)
        self.assertAlmostEqual(result["mean"], 70.0)
        self.assertAlmostEqual(result["median"], 78.0)
        self.assertAlmostEqual(result["q1"], 60.0)
        self.assertAlmostEqual(result["q3"], 85.0)


class TestGetAttendanceCategoryCounts(unittest.TestCase):
    def test_empty_dataframe(self):
        result = dashboard_helpers.get_attendance_category_counts(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_valid_dataframe(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 92, 35],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60],
        })
        result = dashboard_helpers.get_attendance_category_counts(student)
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)


class TestAppImport(unittest.TestCase):
    def test_app_imports_successfully(self):
        try:
            import app
            self.assertTrue(hasattr(app, "page_dashboard"))
            self.assertTrue(hasattr(app, "page_student_management"))
            self.assertTrue(hasattr(app, "page_analytics"))
            self.assertTrue(hasattr(app, "page_regression"))
            self.assertTrue(hasattr(app, "page_classification"))
            self.assertTrue(hasattr(app, "page_prediction"))
        except Exception as e:
            self.fail(f"Failed to import app: {e}")


class TestTrainSpecificRegressionModel(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"],
            "roll": [1, 2, 3, 4, 5, 6, 7],
            "marks": [85, 90, 78, 92, 65, 88, 72],
            "phone": ["1234567890"] * 7,
            "attendance": [95, 98, 80, 96, 60, 92, 75],
            "study_hours": [5.0, 4.0, 6.0, 5.5, 3.0, 6.0, 4.5],
            "assignment_score": [80, 85, 75, 90, 60, 88, 70],
            "midterm_marks": [70, 75, 68, 80, 55, 78, 65],
            "previous_marks": [78, 82, 74, 85, 60, 80, 72]
        })

    def test_valid_prediction_works(self):
        result = prediction_helpers.train_specific_regression_model(self.student, "Linear Regression")
        self.assertIsNotNone(result)
        self.assertIn("model", result)
        self.assertIn("features", result)
        self.assertEqual(result["features"], analytics.FEATURE_COLUMNS)
        self.assertEqual(result["used_rows"], 7)

    def test_insufficient_data_returns_none(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [85, 90],
            "phone": ["1234567890", "1234567891"],
            "attendance": [95, 80],
            "study_hours": [5.0, 4.0],
            "assignment_score": [80, 85],
            "midterm_marks": [70, 75],
            "previous_marks": [78, 82]
        })
        result = prediction_helpers.train_specific_regression_model(student, "Linear Regression")
        self.assertIsNone(result)

    def test_invalid_model_name_returns_none(self):
        result = prediction_helpers.train_specific_regression_model(self.student, "Invalid Model")
        self.assertIsNone(result)

    def test_expected_features_used(self):
        result = prediction_helpers.train_specific_regression_model(self.student, "Random Forest")
        self.assertIsNotNone(result)
        self.assertNotIn("marks", result["features"])
        self.assertNotIn("name", result["features"])
        self.assertNotIn("roll", result["features"])
        self.assertNotIn("phone", result["features"])


class TestTrainSpecificClassificationModel(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry"],
            "roll": [1, 2, 3, 4, 5, 6, 7, 8],
            "marks": [85, 45, 78, 92, 38, 88, 72, 55],
            "phone": ["1234567890"] * 8,
            "attendance": [95, 80, 96, 75, 60, 92, 85, 70],
            "study_hours": [5.0, 4.0, 6.0, 5.5, 3.0, 6.0, 4.5, 3.0],
            "assignment_score": [80, 85, 75, 90, 60, 88, 70, 65],
            "midterm_marks": [70, 75, 68, 80, 55, 78, 65, 60],
            "previous_marks": [78, 82, 74, 85, 60, 80, 72, 68]
        })

    def test_valid_prediction_returns_pass_or_fail(self):
        result = prediction_helpers.train_specific_classification_model(self.student, "Logistic Regression")
        self.assertIsNotNone(result)
        pred = analytics.predict_pass_fail(result["model"], [90, 5.0, 80, 70, 78])
        self.assertIn(pred, ["Pass", "Fail"])

    def test_insufficient_data_returns_none(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [85, 45],
            "phone": ["1234567890", "1234567891"],
            "attendance": [95, 80],
            "study_hours": [5.0, 4.0],
            "assignment_score": [80, 85],
            "midterm_marks": [70, 75],
            "previous_marks": [78, 82]
        })
        result = prediction_helpers.train_specific_classification_model(student, "Logistic Regression")
        self.assertIsNone(result)

    def test_single_class_returns_none(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 90, 88],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60],
            "study_hours": [5.0, 4.0, 6.0],
            "assignment_score": [80, 85, 75],
            "midterm_marks": [70, 75, 68],
            "previous_marks": [78, 82, 74]
        })
        result = prediction_helpers.train_specific_classification_model(student, "Logistic Regression")
        self.assertIsNone(result)

    def test_expected_features_used(self):
        result = prediction_helpers.train_specific_classification_model(self.student, "Random Forest")
        self.assertIsNotNone(result)
        self.assertNotIn("marks", result["features"])
        self.assertNotIn("name", result["features"])
        self.assertNotIn("roll", result["features"])
        self.assertNotIn("phone", result["features"])


class TestGetModelExplanation(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"],
            "roll": [1, 2, 3, 4, 5, 6, 7],
            "marks": [85, 90, 78, 92, 65, 88, 72],
            "phone": ["1234567890"] * 7,
            "attendance": [95, 98, 80, 96, 60, 92, 75],
            "study_hours": [5.0, 4.0, 6.0, 5.5, 3.0, 6.0, 4.5],
            "assignment_score": [80, 85, 75, 90, 60, 88, 70],
            "midterm_marks": [70, 75, 68, 80, 55, 78, 65],
            "previous_marks": [78, 82, 74, 85, 60, 80, 72]
        })

    def test_linear_regression_returns_coefficients(self):
        result = prediction_helpers.train_specific_regression_model(self.student, "Linear Regression")
        explanation = prediction_helpers.get_model_explanation(result["model"], "Linear Regression", "regression")
        self.assertIsNotNone(explanation)
        self.assertEqual(explanation["type"], "coefficients")
        self.assertEqual(len(explanation["values"]), 5)
        self.assertEqual(explanation["features"], analytics.FEATURE_COLUMNS)
        for v in explanation["values"]:
            self.assertIsInstance(v, float)

    def test_logistic_regression_returns_coefficients(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry"],
            "roll": [1, 2, 3, 4, 5, 6, 7, 8],
            "marks": [85, 45, 78, 92, 38, 88, 72, 55],
            "phone": ["1234567890"] * 8,
            "attendance": [95, 80, 96, 75, 60, 92, 85, 70],
            "study_hours": [5.0, 4.0, 6.0, 5.5, 3.0, 6.0, 4.5, 3.0],
            "assignment_score": [80, 85, 75, 90, 60, 88, 70, 65],
            "midterm_marks": [70, 75, 68, 80, 55, 78, 65, 60],
            "previous_marks": [78, 82, 74, 85, 60, 80, 72, 68]
        })
        result = prediction_helpers.train_specific_classification_model(student, "Logistic Regression")
        explanation = prediction_helpers.get_model_explanation(result["model"], "Logistic Regression", "classification")
        self.assertIsNotNone(explanation)
        self.assertEqual(explanation["type"], "coefficients")
        self.assertEqual(len(explanation["values"]), 5)
        self.assertEqual(explanation["features"], analytics.FEATURE_COLUMNS)
        for v in explanation["values"]:
            self.assertIsInstance(v, float)

    def test_decision_tree_returns_feature_importance(self):
        result = prediction_helpers.train_specific_regression_model(self.student, "Decision Tree")
        explanation = prediction_helpers.get_model_explanation(result["model"], "Decision Tree", "regression")
        self.assertIsNotNone(explanation)
        self.assertEqual(explanation["type"], "feature_importance")
        self.assertEqual(len(explanation["values"]), 5)
        self.assertEqual(explanation["features"], analytics.FEATURE_COLUMNS)
        for v in explanation["values"]:
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, 0.0)

    def test_random_forest_returns_feature_importance(self):
        result = prediction_helpers.train_specific_regression_model(self.student, "Random Forest")
        explanation = prediction_helpers.get_model_explanation(result["model"], "Random Forest", "regression")
        self.assertIsNotNone(explanation)
        self.assertEqual(explanation["type"], "feature_importance")
        self.assertEqual(len(explanation["values"]), 5)
        self.assertEqual(explanation["features"], analytics.FEATURE_COLUMNS)
        for v in explanation["values"]:
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, 0.0)

    def test_unsupported_model_returns_none(self):
        explanation = prediction_helpers.get_model_explanation(None, "Unknown Model")
        self.assertIsNone(explanation)


class TestValidatePredictionInputs(unittest.TestCase):
    def test_valid_inputs(self):
        valid, result = prediction_helpers.validate_prediction_inputs([75, 5.0, 80, 70, 78])
        self.assertTrue(valid)
        self.assertEqual(result, [75.0, 5.0, 80.0, 70.0, 78.0])

    def test_none_inputs(self):
        valid, msg = prediction_helpers.validate_prediction_inputs(None)
        self.assertFalse(valid)

    def test_wrong_length(self):
        valid, msg = prediction_helpers.validate_prediction_inputs([75, 5.0])
        self.assertFalse(valid)
        self.assertIn("5", msg)

    def test_non_numeric_inputs(self):
        valid, msg = prediction_helpers.validate_prediction_inputs([75, "abc", 80, 70, 78])
        self.assertFalse(valid)

    def test_feature_leakage_verification(self):
        self.assertNotIn("marks", analytics.FEATURE_COLUMNS)
        self.assertNotIn("name", analytics.FEATURE_COLUMNS)
        self.assertNotIn("roll", analytics.FEATURE_COLUMNS)
        self.assertNotIn("phone", analytics.FEATURE_COLUMNS)
        self.assertEqual(len(analytics.FEATURE_COLUMNS), 5)


if __name__ == "__main__":
    unittest.main()
