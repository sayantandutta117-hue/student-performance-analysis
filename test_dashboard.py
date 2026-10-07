import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashboard_helpers


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
        except Exception as e:
            self.fail(f"Failed to import app: {e}")


if __name__ == "__main__":
    unittest.main()
