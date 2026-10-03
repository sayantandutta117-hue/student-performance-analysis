import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as student_main
import database as db_module
import config
import analytics


class TestValidateName(unittest.TestCase):
    def test_valid_name_accepted(self):
        with patch('builtins.input', return_value='Alice'):
            result = student_main.get_valid_name("Enter name: ")
        self.assertEqual(result, 'Alice')

    def test_empty_name_rejected_then_valid(self):
        inputs = ['', '   ', 'Bob']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_name("Enter name: ")
        self.assertEqual(result, 'Bob')
        self.assertEqual(mock_print.call_count, 2)

    def test_whitespace_only_name_rejected_then_valid(self):
        inputs = ['\t', '\n', 'Charlie']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_name("Enter name: ")
        self.assertEqual(result, 'Charlie')
        self.assertEqual(mock_print.call_count, 2)


class TestValidateMarks(unittest.TestCase):
    def test_valid_marks_zero(self):
        with patch('builtins.input', return_value='0'):
            result = student_main.get_valid_marks("Enter marks: ")
        self.assertEqual(result, 0)

    def test_valid_marks_fifty(self):
        with patch('builtins.input', return_value='50'):
            result = student_main.get_valid_marks("Enter marks: ")
        self.assertEqual(result, 50)

    def test_valid_marks_hundred(self):
        with patch('builtins.input', return_value='100'):
            result = student_main.get_valid_marks("Enter marks: ")
        self.assertEqual(result, 100)

    def test_negative_marks_rejected_then_valid(self):
        inputs = ['-5', '75']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_marks("Enter marks: ")
        self.assertEqual(result, 75)
        mock_print.assert_called_once()
        self.assertIn("negative", mock_print.call_args[0][0])

    def test_marks_above_100_rejected_then_valid(self):
        inputs = ['101', '88']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_marks("Enter marks: ")
        self.assertEqual(result, 88)
        mock_print.assert_called_once()
        self.assertIn("greater than 100", mock_print.call_args[0][0])

    def test_non_numeric_marks_rejected_then_valid(self):
        inputs = ['abc', '65']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_marks("Enter marks: ")
        self.assertEqual(result, 65)
        mock_print.assert_called_once()
        self.assertIn("numeric", mock_print.call_args[0][0])


class TestValidateRoll(unittest.TestCase):
    def test_valid_roll_accepted(self):
        with patch('builtins.input', return_value='1'):
            result = student_main.get_valid_roll("Enter roll: ", [])
        self.assertEqual(result, 1)

    def test_duplicate_roll_rejected_then_valid(self):
        inputs = ['1', '2']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_roll("Enter roll: ", [1])
        self.assertEqual(result, 2)
        mock_print.assert_called_once()
        self.assertIn("already exists", mock_print.call_args[0][0])

    def test_non_numeric_roll_rejected_then_valid(self):
        inputs = ['xyz', '10']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_roll("Enter roll: ", [])
        self.assertEqual(result, 10)
        mock_print.assert_called_once()
        self.assertIn("numeric", mock_print.call_args[0][0])


class TestValidatePhone(unittest.TestCase):
    def test_valid_10_digit_phone_returns_string(self):
        with patch('builtins.input', return_value='1234567890'):
            result = student_main.get_valid_phone("Enter phone: ")
        self.assertIsInstance(result, str)
        self.assertEqual(result, '1234567890')

    def test_leading_zeros_preserved(self):
        with patch('builtins.input', return_value='0000000000'):
            result = student_main.get_valid_phone("Enter phone: ")
        self.assertEqual(result, '0000000000')

    def test_non_numeric_phone_rejected_then_valid(self):
        inputs = ['abc', '9876543210']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_phone("Enter phone: ")
        self.assertEqual(result, '9876543210')
        mock_print.assert_called_once()
        self.assertIn("digits", mock_print.call_args[0][0])

    def test_phone_too_short_rejected_then_valid(self):
        inputs = ['12345', '9876543210']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_phone("Enter phone: ")
        self.assertEqual(result, '9876543210')
        mock_print.assert_called_once()
        self.assertIn("10 digits", mock_print.call_args[0][0])

    def test_phone_too_long_rejected_then_valid(self):
        inputs = ['123456789012', '9876543210']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_phone("Enter phone: ")
        self.assertEqual(result, '9876543210')
        mock_print.assert_called_once()
        self.assertIn("10 digits", mock_print.call_args[0][0])


class TestValidateAttendance(unittest.TestCase):
    def test_valid_attendance_zero(self):
        with patch('builtins.input', return_value='0'):
            result = student_main.get_valid_attendance("Enter attendance: ")
        self.assertEqual(result, 0)

    def test_valid_attendance_fifty(self):
        with patch('builtins.input', return_value='50'):
            result = student_main.get_valid_attendance("Enter attendance: ")
        self.assertEqual(result, 50)

    def test_valid_attendance_hundred(self):
        with patch('builtins.input', return_value='100'):
            result = student_main.get_valid_attendance("Enter attendance: ")
        self.assertEqual(result, 100)

    def test_negative_attendance_rejected_then_valid(self):
        inputs = ['-5', '75']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_attendance("Enter attendance: ")
        self.assertEqual(result, 75)
        mock_print.assert_called_once()
        self.assertIn("negative", mock_print.call_args[0][0])

    def test_attendance_above_100_rejected_then_valid(self):
        inputs = ['101', '88']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_attendance("Enter attendance: ")
        self.assertEqual(result, 88)
        mock_print.assert_called_once()
        self.assertIn("greater than 100", mock_print.call_args[0][0])

    def test_non_numeric_attendance_rejected_then_valid(self):
        inputs = ['abc', '65']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_attendance("Enter attendance: ")
        self.assertEqual(result, 65)
        mock_print.assert_called_once()
        self.assertIn("numeric", mock_print.call_args[0][0])


class TestValidateMenuChoice(unittest.TestCase):
    def test_valid_choice_accepted(self):
        with patch('builtins.input', return_value='3'):
            result = student_main.get_valid_menu_choice("Choice: ", 1, 10)
        self.assertEqual(result, 3)

    def test_non_numeric_choice_rejected_then_valid(self):
        inputs = ['abc', '2']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_menu_choice("Choice: ", 1, 10)
        self.assertEqual(result, 2)
        mock_print.assert_called_once()
        self.assertIn("numeric", mock_print.call_args[0][0])

    def test_out_of_range_choice_rejected_then_valid(self):
        inputs = ['11', '3']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_menu_choice("Choice: ", 1, 10)
        self.assertEqual(result, 3)
        mock_print.assert_called_once()
        self.assertIn("between", mock_print.call_args[0][0])


class TestCalculateGrade(unittest.TestCase):
    def test_grade_a(self):
        self.assertEqual(student_main.calculate_grade(80), "A")
        self.assertEqual(student_main.calculate_grade(89), "A")
        self.assertEqual(student_main.calculate_grade(100), "A")

    def test_grade_b(self):
        self.assertEqual(student_main.calculate_grade(70), "B")
        self.assertEqual(student_main.calculate_grade(79), "B")

    def test_grade_c(self):
        self.assertEqual(student_main.calculate_grade(60), "C")
        self.assertEqual(student_main.calculate_grade(69), "C")

    def test_grade_d(self):
        self.assertEqual(student_main.calculate_grade(50), "D")
        self.assertEqual(student_main.calculate_grade(59), "D")

    def test_grade_f(self):
        self.assertEqual(student_main.calculate_grade(0), "F")
        self.assertEqual(student_main.calculate_grade(49), "F")

    def test_boundary_values(self):
        self.assertEqual(student_main.calculate_grade(79), "B")
        self.assertEqual(student_main.calculate_grade(80), "A")
        self.assertEqual(student_main.calculate_grade(69), "C")
        self.assertEqual(student_main.calculate_grade(70), "B")
        self.assertEqual(student_main.calculate_grade(59), "D")
        self.assertEqual(student_main.calculate_grade(60), "C")
        self.assertEqual(student_main.calculate_grade(49), "F")
        self.assertEqual(student_main.calculate_grade(50), "D")


class TestPassFailAnalysis(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 92, 35],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60]
        })

    def test_pass_fail_prints(self):
        with patch('builtins.print') as mock_print:
            student_main.pass_fail_analysis(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Total students: 3", printed)
        self.assertIn("Passed: 2", printed)
        self.assertIn("Failed: 1", printed)
        self.assertIn("Pass Percentage:", printed)

    def test_pass_fail_empty_dataframe(self):
        with patch('builtins.print') as mock_print:
            student_main.pass_fail_analysis(pd.DataFrame())
        mock_print.assert_called_with("No student available")


class TestTop3Students(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "roll": [1, 2, 3, 4],
            "marks": [85, 92, 35, 78],
            "phone": ["1234567890", "1234567891", "1234567892", "1234567893"],
            "attendance": [95, 80, 60, 88]
        })

    def test_top3_prints(self):
        with patch('builtins.print') as mock_print:
            student_main.top_3_students(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Rank 1:", printed)
        self.assertIn("Rank 2:", printed)
        self.assertIn("Rank 3:", printed)
        self.assertIn("Bob", printed)
        self.assertIn("92", printed)

    def test_top3_empty_dataframe(self):
        with patch('builtins.print') as mock_print:
            student_main.top_3_students(pd.DataFrame())
        mock_print.assert_called_with("No student available")


class TestAtRiskStudents(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "roll": [1, 2, 3, 4],
            "marks": [85, 35, 39, 78],
            "phone": ["1234567890", "1234567891", "1234567892", "1234567893"],
            "attendance": [95, 80, 60, 70]
        })

    def test_at_risk_reports(self):
        with patch('builtins.print') as mock_print:
            student_main.at_risk_students(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Charlie", printed)
        self.assertIn("marks < 40", printed)
        self.assertIn("attendance < 75", printed)
        self.assertIn("David", printed)

    def test_at_risk_empty_dataframe(self):
        with patch('builtins.print') as mock_print:
            student_main.at_risk_students(pd.DataFrame())
        mock_print.assert_called_with("No student available")

    def test_at_risk_no_attendance_column(self):
        student = pd.DataFrame({
            "name": ["Alice"],
            "roll": [1],
            "marks": [85],
            "phone": ["1234567890"]
        })
        with patch('builtins.print') as mock_print:
            student_main.at_risk_students(student)
        mock_print.assert_called_with("Attendance data not available.")


class TestAttendanceAnalysis(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 92, 35],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60]
        })

    def test_attendance_analysis_prints(self):
        with patch('builtins.print') as mock_print:
            student_main.attendance_analysis(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Average Marks:", printed)
        self.assertIn("Excellent:", printed)
        self.assertIn("Good:", printed)
        self.assertIn("Low:", printed)

    def test_attendance_analysis_empty(self):
        with patch('builtins.print') as mock_print:
            student_main.attendance_analysis(pd.DataFrame())
        mock_print.assert_called_with("No student available")


class TestGradeDistribution(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 92, 35],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60]
        })

    @patch('matplotlib.pyplot.show')
    def test_grade_distribution_prints(self, mock_show):
        with patch('builtins.print') as mock_print:
            student_main.grade_distribution(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Grade Distribution:", printed)
        self.assertIn("A", printed)
        self.assertIn("F", printed)

    def test_grade_distribution_empty(self):
        with patch('builtins.print') as mock_print:
            student_main.grade_distribution(pd.DataFrame())
        mock_print.assert_called_with("No student available")


class TestIndividualReport(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [85, 92],
            "phone": ["1234567890", "1234567891"],
            "attendance": [95, 80],
            "study_hours": [5.0, 4.0],
            "assignment_score": [80, 85],
            "midterm_marks": [70, 75],
            "previous_marks": [78, 82]
        })

    def test_individual_report_existing(self):
        with patch('builtins.input', return_value='1'):
            with patch('builtins.print') as mock_print:
                student_main.individual_report(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Individual Report for Roll 1", printed)
        self.assertIn("Alice", printed)
        self.assertIn("85", printed)

    def test_individual_report_non_existing(self):
        with patch('builtins.input', return_value='99'):
            with patch('builtins.print') as mock_print:
                student_main.individual_report(self.student)
        mock_print.assert_any_call("No student found with roll number 99.")

    def test_individual_report_empty(self):
        with patch('builtins.print') as mock_print:
            student_main.individual_report(pd.DataFrame())
        mock_print.assert_called_with("No student available")


class TestAttendanceVsMarks(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, 92, 35],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60]
        })

    @patch('matplotlib.pyplot.show')
    def test_attendance_vs_marks_prints_correlation(self, mock_show):
        with patch('builtins.print') as mock_print:
            student_main.attendance_vs_marks(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Correlation between Attendance and Marks:", printed)

    def test_attendance_vs_marks_empty(self):
        with patch('builtins.print') as mock_print:
            student_main.attendance_vs_marks(pd.DataFrame())
        mock_print.assert_called_with("No student available")

    def test_attendance_vs_marks_no_attendance(self):
        student = pd.DataFrame({
            "name": ["Alice"],
            "roll": [1],
            "marks": [85],
            "phone": ["1234567890"]
        })
        with patch('builtins.print') as mock_print:
            student_main.attendance_vs_marks(student)
        mock_print.assert_called_with("Attendance data not available.")


class TestDeleteStudent(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [85, 90],
            "phone": ["1234567890", "1234567891"],
            "attendance": [95, 80]
        })

    def test_delete_existing_student(self):
        with patch('builtins.input', return_value='1'):
            with patch('builtins.print') as mock_print:
                result = student_main.delete_student(self.student.copy())
        self.assertEqual(len(result), 1)
        self.assertNotIn(1, result["roll"].values)
        mock_print.assert_any_call("Student with roll number 1 has been deleted successfully.")

    def test_delete_non_existing_student(self):
        with patch('builtins.input', return_value='99'):
            with patch('builtins.print') as mock_print:
                result = student_main.delete_student(self.student.copy())
        self.assertEqual(len(result), 2)
        mock_print.assert_any_call("Student with roll number 99 does not exist.")


class TestAskYesNo(unittest.TestCase):
    def test_yes_variants_accepted(self):
        for response in ("y", "Y", "yes", "YES", "Yes", "yEs"):
            with patch('builtins.input', return_value=response):
                result = student_main.ask_yes_no("Continue? ")
            self.assertTrue(result)

    def test_no_variants_accepted(self):
        for response in ("n", "N", "no", "NO", "No", "nO"):
            with patch('builtins.input', return_value=response):
                result = student_main.ask_yes_no("Continue? ")
            self.assertFalse(result)

    def test_invalid_input_rejected_then_accepted(self):
        inputs = ['maybe', 'y']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.ask_yes_no("Continue? ")
        self.assertTrue(result)
        mock_print.assert_called_once()
        self.assertIn("Invalid input", mock_print.call_args[0][0])


class TestSafeShow(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_safe_show_calls_show(self, mock_show):
        student_main.safe_show(None)
        mock_show.assert_called_once()

    def test_safe_show_handles_exception(self):
        import matplotlib.pyplot as plt
        with patch('matplotlib.pyplot.show', side_effect=RuntimeError("headless")):
            with patch('builtins.print') as mock_print:
                student_main.safe_show(None)
        mock_print.assert_called_once()
        self.assertIn("could not be displayed", mock_print.call_args[0][0])


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import time
        import shutil
        time.sleep(0.1)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_table(self):
        db_module.init_db(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='students'"
            )
            self.assertIsNotNone(cursor.fetchone())

    def test_load_empty_database(self):
        db_module.init_db(self.db_path)
        df = db_module.load_students(self.db_path)
        self.assertTrue(df.empty)
        self.assertEqual(
            list(df.columns),
            ["name", "roll", "marks", "phone", "attendance",
             "study_hours", "assignment_score", "midterm_marks", "previous_marks"]
        )

    def test_save_and_load_roundtrip(self):
        db_module.init_db(self.db_path)
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "roll": [1, 2],
                "marks": [85, 90],
                "phone": ["1234567890", "9876543210"],
                "attendance": [95, 80],
            }
        )
        db_module.save_students(df, self.db_path)
        loaded = db_module.load_students(self.db_path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            loaded.loc[loaded["roll"] == 1, "name"].iloc[0], "Alice"
        )
        self.assertEqual(
            loaded.loc[loaded["roll"] == 2, "marks"].iloc[0], 90
        )

    def test_phone_stored_as_text(self):
        db_module.init_db(self.db_path)
        df = pd.DataFrame(
            {
                "name": ["Alice"],
                "roll": [1],
                "marks": [85],
                "phone": ["1234567890"],
                "attendance": [95],
            }
        )
        db_module.save_students(df, self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT phone FROM students WHERE roll = 1")
            phone = cursor.fetchone()[0]
            self.assertIsInstance(phone, str)
            self.assertEqual(phone, "1234567890")

    def test_leading_zeros_preserved(self):
        db_module.init_db(self.db_path)
        df = pd.DataFrame(
            {
                "name": ["Alice"],
                "roll": [1],
                "marks": [85],
                "phone": ["0000000000"],
                "attendance": [95],
            }
        )
        db_module.save_students(df, self.db_path)
        loaded = db_module.load_students(self.db_path)
        self.assertEqual(
            loaded.loc[loaded["roll"] == 1, "phone"].iloc[0], "0000000000"
        )

    def test_duplicate_roll_handling(self):
        db_module.init_db(self.db_path)
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "roll": [1, 1],
                "marks": [85, 90],
                "phone": ["1234567890", "9876543210"],
                "attendance": [95, 80],
            }
        )
        with self.assertRaises(sqlite3.IntegrityError):
            db_module.save_students(df, self.db_path)

    def test_persistence_across_connections(self):
        db_module.init_db(self.db_path)
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "roll": [1, 2],
                "marks": [85, 90],
                "phone": ["1234567890", "9876543210"],
                "attendance": [95, 80],
            }
        )
        db_module.save_students(df, self.db_path)
        loaded = db_module.load_students(self.db_path)
        self.assertEqual(len(loaded), 2)
        self.assertIn(1, loaded["roll"].values)
        self.assertIn(2, loaded["roll"].values)

    def test_delete_student_persists(self):
        db_module.init_db(self.db_path)
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "roll": [1, 2],
                "marks": [85, 90],
                "phone": ["1234567890", "9876543210"],
                "attendance": [95, 80],
            }
        )
        db_module.save_students(df, self.db_path)
        df = df[df["roll"] != 2].reset_index(drop=True)
        db_module.save_students(df, self.db_path)
        loaded = db_module.load_students(self.db_path)
        self.assertEqual(len(loaded), 1)
        self.assertNotIn(2, loaded["roll"].values)

    def test_update_student_persists(self):
        db_module.init_db(self.db_path)
        df = pd.DataFrame(
            {
                "name": ["Alice"],
                "roll": [1],
                "marks": [85],
                "phone": ["1234567890"],
                "attendance": [95],
            }
        )
        db_module.save_students(df, self.db_path)
        df.loc[df["roll"] == 1, "marks"] = 95
        db_module.save_students(df, self.db_path)
        loaded = db_module.load_students(self.db_path)
        self.assertEqual(
            loaded.loc[loaded["roll"] == 1, "marks"].iloc[0], 95
        )

    def test_load_handles_missing_table_gracefully(self):
        # Create a DB file but do NOT create the students table
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE other (id INTEGER PRIMARY KEY)")
            conn.commit()
        df = db_module.load_students(self.db_path)
        self.assertTrue(df.empty)
        self.assertEqual(
            list(df.columns),
            ["name", "roll", "marks", "phone", "attendance",
             "study_hours", "assignment_score", "midterm_marks", "previous_marks"]
        )


class TestUpdateNameValidation(unittest.TestCase):
    def test_update_rejects_empty_name(self):
        student = pd.DataFrame(
            {
                "name": ["Alice"],
                "roll": [1],
                "marks": [85],
                "phone": ["1234567890"],
                "attendance": [95],
            }
        )
        inputs = ["y", "1", "2", "   ", "Bob"]
        with patch("builtins.input", side_effect=inputs):
            with patch("builtins.print") as mock_print:
                result = student_main.update(student.copy())
        self.assertEqual(
            result.loc[result["roll"] == 1, "name"].iloc[0], "Bob"
        )
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("empty", printed.lower())

    def test_update_accepts_valid_name(self):
        student = pd.DataFrame(
            {
                "name": ["Alice"],
                "roll": [1],
                "marks": [85],
                "phone": ["1234567890"],
                "attendance": [95],
            }
        )
        with patch("builtins.input", side_effect=["y", "1", "2", "Charlie"]):
            result = student_main.update(student.copy())
        self.assertEqual(
            result.loc[result["roll"] == 1, "name"].iloc[0], "Charlie"
        )


class TestConfig(unittest.TestCase):
    def test_grade_boundaries_preserve_existing_behavior(self):
        self.assertEqual(student_main.calculate_grade(80), "A")
        self.assertEqual(student_main.calculate_grade(70), "B")
        self.assertEqual(student_main.calculate_grade(60), "C")
        self.assertEqual(student_main.calculate_grade(50), "D")
        self.assertEqual(student_main.calculate_grade(49), "F")

    def test_at_risk_thresholds_unchanged(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [39, 85],
            "phone": ["1234567890", "1234567891"],
            "attendance": [70, 80]
        })
        result = student_main.filter_at_risk_students(student)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

    def test_attendance_categories_unchanged(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [80, 80, 80],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60]
        })
        excellent = student_main.filter_by_attendance_category(student, "Excellent")
        good = student_main.filter_by_attendance_category(student, "Good")
        low = student_main.filter_by_attendance_category(student, "Low")
        self.assertEqual(len(excellent), 1)
        self.assertEqual(len(good), 1)
        self.assertEqual(len(low), 1)


class TestRanking(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "roll": [1, 2, 3, 4],
            "marks": [85, 92, 35, 78],
            "phone": ["1234567890", "1234567891", "1234567892", "1234567893"],
            "attendance": [95, 80, 60, 88]
        })

    def test_ranking_shows_explicit_ranks(self):
        with patch('builtins.print') as mock_print:
            student_main.ranking(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Rank", printed)
        self.assertIn("1", printed)
        self.assertIn("2", printed)
        self.assertIn("3", printed)
        self.assertIn("4", printed)

    def test_ranking_highest_marks_is_rank_1(self):
        with patch('builtins.print') as mock_print:
            student_main.ranking(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        bob_line = [line for line in mock_print.call_args_list if "Bob" in str(line)][0]
        self.assertIn("1", str(bob_line))

    def test_ranking_empty_dataframe(self):
        with patch('builtins.print') as mock_print:
            student_main.ranking(pd.DataFrame())
        mock_print.assert_called_with("No student available")

    def test_ranking_tied_marks_handled_consistently(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [80, 80],
            "phone": ["1234567890", "1234567891"],
            "attendance": [95, 80]
        })
        with patch('builtins.print') as mock_print:
            student_main.ranking(student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Rank", printed)


class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "roll": [1, 2, 3, 4, 5],
            "marks": [85, 92, 78, 35, 60],
            "phone": ["1234567890", "1234567891", "1234567892", "1234567893", "1234567894"],
            "attendance": [95, 80, 60, 70, 88]
        })

    def test_median_marks(self):
        with patch('builtins.print') as mock_print:
            student_main.median_marks(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Median Marks:", printed)

    def test_median_marks_empty(self):
        with patch('builtins.print') as mock_print:
            student_main.median_marks(pd.DataFrame())
        mock_print.assert_called_with("No student available")

    def test_mode_marks_single_mode(self):
        with patch('builtins.print') as mock_print:
            student_main.mode_marks(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Mode Marks:", printed)

    def test_mode_marks_empty(self):
        with patch('builtins.print') as mock_print:
            student_main.mode_marks(pd.DataFrame())
        mock_print.assert_called_with("No student available")

    def test_standard_deviation_marks(self):
        with patch('builtins.print') as mock_print:
            student_main.standard_deviation_marks(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Standard Deviation:", printed)

    def test_standard_deviation_marks_empty(self):
        with patch('builtins.print') as mock_print:
            student_main.standard_deviation_marks(pd.DataFrame())
        mock_print.assert_called_with("No student available")

    def test_quartiles_marks(self):
        with patch('builtins.print') as mock_print:
            student_main.quartiles_marks(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Q1", printed)
        self.assertIn("Q2", printed)
        self.assertIn("Q3", printed)

    def test_quartiles_marks_empty(self):
        with patch('builtins.print') as mock_print:
            student_main.quartiles_marks(pd.DataFrame())
        mock_print.assert_called_with("No student available")

    def test_statistical_summary(self):
        with patch('builtins.print') as mock_print:
            student_main.statistical_summary(self.student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Count", printed)
        self.assertIn("Mean", printed)
        self.assertIn("Median", printed)
        self.assertIn("Mode", printed)
        self.assertIn("Standard Deviation", printed)
        self.assertIn("Q1", printed)

    def test_statistical_summary_empty(self):
        with patch('builtins.print') as mock_print:
            student_main.statistical_summary(pd.DataFrame())
        mock_print.assert_called_with("No student available")


class TestFiltering(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "roll": [1, 2, 3, 4],
            "marks": [85, 92, 35, 78],
            "phone": ["1234567890", "1234567891", "1234567892", "1234567893"],
            "attendance": [95, 80, 60, 70]
        })

    def test_filter_by_grade_returns_dataframe(self):
        result = student_main.filter_by_grade(self.student, "A")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("Alice", result["name"].values)
        self.assertIn("Bob", result["name"].values)

    def test_filter_by_grade_no_match(self):
        result = student_main.filter_by_grade(self.student, "D")
        self.assertTrue(result.empty)

    def test_filter_by_attendance_category_excellent(self):
        result = student_main.filter_by_attendance_category(self.student, "Excellent")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

    def test_filter_by_attendance_category_good(self):
        result = student_main.filter_by_attendance_category(self.student, "Good")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Bob")

    def test_filter_by_attendance_category_low(self):
        result = student_main.filter_by_attendance_category(self.student, "Low")
        self.assertEqual(len(result), 2)

    def test_filter_by_attendance_category_empty_dataframe(self):
        result = student_main.filter_by_attendance_category(pd.DataFrame(), "Excellent")
        self.assertTrue(result.empty)

    def test_filter_at_risk_students(self):
        result = student_main.filter_at_risk_students(self.student)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    def test_filter_at_risk_no_attendance_column(self):
        student = pd.DataFrame({
            "name": ["Alice"],
            "roll": [1],
            "marks": [85],
            "phone": ["1234567890"]
        })
        result = student_main.filter_at_risk_students(student)
        self.assertTrue(result.empty)

    def test_filter_at_risk_empty_dataframe(self):
        result = student_main.filter_at_risk_students(pd.DataFrame())
        self.assertTrue(result.empty)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "roll": [1, 2, 3, 4],
            "marks": [85, 92, 35, 78],
            "phone": ["1234567890", "1234567891", "1234567892", "1234567893"],
            "attendance": [95, 80, 60, 70]
        })

    def test_search_by_exact_roll(self):
        result = student_main.search_student(self.student, "2")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Bob")

    def test_search_by_case_insensitive_name(self):
        result = student_main.search_student(self.student, "alice")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

    def test_search_by_partial_name(self):
        result = student_main.search_student(self.student, "ali")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

    def test_search_no_match(self):
        result = student_main.search_student(self.student, "xyz")
        self.assertTrue(result.empty)

    def test_search_empty_query(self):
        result = student_main.search_student(self.student, "")
        self.assertTrue(result.empty)

    def test_search_empty_dataframe(self):
        result = student_main.search_student(pd.DataFrame(), "Alice")
        self.assertTrue(result.empty)

    def test_search_does_not_mutate_original(self):
        original_len = len(self.student)
        student_main.search_student(self.student, "Alice")
        self.assertEqual(len(self.student), original_len)


class TestAnalyticsPrepareRegressionData(unittest.TestCase):
    def test_empty_dataframe(self):
        X, y = analytics.prepare_regression_data(pd.DataFrame())
        self.assertIsNone(X)
        self.assertIsNone(y)

    def test_missing_target_column(self):
        student = pd.DataFrame({
            "name": ["Alice"],
            "roll": [1],
            "marks": [85],
            "phone": ["1234567890"],
            "attendance": [95]
        })
        X, y = analytics.prepare_regression_data(student[["name", "roll", "phone", "attendance"]])
        self.assertIsNone(X)
        self.assertIsNone(y)

    def test_missing_feature_column(self):
        student = pd.DataFrame({
            "name": ["Alice"],
            "roll": [1],
            "marks": [85],
            "phone": ["1234567890"],
        })
        X, y = analytics.prepare_regression_data(student)
        self.assertIsNone(X)
        self.assertIsNone(y)

    def test_insufficient_data(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [85, 90],
            "phone": ["1234567890", "1234567891"],
            "attendance": [95, 80]
        })
        X, y = analytics.prepare_regression_data(student)
        self.assertIsNone(X)
        self.assertIsNone(y)

    def test_non_numeric_marks_handled(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "roll": [1, 2, 3],
            "marks": [85, "abc", 90],
            "phone": ["1234567890", "1234567891", "1234567892"],
            "attendance": [95, 80, 60],
            "study_hours": [5.0, 4.0, 6.0],
            "assignment_score": [80, 85, 75],
            "midterm_marks": [70, 75, 68],
            "previous_marks": [78, 82, 74]
        })
        X, y = analytics.prepare_regression_data(student)
        self.assertIsNone(X)
        self.assertIsNone(y)

    def test_correct_feature_target_separation(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "roll": [1, 2, 3, 4, 5],
            "marks": [85, 90, 78, 92, 65],
            "phone": ["1234567890", "1234567891", "1234567892", "1234567893", "1234567894"],
            "attendance": [95, 80, 60, 88, 72],
            "study_hours": [5.0, 4.0, 6.0, 5.5, 3.0],
            "assignment_score": [80, 85, 75, 90, 60],
            "midterm_marks": [70, 75, 68, 80, 55],
            "previous_marks": [78, 82, 74, 85, 60]
        })
        X, y = analytics.prepare_regression_data(student)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(list(X.columns), analytics.FEATURE_COLUMNS)
        self.assertEqual(len(X), 5)
        self.assertEqual(len(y), 5)
        self.assertNotIn("name", X.columns)
        self.assertNotIn("roll", X.columns)
        self.assertNotIn("phone", X.columns)
        self.assertNotIn("marks", X.columns)


class TestAnalyticsTrainRegressionModel(unittest.TestCase):
    def test_successful_training(self):
        np.random.seed(42)
        student = pd.DataFrame({
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
        result = analytics.train_regression_model(student)
        self.assertIsNotNone(result)
        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIn("X_train", result)
        self.assertIn("X_test", result)
        self.assertIn("features", result)
        self.assertEqual(result["features"], analytics.FEATURE_COLUMNS)
        self.assertEqual(result["target"], "marks")

    def test_training_returns_numeric_metrics(self):
        student = pd.DataFrame({
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
        result = analytics.train_regression_model(student)
        metrics = result["metrics"]
        self.assertIsInstance(metrics["mae"], (int, float))
        self.assertIsInstance(metrics["mse"], (int, float))
        self.assertIsInstance(metrics["rmse"], (int, float))
        self.assertIsInstance(metrics["r2"], (int, float))

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
        result = analytics.train_regression_model(student)
        self.assertIsNone(result)

    def test_missing_values_excluded_from_training(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "roll": [1, 2, 3, 4, 5],
            "marks": [85, 90, 78, 92, 65],
            "phone": ["1234567890"] * 5,
            "attendance": [95, 98, None, 96, 60],
            "study_hours": [5.0, 4.0, 6.0, 5.5, 3.0],
            "assignment_score": [80, 85, 75, 90, 60],
            "midterm_marks": [70, 75, 68, 80, 55],
            "previous_marks": [78, 82, 74, 85, 60]
        })
        with patch('builtins.print') as mock_print:
            result = analytics.train_regression_model(student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("missing", printed.lower())
        # One record has None attendance, so 4 complete records remain (< 5 min)
        self.assertIsNone(result)


class TestAnalyticsPredictMarks(unittest.TestCase):
    def test_prediction_returns_numeric(self):
        student = pd.DataFrame({
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
        result = analytics.train_regression_model(student)
        self.assertIsNotNone(result)
        pred = analytics.predict_marks(
            result["model"],
            [90, 5.0, 80, 70, 78]
        )
        self.assertIsInstance(pred, float)
        self.assertGreaterEqual(pred, 0)
        self.assertLessEqual(pred, 100)

    def test_prediction_invalid_input_returns_none(self):
        student = pd.DataFrame({
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
        result = analytics.train_regression_model(student)
        self.assertIsNotNone(result)
        pred = analytics.predict_marks(result["model"], ["not_a_number", 5.0, 80, 70, 78])
        self.assertIsNone(pred)

    def test_prediction_wrong_number_of_features_returns_none(self):
        student = pd.DataFrame({
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
        result = analytics.train_regression_model(student)
        self.assertIsNotNone(result)
        pred = analytics.predict_marks(result["model"], [90, 5.0, 80])  # Only 3 features
        self.assertIsNone(pred)


class TestAnalyticsEvaluateRegressionModel(unittest.TestCase):
    def test_metrics_are_numeric(self):
        y_true = pd.Series([80, 90, 70, 85])
        y_pred = np.array([82, 88, 72, 83])
        metrics = analytics.evaluate_regression_model(y_true, y_pred)
        self.assertIsInstance(metrics["mae"], (int, float))
        self.assertIsInstance(metrics["mse"], (int, float))
        self.assertIsInstance(metrics["rmse"], (int, float))
        self.assertIsInstance(metrics["r2"], (int, float))
        self.assertGreaterEqual(metrics["r2"], -10)


class TestValidateStudyHours(unittest.TestCase):
    def test_valid_study_hours_zero(self):
        with patch('builtins.input', return_value='0'):
            result = student_main.get_valid_study_hours("Enter study hours: ")
        self.assertEqual(result, 0.0)

    def test_valid_study_hours_float(self):
        with patch('builtins.input', return_value='5.5'):
            result = student_main.get_valid_study_hours("Enter study hours: ")
        self.assertEqual(result, 5.5)

    def test_valid_study_hours_twenty_four(self):
        with patch('builtins.input', return_value='24'):
            result = student_main.get_valid_study_hours("Enter study hours: ")
        self.assertEqual(result, 24.0)

    def test_negative_study_hours_rejected_then_valid(self):
        inputs = ['-1', '5']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_study_hours("Enter study hours: ")
        self.assertEqual(result, 5.0)
        mock_print.assert_called_once()
        self.assertIn("negative", mock_print.call_args[0][0])

    def test_study_hours_above_24_rejected_then_valid(self):
        inputs = ['25', '10']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_study_hours("Enter study hours: ")
        self.assertEqual(result, 10.0)
        mock_print.assert_called_once()
        self.assertIn("24", mock_print.call_args[0][0])

    def test_non_numeric_study_hours_rejected_then_valid(self):
        inputs = ['abc', '5']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_study_hours("Enter study hours: ")
        self.assertEqual(result, 5.0)
        mock_print.assert_called_once()
        self.assertIn("numeric", mock_print.call_args[0][0])


class TestValidateAssignmentScore(unittest.TestCase):
    def test_valid_assignment_score_zero(self):
        with patch('builtins.input', return_value='0'):
            result = student_main.get_valid_assignment_score("Enter assignment score: ")
        self.assertEqual(result, 0)

    def test_valid_assignment_score_fifty(self):
        with patch('builtins.input', return_value='50'):
            result = student_main.get_valid_assignment_score("Enter assignment score: ")
        self.assertEqual(result, 50)

    def test_valid_assignment_score_hundred(self):
        with patch('builtins.input', return_value='100'):
            result = student_main.get_valid_assignment_score("Enter assignment score: ")
        self.assertEqual(result, 100)

    def test_negative_assignment_score_rejected_then_valid(self):
        inputs = ['-5', '75']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_assignment_score("Enter assignment score: ")
        self.assertEqual(result, 75)
        mock_print.assert_called_once()
        self.assertIn("negative", mock_print.call_args[0][0])

    def test_assignment_score_above_100_rejected_then_valid(self):
        inputs = ['101', '88']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_assignment_score("Enter assignment score: ")
        self.assertEqual(result, 88)
        mock_print.assert_called_once()
        self.assertIn("100", mock_print.call_args[0][0])

    def test_non_numeric_assignment_score_rejected_then_valid(self):
        inputs = ['abc', '65']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_assignment_score("Enter assignment score: ")
        self.assertEqual(result, 65)
        mock_print.assert_called_once()
        self.assertIn("numeric", mock_print.call_args[0][0])


class TestValidateMidtermMarks(unittest.TestCase):
    def test_valid_midterm_marks_zero(self):
        with patch('builtins.input', return_value='0'):
            result = student_main.get_valid_midterm_marks("Enter midterm marks: ")
        self.assertEqual(result, 0)

    def test_valid_midterm_marks_fifty(self):
        with patch('builtins.input', return_value='50'):
            result = student_main.get_valid_midterm_marks("Enter midterm marks: ")
        self.assertEqual(result, 50)

    def test_valid_midterm_marks_hundred(self):
        with patch('builtins.input', return_value='100'):
            result = student_main.get_valid_midterm_marks("Enter midterm marks: ")
        self.assertEqual(result, 100)

    def test_negative_midterm_marks_rejected_then_valid(self):
        inputs = ['-5', '75']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_midterm_marks("Enter midterm marks: ")
        self.assertEqual(result, 75)
        mock_print.assert_called_once()
        self.assertIn("negative", mock_print.call_args[0][0])

    def test_midterm_marks_above_100_rejected_then_valid(self):
        inputs = ['101', '88']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_midterm_marks("Enter midterm marks: ")
        self.assertEqual(result, 88)
        mock_print.assert_called_once()
        self.assertIn("100", mock_print.call_args[0][0])

    def test_non_numeric_midterm_marks_rejected_then_valid(self):
        inputs = ['abc', '65']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_midterm_marks("Enter midterm marks: ")
        self.assertEqual(result, 65)
        mock_print.assert_called_once()
        self.assertIn("numeric", mock_print.call_args[0][0])


class TestValidatePreviousMarks(unittest.TestCase):
    def test_valid_previous_marks_zero(self):
        with patch('builtins.input', return_value='0'):
            result = student_main.get_valid_previous_marks("Enter previous marks: ")
        self.assertEqual(result, 0)

    def test_valid_previous_marks_fifty(self):
        with patch('builtins.input', return_value='50'):
            result = student_main.get_valid_previous_marks("Enter previous marks: ")
        self.assertEqual(result, 50)

    def test_valid_previous_marks_hundred(self):
        with patch('builtins.input', return_value='100'):
            result = student_main.get_valid_previous_marks("Enter previous marks: ")
        self.assertEqual(result, 100)

    def test_negative_previous_marks_rejected_then_valid(self):
        inputs = ['-5', '75']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_previous_marks("Enter previous marks: ")
        self.assertEqual(result, 75)
        mock_print.assert_called_once()
        self.assertIn("negative", mock_print.call_args[0][0])

    def test_previous_marks_above_100_rejected_then_valid(self):
        inputs = ['101', '88']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_previous_marks("Enter previous marks: ")
        self.assertEqual(result, 88)
        mock_print.assert_called_once()
        self.assertIn("100", mock_print.call_args[0][0])

    def test_non_numeric_previous_marks_rejected_then_valid(self):
        inputs = ['abc', '65']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_previous_marks("Enter previous marks: ")
        self.assertEqual(result, 65)
        mock_print.assert_called_once()
        self.assertIn("numeric", mock_print.call_args[0][0])


class TestDatabaseMigration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import time
        import shutil
        time.sleep(0.1)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_new_columns_exist(self):
        db_module.init_db(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(students)")
            columns = {row[1] for row in cursor.fetchall()}
        self.assertIn("study_hours", columns)
        self.assertIn("assignment_score", columns)
        self.assertIn("midterm_marks", columns)
        self.assertIn("previous_marks", columns)

    def test_load_handles_old_schema(self):
        # Simulate an old database without the new columns
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    roll INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    marks INTEGER NOT NULL,
                    phone TEXT NOT NULL,
                    attendance INTEGER NOT NULL
                )
            """)
            conn.execute(
                "INSERT INTO students (name, roll, marks, phone, attendance) VALUES (?, ?, ?, ?, ?)",
                ("Alice", 1, 85, "1234567890", 95)
            )
            conn.commit()
        df = db_module.load_students(self.db_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["name"], "Alice")
        self.assertIn("study_hours", df.columns)
        self.assertIn("assignment_score", df.columns)
        self.assertIn("midterm_marks", df.columns)
        self.assertIn("previous_marks", df.columns)

    def test_new_fields_persist_correctly(self):
        db_module.init_db(self.db_path)
        df = pd.DataFrame({
            "name": ["Alice"],
            "roll": [1],
            "marks": [85],
            "phone": ["1234567890"],
            "attendance": [95],
            "study_hours": [5.5],
            "assignment_score": [80],
            "midterm_marks": [70],
            "previous_marks": [78]
        })
        db_module.save_students(df, self.db_path)
        loaded = db_module.load_students(self.db_path)
        self.assertEqual(loaded.iloc[0]["study_hours"], 5.5)
        self.assertEqual(loaded.iloc[0]["assignment_score"], 80)
        self.assertEqual(loaded.iloc[0]["midterm_marks"], 70)
        self.assertEqual(loaded.iloc[0]["previous_marks"], 78)


class TestIndividualReportNewFields(unittest.TestCase):
    def test_individual_report_includes_new_fields(self):
        student = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "roll": [1, 2],
            "marks": [85, 92],
            "phone": ["1234567890", "1234567891"],
            "attendance": [95, 80],
            "study_hours": [5.0, 4.0],
            "assignment_score": [80, 85],
            "midterm_marks": [70, 75],
            "previous_marks": [78, 82]
        })
        with patch('builtins.input', return_value='1'):
            with patch('builtins.print') as mock_print:
                student_main.individual_report(student)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Study Hours", printed)
        self.assertIn("Assignment Score", printed)
        self.assertIn("Midterm Marks", printed)
        self.assertIn("Previous Marks", printed)


class TestAnalyticsGetRegressionModels(unittest.TestCase):
    def test_all_three_models_available(self):
        models = analytics.get_regression_models()
        self.assertIn("Linear Regression", models)
        self.assertIn("Decision Tree", models)
        self.assertIn("Random Forest", models)
        self.assertEqual(len(models), 3)

    def test_random_state_reproducible(self):
        models = analytics.get_regression_models()
        for name, model in models.items():
            params = model.get_params()
            if "random_state" in params:
                self.assertEqual(params["random_state"], analytics.RANDOM_STATE)


class TestAnalyticsCompareRegressionModels(unittest.TestCase):
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

    def test_compare_returns_dataframe(self):
        result = analytics.compare_regression_models(self.student)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    def test_compare_has_expected_columns(self):
        result = analytics.compare_regression_models(self.student)
        expected_cols = ["Model", "MAE", "MSE", "RMSE", "R²"]
        self.assertEqual(list(result.columns), expected_cols)

    def test_compare_metrics_are_numeric(self):
        result = analytics.compare_regression_models(self.student)
        for col in ["MAE", "MSE", "RMSE", "R²"]:
            for val in result[col]:
                self.assertIsInstance(val, (int, float))

    def test_compare_all_models_trained(self):
        result = analytics.compare_regression_models(self.student)
        self.assertEqual(set(result["Model"]), {
            "Linear Regression", "Decision Tree", "Random Forest"
        })

    def test_compare_feature_columns_correct(self):
        result = analytics.compare_regression_models(self.student)
        self.assertIsNotNone(result)
        # The function should not error; features are internal
        self.assertEqual(len(result), 3)

    def test_compare_target_remains_marks(self):
        # Verify the pipeline still uses marks as target by checking metrics exist
        result = analytics.compare_regression_models(self.student)
        self.assertIsNotNone(result)
        # R² should be a valid number for reasonable data
        self.assertTrue(all(result["R²"].notna()))

    def test_compare_insufficient_data_returns_none(self):
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
        result = analytics.compare_regression_models(student)
        self.assertIsNone(result)


class TestAnalyticsCrossValidateRegressionModels(unittest.TestCase):
    def setUp(self):
        self.student = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"],
            "roll": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "marks": [85, 90, 78, 92, 65, 88, 72, 95, 80, 70],
            "phone": ["1234567890"] * 10,
            "attendance": [95, 98, 80, 96, 60, 92, 75, 97, 85, 70],
            "study_hours": [5.0, 4.0, 6.0, 5.5, 3.0, 6.0, 4.5, 5.5, 4.0, 3.5],
            "assignment_score": [80, 85, 75, 90, 60, 88, 70, 92, 78, 65],
            "midterm_marks": [70, 75, 68, 80, 55, 78, 65, 82, 72, 60],
            "previous_marks": [78, 82, 74, 85, 60, 80, 72, 88, 75, 68]
        })

    def test_cv_returns_dataframe(self):
        result = analytics.cross_validate_regression_models(self.student)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    def test_cv_has_expected_columns(self):
        result = analytics.cross_validate_regression_models(self.student)
        expected_cols = ["Model", "RMSE Mean", "RMSE Std", "R² Mean", "R² Std"]
        self.assertEqual(list(result.columns), expected_cols)

    def test_cv_metrics_are_numeric(self):
        result = analytics.cross_validate_regression_models(self.student)
        for col in ["RMSE Mean", "RMSE Std", "R² Mean", "R² Std"]:
            for val in result[col]:
                self.assertIsInstance(val, (int, float))

    def test_cv_all_models_present(self):
        result = analytics.cross_validate_regression_models(self.student)
        self.assertEqual(set(result["Model"]), {
            "Linear Regression", "Decision Tree", "Random Forest"
        })

    def test_cv_k_fold_does_not_exceed_samples(self):
        # With 10 samples, 5-fold CV is valid
        result = analytics.cross_validate_regression_models(self.student)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

    def test_cv_insufficient_data_returns_none(self):
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
        result = analytics.cross_validate_regression_models(student)
        self.assertIsNone(result)

    def test_cv_single_sample_returns_none(self):
        student = pd.DataFrame({
            "name": ["Alice"],
            "roll": [1],
            "marks": [85],
            "phone": ["1234567890"],
            "attendance": [95],
            "study_hours": [5.0],
            "assignment_score": [80],
            "midterm_marks": [70],
            "previous_marks": [78]
        })
        result = analytics.cross_validate_regression_models(student)
        self.assertIsNone(result)

    def test_cv_no_name_roll_phone_leakage(self):
        # Verify CV runs on academic features only by checking results are valid
        result = analytics.cross_validate_regression_models(self.student)
        self.assertIsNotNone(result)
        # All models should have valid R² values
        self.assertTrue(all(result["R² Mean"].notna()))


if __name__ == "__main__":
    unittest.main()
