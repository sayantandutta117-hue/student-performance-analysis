import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as student_main


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
    def test_valid_10_digit_phone(self):
        with patch('builtins.input', return_value='1234567890'):
            result = student_main.get_valid_phone("Enter phone: ")
        self.assertEqual(result, 1234567890)

    def test_non_numeric_phone_rejected_then_valid(self):
        inputs = ['abc', '9876543210']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_phone("Enter phone: ")
        self.assertEqual(result, 9876543210)
        mock_print.assert_called_once()
        self.assertIn("digits", mock_print.call_args[0][0])

    def test_phone_too_short_rejected_then_valid(self):
        inputs = ['12345', '9876543210']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_phone("Enter phone: ")
        self.assertEqual(result, 9876543210)
        mock_print.assert_called_once()
        self.assertIn("10 digits", mock_print.call_args[0][0])

    def test_phone_too_long_rejected_then_valid(self):
        inputs = ['123456789012', '9876543210']
        with patch('builtins.input', side_effect=inputs):
            with patch('builtins.print') as mock_print:
                result = student_main.get_valid_phone("Enter phone: ")
        self.assertEqual(result, 9876543210)
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
            "phone": [1234567890, 1234567891, 1234567892],
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
            "phone": [1234567890, 1234567891, 1234567892, 1234567893],
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
            "phone": [1234567890, 1234567891, 1234567892, 1234567893],
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
            "phone": [1234567890]
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
            "phone": [1234567890, 1234567891, 1234567892],
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
            "phone": [1234567890, 1234567891, 1234567892],
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
            "phone": [1234567890, 1234567891],
            "attendance": [95, 80]
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
            "phone": [1234567890, 1234567891, 1234567892],
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
            "phone": [1234567890]
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
            "phone": [1234567890, 1234567891],
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


if __name__ == '__main__':
    unittest.main()
