import unittest
from unittest.mock import patch, MagicMock
import sys
import os

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


if __name__ == '__main__':
    unittest.main()
