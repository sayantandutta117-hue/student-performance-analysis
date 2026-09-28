# Student Performance Analysis

## About
This project analyzes student performance using Python.
It allows adding, updating, deleting, ranking and visualizing student marks and attendance.

## Features
- Add student details
- Find maximum and minimum marks student
- Update student details
- Delete student details
- Calculate average marks
- Visualize performance using bar chart
- Rank students by marks
- Input validation for student names, marks, roll numbers, phone numbers, attendance, and menu choices (prevents crashes from invalid input)
- Grade calculation (A, B, C, D, F):
  - A: 80-100
  - B: 70-79
  - C: 60-69
  - D: 50-59
  - F: 0-49
- Pass/Fail analysis with pass percentage
- Attendance tracking (0-100) with update support
- Top 3 students report with rank, marks, and grade
- At-risk students report (marks < 40 OR attendance < 75) with reasons shown
- Attendance analysis with categories (Excellent: 90-100, Good: 75-89, Low: 0-74) and average marks
- Attendance vs Marks scatter plot with trend line and correlation coefficient
- Grade distribution counts with bar chart
- Individual student report by roll number

## Technologies Used
- Python
- Pandas
- Numpy
- Matplotlib

## How to Run
1. Clone this repository
2. Install requirements using: pip install -r requirements.txt
3. Run: python main.py
