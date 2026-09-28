#student analysis report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import sys

# ------------------ Validation helpers ------------------

def get_valid_name(prompt):
    while True:
        name = input(prompt).strip()
        if name:
            return name
        print("Invalid name. Name cannot be empty or whitespace-only. Please try again.")

def get_valid_marks(prompt):
    while True:
        try:
            marks = int(input(prompt))
            if 0 <= marks <= 100:
                return marks
            elif marks < 0:
                print("Invalid marks. Marks cannot be negative. Please enter a value between 0 and 100.")
            else:
                print("Invalid marks. Marks cannot be greater than 100. Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for marks.")

def get_valid_roll(prompt, existing_rolls):
    while True:
        try:
            roll = int(input(prompt))
            if roll in existing_rolls:
                print(f"Roll number {roll} already exists. Please enter a unique roll number.")
            else:
                return roll
        except ValueError:
            print("Invalid input. Please enter a numeric value for roll number.")

def get_valid_phone(prompt):
    while True:
        phone_str = input(prompt).strip()
        if not phone_str.isdigit():
            print("Invalid phone number. Please enter digits only.")
            continue
        if len(phone_str) != 10:
            print("Invalid phone number. It must be exactly 10 digits. Please try again.")
            continue
        return int(phone_str)

def get_valid_attendance(prompt):
    while True:
        try:
            attendance = int(input(prompt))
            if 0 <= attendance <= 100:
                return attendance
            elif attendance < 0:
                print("Invalid attendance. Attendance cannot be negative. Please enter a value between 0 and 100.")
            else:
                print("Invalid attendance. Attendance cannot be greater than 100. Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for attendance.")

def get_valid_menu_choice(prompt, min_choice, max_choice):
    while True:
        try:
            choice = int(input(prompt))
            if min_choice <= choice <= max_choice:
                return choice
            else:
                print(f"Invalid choice. Please enter a number between {min_choice} and {max_choice}.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# ------------------ Analysis helpers ------------------

def calculate_grade(marks):
    if marks >= 80:
        return "A"
    elif marks >= 70:
        return "B"
    elif marks >= 60:
        return "C"
    elif marks >= 50:
        return "D"
    else:
        return "F"

def assign_grades(student):
    if student.empty:
        return student
    student = student.copy()
    student["grade"] = student["marks"].apply(calculate_grade)
    return student

def pass_fail_analysis(student):
    if student.empty:
        print("No student available")
        return
    student = assign_grades(student)
    passed = student[student["grade"] != "F"]
    failed = student[student["grade"] == "F"]
    total = len(student)
    passed_count = len(passed)
    failed_count = len(failed)
    pass_pct = (passed_count / total) * 100 if total > 0 else 0.0
    print(f"Total students: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Pass Percentage: {pass_pct:.2f}%")

def top_3_students(student):
    if student.empty:
        print("No student available")
        return
    student = assign_grades(student)
    top3 = student.sort_values(by="marks", ascending=False).head(3).reset_index(drop=True)
    print("Top 3 Students:")
    for idx, row in top3.iterrows():
        print(f"Rank {idx + 1}: Roll {row['roll']}, {row['name']}, Marks: {row['marks']}, Grade: {row['grade']}")

def at_risk_students(student):
    if student.empty:
        print("No student available")
        return
    if "attendance" not in student.columns:
        print("Attendance data not available.")
        return
    at_risk = student[(student["marks"] < 40) | (student["attendance"] < 75)]
    if at_risk.empty:
        print("No at-risk students found.")
        return
    print("At-Risk Students:")
    for _, row in at_risk.iterrows():
        reasons = []
        if row["marks"] < 40:
            reasons.append("marks < 40")
        if row["attendance"] < 75:
            reasons.append("attendance < 75")
        print(f"Roll {row['roll']}, {row['name']}, Marks: {row['marks']}, Attendance: {row['attendance']}% | Reason: {', '.join(reasons)}")

def attendance_analysis(student):
    if student.empty:
        print("No student available")
        return
    if "attendance" not in student.columns:
        print("Attendance data not available.")
        return
    student = assign_grades(student)
    avg_marks = student["marks"].mean()

    def category(att):
        if att >= 90:
            return "Excellent"
        elif att >= 75:
            return "Good"
        else:
            return "Low"

    student = student.copy()
    student["attendance_category"] = student["attendance"].apply(category)
    print(f"Average Marks: {avg_marks:.2f}")
    print("Attendance Categories:")
    for cat in ["Excellent", "Good", "Low"]:
        count = len(student[student["attendance_category"] == cat])
        print(f"  {cat}: {count} students")
    print(student[["name", "roll", "marks", "attendance", "attendance_category"]])

def attendance_vs_marks(student):
    if student.empty:
        print("No student available")
        return
    if "attendance" not in student.columns:
        print("Attendance data not available.")
        return
    x_axis = student["attendance"]
    y_axis = student["marks"]
    plt.scatter(x_axis, y_axis, color="blue")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Marks")
    plt.title("Attendance vs Marks")
    if len(student) > 1:
        z = np.polyfit(x_axis, y_axis, 1)
        p = np.poly1d(z)
        plt.plot(x_axis, p(x_axis), "r--")
    plt.show()
    if len(student) > 1:
        corr = student["attendance"].corr(student["marks"])
        print(f"Correlation between Attendance and Marks: {corr:.2f}")
    else:
        print("Not enough data to calculate correlation.")

def grade_distribution(student):
    if student.empty:
        print("No student available")
        return
    student = assign_grades(student)
    counts = student["grade"].value_counts().sort_index()
    print("Grade Distribution:")
    for grade, count in counts.items():
        print(f"  {grade}: {count}")
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color="skyblue")
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.title("Grade Distribution")
    plt.xticks(rotation=0)
    plt.show()

def individual_report(student):
    if student.empty:
        print("No student available")
        return
    print("Available roll numbers:", student["roll"].tolist())
    r = get_valid_menu_choice("enter roll number::", -999999, 999999)
    if r not in student["roll"].values:
        print(f"No student found with roll number {r}.")
        return
    student_full = assign_grades(student)
    row = student_full[student_full["roll"] == r].iloc[0]
    print("=" * 40)
    print(f"Individual Report for Roll {r}")
    print("=" * 40)
    print(f"Name      : {row['name']}")
    print(f"Roll      : {row['roll']}")
    print(f"Marks     : {row['marks']}")
    print(f"Grade     : {row['grade']}")
    if "attendance" in student.columns:
        print(f"Attendance: {row['attendance']}%")
    print(f"Phone     : {row['phone']}")
    print("=" * 40)

# ------------------ Existing features ------------------

def main():
    n = get_valid_menu_choice("enter the number of students::", 0, 10000)
    name = []
    roll = []
    marks = []
    phone = []
    attendance = []
    existing_rolls = []
    for i in range(n):
        student_name = get_valid_name("enter student name:")
        student_roll = get_valid_roll("enter student roll::", existing_rolls)
        existing_rolls.append(student_roll)
        student_total_marks = get_valid_marks("enter total marks:")
        student_attendance = get_valid_attendance("enter attendance percentage (0-100):")
        student_phone_number = get_valid_phone("enter student phone number::")
        name.append(student_name)
        roll.append(student_roll)
        marks.append(student_total_marks)
        phone.append(student_phone_number)
        attendance.append(student_attendance)
    student = pd.DataFrame({"name": name, "roll": roll, "marks": marks, "phone": phone, "attendance": attendance})
    print(student)
    return student

def print_max_marks_student(student):
    if student.empty:
        print("no student available")
        return
    row = student.loc[student["marks"].idxmax()]
    print(row["roll"])
    print(row["name"])

def add(student):
    u = input("if you want to add new students::")
    if u.lower() == "y" or u.lower() == "yes":
        n = get_valid_menu_choice("enter how many students you want to add::", 0, 10000)
        name = []
        roll = []
        marks = []
        phone = []
        attendance = []
        existing_rolls = student["roll"].tolist()
        for i in range(n):
            new_student_name = get_valid_name("enter student name:")
            new_student_roll = get_valid_roll("enter student roll::", existing_rolls)
            existing_rolls.append(new_student_roll)
            new_student_total_marks = get_valid_marks("enter total marks:")
            new_student_attendance = get_valid_attendance("enter attendance percentage (0-100):")
            new_student_phone_number = get_valid_phone("enter student phone number::")
            name.append(new_student_name)
            roll.append(new_student_roll)
            marks.append(new_student_total_marks)
            phone.append(new_student_phone_number)
            attendance.append(new_student_attendance)
        new_student = pd.DataFrame({"name": name, "roll": roll, "marks": marks, "phone": phone, "attendance": attendance})
        student = pd.concat([student, new_student], ignore_index=True)
        return student
    elif u.lower() == "n" or u.lower() == "no":
        print("thank you , no new students")
    return student

def print_min_marks_student(student):
    if student.empty:
        print("No student available")
        return
    row = student.loc[student["marks"].idxmin()]
    print(row["roll"])
    print(row["name"])

def update(student):
    if student.empty:
        print("no student available")
        return student
    n = input("want to update::y/n")
    if n.lower() == "y" or n.lower() == "yes":
        r = get_valid_menu_choice("enter roll::", -999999, 999999)
        if r not in student["roll"].values:
            print("no such student present")
            return student
        else:
            print("what do you want to update::")
            print("1.update roll")
            print("2.update name")
            print("3.update marks")
            print("4.update phone")
            print("5.update attendance")
            choice = get_valid_menu_choice("enter choice::", 1, 5)
            if choice == 1:
                other_rolls = student.loc[student["roll"] != r, "roll"].tolist()
                new_r = get_valid_roll("enter new roll::", other_rolls)
                student.loc[student["roll"] == r, "roll"] = new_r
            elif choice == 2:
                new_n = get_valid_name("enter new name::")
                student.loc[student["roll"] == r, "name"] = new_n
            elif choice == 3:
                new_m = get_valid_marks("enter new marks:::")
                student.loc[student["roll"] == r, "marks"] = new_m
            elif choice == 4:
                new_ph = get_valid_phone("enter new phone number::")
                student.loc[student["roll"] == r, "phone"] = new_ph
            elif choice == 5:
                new_att = get_valid_attendance("enter new attendance percentage (0-100):")
                student.loc[student["roll"] == r, "attendance"] = new_att
    elif n.lower() == "n" or n.lower() == "no":
        print("everything is correct")
    print(student)
    return student

def delete_student(student):
    if student.empty:
        print("no student available")
        return student
    r = get_valid_menu_choice("enter roll number to delete:", -999999, 999999)
    if r not in student["roll"].values:
        print(f"Student with roll number {r} does not exist.")
        return student
    student = student[student["roll"] != r].reset_index(drop=True)
    print(f"Student with roll number {r} has been deleted successfully.")
    print(student)
    return student

def average(student):
    if student.empty:
        print("no student available")
        return
    p = student["marks"].mean()
    print("Average marks::", p)

def visualize(student):
    if student.empty:
        print("no student available")
        return
    x_axis = student["name"]
    y_axis = student["marks"]
    plt.bar(x_axis, y_axis, color="green")
    plt.xlabel("student name", color="RED", fontsize=15)
    plt.ylabel("student marks", color="BLUE", fontsize=15)
    plt.title("Students performance", color="YELLOW", fontsize=15)
    plt.show()

def show_all_students(student):
    if student.empty:
        print("no student available")
        return
    print(student)

def ranking(student):
    if student.empty:
        print("no student available")
        return
    s = student.sort_values(by="marks", ascending=False)
    print(s)

# ------------------ Main CLI ------------------

if __name__ == "__main__":
    student = pd.DataFrame(columns=["name", "roll", "marks", "phone", "attendance"])
    while True:
        print("1. PREPARING STUDENTS DETAILS")
        print("2. ADDING NEW STUDENTS")
        print("3. FINDING STUDENT WITH MAXIMUM MARKS")
        print("4. FINDING STUDENT WITH MINIMUM MARKS")
        print("5. UPDATING STUDENT")
        print("6. FINDING AVERAGE MARKS")
        print("7. VISUALISING STUDENT PERFMANCE")
        print("8. SHOW ALL THE STUDENTS")
        print("9. RANKING THE STUDENTS")
        print("10. DELETE STUDENT")
        print("11. GRADE ANALYSIS")
        print("12. PASS/FAIL ANALYSIS")
        print("13. TOP 3 STUDENTS")
        print("14. AT-RISK STUDENTS")
        print("15. ATTENDANCE ANALYSIS")
        print("16. ATTENDANCE VS MARKS")
        print("17. GRADE DISTRIBUTION")
        print("18. INDIVIDUAL STUDENT REPORT")
        print("19. EXIT")
        k = get_valid_menu_choice("enter your choice::", 1, 19)
        if k == 1:
            student = main()
        elif k == 2:
            student = add(student)
        elif k == 3:
            print_max_marks_student(student)
        elif k == 4:
            print_min_marks_student(student)
        elif k == 5:
            student = update(student)
        elif k == 6:
            average(student)
        elif k == 7:
            visualize(student)
        elif k == 8:
            show_all_students(student)
        elif k == 9:
            ranking(student)
        elif k == 10:
            student = delete_student(student)
        elif k == 11:
            if student.empty:
                print("no student available")
            else:
                student = assign_grades(student)
                print(student[["name", "roll", "marks", "grade"]])
        elif k == 12:
            pass_fail_analysis(student)
        elif k == 13:
            top_3_students(student)
        elif k == 14:
            at_risk_students(student)
        elif k == 15:
            attendance_analysis(student)
        elif k == 16:
            attendance_vs_marks(student)
        elif k == 17:
            grade_distribution(student)
        elif k == 18:
            individual_report(student)
        elif k == 19:
            sys.exit()
