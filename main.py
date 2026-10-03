#student analysis report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import sys
import database
import config
import analytics

# ------------------ Constants ------------------

MENU_MAX = 24

# ------------------ Validation helpers ------------------

def ask_yes_no(prompt):
    while True:
        response = input(prompt).strip().lower()
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Invalid input. Please enter y/yes or n/no.")

def safe_show(fig):
    try:
        plt.show()
    except Exception:
        print("Visualization could not be displayed in this environment.")

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
        return phone_str

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

def get_valid_study_hours(prompt):
    while True:
        try:
            hours = float(input(prompt))
            if 0 <= hours <= 24:
                return hours
            elif hours < 0:
                print("Invalid study hours. Cannot be negative. Please enter a value between 0 and 24.")
            else:
                print("Invalid study hours. Cannot exceed 24. Please enter a value between 0 and 24.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for study hours.")

def get_valid_assignment_score(prompt):
    while True:
        try:
            score = int(input(prompt))
            if 0 <= score <= 100:
                return score
            elif score < 0:
                print("Invalid assignment score. Cannot be negative. Please enter a value between 0 and 100.")
            else:
                print("Invalid assignment score. Cannot exceed 100. Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for assignment score.")

def get_valid_midterm_marks(prompt):
    while True:
        try:
            marks = int(input(prompt))
            if 0 <= marks <= 100:
                return marks
            elif marks < 0:
                print("Invalid midterm marks. Cannot be negative. Please enter a value between 0 and 100.")
            else:
                print("Invalid midterm marks. Cannot exceed 100. Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for midterm marks.")

def get_valid_previous_marks(prompt):
    while True:
        try:
            marks = int(input(prompt))
            if 0 <= marks <= 100:
                return marks
            elif marks < 0:
                print("Invalid previous marks. Cannot be negative. Please enter a value between 0 and 100.")
            else:
                print("Invalid previous marks. Cannot exceed 100. Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for previous marks.")

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
    if marks >= config.GRADE_A_MIN:
        return "A"
    elif marks >= config.GRADE_B_MIN:
        return "B"
    elif marks >= config.GRADE_C_MIN:
        return "C"
    elif marks >= config.GRADE_D_MIN:
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
    at_risk = student[(student["marks"] < config.AT_RISK_MARK) | (student["attendance"] < config.AT_RISK_ATTENDANCE)]
    if at_risk.empty:
        print("No at-risk students found.")
        return
    print("At-Risk Students:")
    for _, row in at_risk.iterrows():
        reasons = []
        if row["marks"] < config.AT_RISK_MARK:
            reasons.append(f"marks < {config.AT_RISK_MARK}")
        if row["attendance"] < config.AT_RISK_ATTENDANCE:
            reasons.append(f"attendance < {config.AT_RISK_ATTENDANCE}")
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
        if att >= config.ATTENDANCE_EXCELLENT_MIN:
            return "Excellent"
        elif att >= config.ATTENDANCE_GOOD_MIN:
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
    safe_show(None)
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
    safe_show(None)

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
    print("=" * 50)
    print(f"Individual Report for Roll {r}")
    print("=" * 50)
    print(f"Name           : {row['name']}")
    print(f"Roll           : {row['roll']}")
    print(f"Phone          : {row['phone']}")
    print(f"Attendance     : {row['attendance']}%")
    print(f"Study Hours    : {row['study_hours']}")
    print(f"Assignment Score: {row['assignment_score']}")
    print(f"Midterm Marks  : {row['midterm_marks']}")
    print(f"Previous Marks : {row['previous_marks']}")
    print(f"Marks          : {row['marks']}")
    print(f"Grade          : {row['grade']}")
    print("=" * 50)

# ------------------ New Phase 3 features ------------------

def ranking(student):
    if student.empty:
        print("No student available")
        return
    ranked = student.sort_values(by="marks", ascending=False).copy()
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    print(ranked.to_string(index=False))

def median_marks(student):
    if student.empty:
        print("No student available")
        return
    print(f"Median Marks: {student['marks'].median():.2f}")

def mode_marks(student):
    if student.empty:
        print("No student available")
        return
    modes = student["marks"].mode()
    if modes.empty:
        print("No mode available")
        return
    if len(modes) == 1:
        print(f"Mode Marks: {modes.iloc[0]}")
    else:
        print(f"Mode Marks: {', '.join(str(m) for m in modes.tolist())}")

def standard_deviation_marks(student):
    if student.empty:
        print("No student available")
        return
    print(f"Standard Deviation: {student['marks'].std():.2f}")

def quartiles_marks(student):
    if student.empty:
        print("No student available")
        return
    q = student["marks"].quantile([0.25, 0.5, 0.75])
    print("Quartiles:")
    print(f"  Q1 (25%): {q.iloc[0]:.2f}")
    print(f"  Q2 (50%): {q.iloc[1]:.2f}")
    print(f"  Q3 (75%): {q.iloc[2]:.2f}")

def statistical_summary(student):
    if student.empty:
        print("No student available")
        return
    print("Statistical Summary:")
    print(f"  Count    : {len(student)}")
    print(f"  Mean     : {student['marks'].mean():.2f}")
    median_marks(student)
    mode_marks(student)
    standard_deviation_marks(student)
    quartiles_marks(student)

def filter_by_grade(student, grade):
    graded = assign_grades(student)
    return graded[graded["grade"] == grade].copy()

def filter_by_attendance_category(student, category):
    if student.empty or "attendance" not in student.columns:
        return pd.DataFrame(columns=student.columns)
    graded = assign_grades(student)

    def cat(att):
        if att >= config.ATTENDANCE_EXCELLENT_MIN:
            return "Excellent"
        elif att >= config.ATTENDANCE_GOOD_MIN:
            return "Good"
        else:
            return "Low"

    result = graded.copy()
    result["attendance_category"] = result["attendance"].apply(cat)
    return result[result["attendance_category"] == category].copy()

def filter_at_risk_students(student):
    if student.empty or "attendance" not in student.columns:
        return pd.DataFrame(columns=student.columns)
    return student[(student["marks"] < config.AT_RISK_MARK) | (student["attendance"] < config.AT_RISK_ATTENDANCE)].copy()

def search_student(student, query):
    if student.empty:
        return pd.DataFrame(columns=student.columns)
    query = query.strip()
    if not query:
        return pd.DataFrame(columns=student.columns)
    try:
        roll_query = int(query)
        return student[student["roll"] == roll_query].copy()
    except ValueError:
        mask = student["name"].str.lower().str.contains(query.lower(), na=False)
        return student[mask].copy()

# ------------------ Existing features ------------------

def main():
    n = get_valid_menu_choice("enter the number of students::", 0, 10000)
    name = []
    roll = []
    marks = []
    phone = []
    attendance = []
    study_hours = []
    assignment_score = []
    midterm_marks = []
    previous_marks = []
    existing_rolls = []
    for i in range(n):
        student_name = get_valid_name("enter student name:")
        student_roll = get_valid_roll("enter student roll::", existing_rolls)
        existing_rolls.append(student_roll)
        student_total_marks = get_valid_marks("enter total marks:")
        student_attendance = get_valid_attendance("enter attendance percentage (0-100):")
        student_phone_number = get_valid_phone("enter student phone number::")
        student_study_hours = get_valid_study_hours("enter study hours per day (0-24):")
        student_assignment_score = get_valid_assignment_score("enter assignment score (0-100):")
        student_midterm_marks = get_valid_midterm_marks("enter midterm marks (0-100):")
        student_previous_marks = get_valid_previous_marks("enter previous marks (0-100):")
        name.append(student_name)
        roll.append(student_roll)
        marks.append(student_total_marks)
        phone.append(student_phone_number)
        attendance.append(student_attendance)
        study_hours.append(student_study_hours)
        assignment_score.append(student_assignment_score)
        midterm_marks.append(student_midterm_marks)
        previous_marks.append(student_previous_marks)
    student = pd.DataFrame({
        "name": name, "roll": roll, "marks": marks, "phone": phone,
        "attendance": attendance, "study_hours": study_hours,
        "assignment_score": assignment_score, "midterm_marks": midterm_marks,
        "previous_marks": previous_marks
    })
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
    if not ask_yes_no("if you want to add new students::"):
        print("thank you , no new students")
        return student
    n = get_valid_menu_choice("enter how many students you want to add::", 0, 10000)
    name = []
    roll = []
    marks = []
    phone = []
    attendance = []
    study_hours = []
    assignment_score = []
    midterm_marks = []
    previous_marks = []
    existing_rolls = student["roll"].tolist()
    for i in range(n):
        new_student_name = get_valid_name("enter student name:")
        new_student_roll = get_valid_roll("enter student roll::", existing_rolls)
        existing_rolls.append(new_student_roll)
        new_student_total_marks = get_valid_marks("enter total marks:")
        new_student_attendance = get_valid_attendance("enter attendance percentage (0-100):")
        new_student_phone_number = get_valid_phone("enter student phone number::")
        new_student_study_hours = get_valid_study_hours("enter study hours per day (0-24):")
        new_student_assignment_score = get_valid_assignment_score("enter assignment score (0-100):")
        new_student_midterm_marks = get_valid_midterm_marks("enter midterm marks (0-100):")
        new_student_previous_marks = get_valid_previous_marks("enter previous marks (0-100):")
        name.append(new_student_name)
        roll.append(new_student_roll)
        marks.append(new_student_total_marks)
        phone.append(new_student_phone_number)
        attendance.append(new_student_attendance)
        study_hours.append(new_student_study_hours)
        assignment_score.append(new_student_assignment_score)
        midterm_marks.append(new_student_midterm_marks)
        previous_marks.append(new_student_previous_marks)
    new_student = pd.DataFrame({
        "name": name, "roll": roll, "marks": marks, "phone": phone,
        "attendance": attendance, "study_hours": study_hours,
        "assignment_score": assignment_score, "midterm_marks": midterm_marks,
        "previous_marks": previous_marks
    })
    student = pd.concat([student, new_student], ignore_index=True)
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
    if not ask_yes_no("want to update::y/n"):
        print("everything is correct")
        print(student)
        return student
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
        print("6.update study hours")
        print("7.update assignment score")
        print("8.update midterm marks")
        print("9.update previous marks")
        choice = get_valid_menu_choice("enter choice::", 1, 9)
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
        elif choice == 6:
            new_sh = get_valid_study_hours("enter new study hours per day (0-24):")
            student.loc[student["roll"] == r, "study_hours"] = new_sh
        elif choice == 7:
            new_as = get_valid_assignment_score("enter new assignment score (0-100):")
            student.loc[student["roll"] == r, "assignment_score"] = new_as
        elif choice == 8:
            new_mm = get_valid_midterm_marks("enter new midterm marks (0-100):")
            student.loc[student["roll"] == r, "midterm_marks"] = new_mm
        elif choice == 9:
            new_pm = get_valid_previous_marks("enter new previous marks (0-100):")
            student.loc[student["roll"] == r, "previous_marks"] = new_pm
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
    safe_show(None)

def show_all_students(student):
    if student.empty:
        print("no student available")
        return
    print(student)

# ------------------ Main CLI ------------------

if __name__ == "__main__":
    database.init_db()
    student = database.load_students()
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
        print("19. STATISTICAL SUMMARY")
        print("20. SEARCH STUDENT")
        print("21. FILTER STUDENTS")
        print("22. TRAIN MARKS PREDICTION MODEL")
        print("23. COMPARE REGRESSION MODELS")
        print("24. EXIT")
        k = get_valid_menu_choice("enter your choice::", 1, MENU_MAX)
        if k == 1:
            student = main()
            database.save_students(student)
        elif k == 2:
            student = add(student)
            database.save_students(student)
        elif k == 3:
            print_max_marks_student(student)
        elif k == 4:
            print_min_marks_student(student)
        elif k == 5:
            student = update(student)
            database.save_students(student)
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
            database.save_students(student)
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
            statistical_summary(student)
        elif k == 20:
            if student.empty:
                print("no student available")
            else:
                query = input("enter roll number or name to search::")
                result = search_student(student, query)
                if result.empty:
                    print("No matching student found.")
                else:
                    print(result.to_string(index=False))
        elif k == 21:
            if student.empty:
                print("no student available")
            else:
                print("Filter by:")
                print("1. Grade")
                print("2. Attendance Category")
                print("3. At-Risk Students")
                sub = get_valid_menu_choice("enter choice::", 1, 3)
                if sub == 1:
                    grade = input("enter grade (A/B/C/D/F)::").strip().upper()
                    result = filter_by_grade(student, grade)
                    if result.empty:
                        print(f"No students found with grade {grade}.")
                    else:
                        print(result.to_string(index=False))
                elif sub == 2:
                    category = input("enter category (Excellent/Good/Low)::").strip().capitalize()
                    result = filter_by_attendance_category(student, category)
                    if result.empty:
                        print(f"No students found in category {category}.")
                    else:
                        print(result.to_string(index=False))
                elif sub == 3:
                    result = filter_at_risk_students(student)
                    if result.empty:
                        print("No at-risk students found.")
                    else:
                        print(result.to_string(index=False))
        elif k == 22:
            result = analytics.train_regression_model(student)
            if result is not None:
                print()
                print("=" * 50)
                print("Marks Prediction Model - Training Complete")
                print("=" * 50)
                print(f"Model: Linear Regression")
                print(f"Features: {result['features']}")
                print(f"Target: {result['target']}")
                print(f"Training samples: {len(result['X_train'])}")
                print(f"Testing samples: {len(result['X_test'])}")
                print()
                metrics = result["metrics"]
                print(f"MAE: {metrics['mae']:.2f}")
                print(f"MSE: {metrics['mse']:.2f}")
                print(f"RMSE: {metrics['rmse']:.2f}")
                print(f"R²: {metrics['r2']:.4f}")
                print("=" * 50)
                if ask_yes_no("Do you want to try a prediction? (y/n): "):
                    print("Enter values for:")
                    print(", ".join(analytics.FEATURE_COLUMNS))
                    values = []
                    for col in analytics.FEATURE_COLUMNS:
                        prompt = f"  {col}: "
                        values.append(input(prompt).strip())
                    pred = analytics.predict_marks(result["model"], values)
                    if pred is not None:
                        print(f"Predicted marks: {pred:.2f}")
        elif k == 23:
            if student.empty:
                print("no student available")
            else:
                print()
                print("=" * 60)
                print("REGRESSION MODEL COMPARISON")
                print("=" * 60)
                comparison = analytics.compare_regression_models(student)
                if comparison is not None:
                    print(comparison.to_string(index=False))
                    print()
                    cv_results = analytics.cross_validate_regression_models(student)
                    if cv_results is not None:
                        print("-" * 60)
                        print("CROSS-VALIDATION")
                        print("-" * 60)
                        print(cv_results.to_string(index=False))
                    print("=" * 60)
        elif k == 24:
            sys.exit()
