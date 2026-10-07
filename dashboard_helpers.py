import pandas as pd
import main as student_main
import config


def compute_kpis(student):
    if student.empty:
        return {
            "total_students": 0,
            "average_marks": None,
            "pass_count": 0,
            "pass_rate": None,
            "fail_count": 0,
            "average_attendance": None,
        }
    graded = student_main.assign_grades(student)
    total = int(len(graded))
    avg_marks = float(graded["marks"].mean())
    passed = graded[graded["grade"] != "F"]
    pass_count = int(len(passed))
    fail_count = total - pass_count
    pass_rate = (pass_count / total) * 100.0 if total > 0 else 0.0
    avg_attendance = float(graded["attendance"].mean()) if "attendance" in graded.columns else None
    return {
        "total_students": total,
        "average_marks": avg_marks,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "fail_count": fail_count,
        "average_attendance": avg_attendance,
    }


def get_grade_distribution_df(student):
    if student.empty:
        return pd.DataFrame(columns=["Grade", "Count"])
    graded = student_main.assign_grades(student)
    counts = graded["grade"].value_counts().sort_index().reset_index()
    counts.columns = ["Grade", "Count"]
    return counts


def get_at_risk_df(student):
    if student.empty or "attendance" not in student.columns:
        return pd.DataFrame(columns=student.columns)
    return student_main.filter_at_risk_students(student)


def get_ranked_df(student):
    if student.empty:
        return pd.DataFrame(columns=["Rank"] + list(student.columns))
    ranked = student.sort_values(by="marks", ascending=False).copy()
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    return ranked


def search_students(student, query):
    return student_main.search_student(student, query)


def filter_by_grade(student, grade):
    if student.empty or "marks" not in student.columns:
        return pd.DataFrame(columns=student.columns)
    return student_main.filter_by_grade(student, grade)


def filter_by_attendance_category(student, category):
    return student_main.filter_by_attendance_category(student, category)


def get_descriptive_stats(student):
    if student.empty:
        return {}
    return {
        "count": int(len(student)),
        "mean": float(student["marks"].mean()),
        "median": float(student["marks"].median()),
        "std": float(student["marks"].std()),
        "q1": float(student["marks"].quantile(0.25)),
        "q2": float(student["marks"].quantile(0.50)),
        "q3": float(student["marks"].quantile(0.75)),
    }


def get_attendance_category_counts(student):
    if student.empty or "attendance" not in student.columns:
        return pd.DataFrame(columns=["attendance_category", "Count"])
    graded = student_main.assign_grades(student)

    def category(att):
        if att >= config.ATTENDANCE_EXCELLENT_MIN:
            return "Excellent"
        elif att >= config.ATTENDANCE_GOOD_MIN:
            return "Good"
        else:
            return "Low"

    result = graded.copy()
    result["attendance_category"] = result["attendance"].apply(category)
    counts = result["attendance_category"].value_counts().sort_index().reset_index()
    counts.columns = ["attendance_category", "Count"]
    return counts
