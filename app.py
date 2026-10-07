import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database

database.DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "students.db")

import analytics
import config
import main as student_main
from dashboard_helpers import (
    compute_kpis,
    get_grade_distribution_df,
    get_at_risk_df,
    get_ranked_df,
    search_students,
    filter_by_grade,
    filter_by_attendance_category,
    get_descriptive_stats,
    get_attendance_category_counts,
)
from prediction_helpers import (
    train_specific_regression_model,
    train_specific_classification_model,
    get_model_explanation,
    validate_prediction_inputs,
)

st.set_page_config(page_title="Student Performance Analysis", layout="wide")

database.init_db()

PAGE_DASHBOARD = "Dashboard Overview"
PAGE_STUDENT_MGMT = "Student Management"
PAGE_ANALYTICS = "Student Analytics"
PAGE_REGRESSION = "Regression Lab"
PAGE_CLASSIFICATION = "Classification Lab"
PAGE_PREDICTION = "Prediction Lab"

page = st.sidebar.radio(
    "Navigation",
    [PAGE_DASHBOARD, PAGE_STUDENT_MGMT, PAGE_ANALYTICS, PAGE_REGRESSION, PAGE_CLASSIFICATION, PAGE_PREDICTION],
)


def get_student_df():
    return database.load_students()


def page_dashboard():
    st.header("Dashboard Overview")
    student = get_student_df()

    if student.empty:
        st.info("No student records found. Add students to see dashboard metrics.")
        return

    kpis = compute_kpis(student)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Students", kpis["total_students"])
    col2.metric("Average Marks", f"{kpis['average_marks']:.2f}")
    col3.metric("Pass Count", kpis["pass_count"])
    col4.metric("Pass Rate", f"{kpis['pass_rate']:.1f}%")
    col5.metric("Fail Count", kpis["fail_count"])

    if kpis["average_attendance"] is not None:
        st.metric("Average Attendance", f"{kpis['average_attendance']:.1f}%")

    st.subheader("Grade Distribution")
    grade_df = get_grade_distribution_df(student)
    if not grade_df.empty:
        st.bar_chart(grade_df.set_index("Grade"))
    else:
        st.info("No grade data available.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Marks Distribution")
        fig, ax = plt.subplots()
        ax.hist(student["marks"].dropna(), bins=10, color="skyblue", edgecolor="black")
        ax.set_xlabel("Marks")
        ax.set_ylabel("Count")
        ax.set_title("Marks Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Attendance vs Marks")
        if "attendance" in student.columns and student["attendance"].notna().any():
            fig, ax = plt.subplots()
            plot_df = student[["attendance", "marks"]].dropna()
            x = plot_df["attendance"].values
            y = plot_df["marks"].values
            ax.scatter(x, y, color="blue")
            ax.set_xlabel("Attendance (%)")
            ax.set_ylabel("Marks")
            ax.set_title("Attendance vs Marks")
            if len(x) > 1:
                sorted_idx = np.argsort(x)
                x_sorted = x[sorted_idx]
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x_sorted, p(x_sorted), "r--")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Attendance data not available.")

    st.subheader("Student Performance Summary")
    graded = student_main.assign_grades(student)
    st.dataframe(
        graded[["name", "roll", "marks", "grade", "attendance", "study_hours"]],
        use_container_width=True,
    )


def page_student_management():
    st.header("Student Management")
    student = get_student_df()

    action = st.sidebar.selectbox(
        "Action",
        ["View Students", "Add Student", "Update Student", "Delete Student", "Search Students", "Filter Students"],
    )

    if action == "View Students":
        if student.empty:
            st.info("No student records found.")
        else:
            st.dataframe(student, use_container_width=True)

    elif action == "Add Student":
        with st.form("add_student_form", clear_on_submit=True):
            st.subheader("Add New Student")
            name = st.text_input("Name")
            roll = st.number_input("Roll Number", min_value=1, step=1)
            marks = st.number_input("Marks (0-100)", min_value=0, max_value=100, step=1)
            phone = st.text_input("Phone (10 digits)")
            attendance = st.number_input("Attendance (0-100)", min_value=0, max_value=100, step=1)
            study_hours = st.number_input("Study Hours per Day (0-24)", min_value=0.0, max_value=24.0, step=0.5)
            assignment_score = st.number_input("Assignment Score (0-100)", min_value=0, max_value=100, step=1)
            midterm_marks = st.number_input("Midterm Marks (0-100)", min_value=0, max_value=100, step=1)
            previous_marks = st.number_input("Previous Marks (0-100)", min_value=0, max_value=100, step=1)
            submitted = st.form_submit_button("Add Student")

            if submitted:
                errors = []
                if not name.strip():
                    errors.append("Name cannot be empty.")
                if not phone.isdigit() or len(phone) != 10:
                    errors.append("Phone must be exactly 10 digits.")
                if int(roll) in student["roll"].values:
                    errors.append(f"Roll number {roll} already exists.")
                
                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    new_row = {
                        "name": name.strip(),
                        "roll": int(roll),
                        "marks": int(marks),
                        "phone": phone,
                        "attendance": int(attendance),
                        "study_hours": float(study_hours),
                        "assignment_score": int(assignment_score),
                        "midterm_marks": int(midterm_marks),
                        "previous_marks": int(previous_marks),
                    }
                    updated = pd.concat([student, pd.DataFrame([new_row])], ignore_index=True)
                    try:
                        database.save_students(updated)
                        st.success(f"Student '{name}' added successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save student: {e}")

    elif action == "Update Student":
        if student.empty:
            st.info("No student records found.")
            return

        with st.form("update_student_form"):
            st.subheader("Update Student")
            roll_options = {f"{row['roll']} - {row['name']}": row["roll"] for _, row in student.iterrows()}
            selected_label = st.selectbox("Select Student", list(roll_options.keys()))
            selected_roll = roll_options[selected_label]
            row = student[student["roll"] == selected_roll].iloc[0]

            field = st.selectbox("Field to Update", [
                "name", "marks", "phone", "attendance", "study_hours",
                "assignment_score", "midterm_marks", "previous_marks"
            ])
            value = st.text_input("New Value")
            submitted = st.form_submit_button("Update")

            if submitted:
                if not value.strip():
                    st.error("Value cannot be empty.")
                else:
                    updated = student.copy()
                    if field == "name":
                        updated.loc[updated["roll"] == selected_roll, "name"] = value.strip()
                    elif field == "phone":
                        if not value.isdigit() or len(value) != 10:
                            st.error("Phone must be exactly 10 digits.")
                            return
                        updated.loc[updated["roll"] == selected_roll, "phone"] = value
                    elif field in ["marks", "attendance", "assignment_score", "midterm_marks", "previous_marks"]:
                        try:
                            val = int(value)
                            if field == "marks" and not (0 <= val <= 100):
                                st.error("Marks must be between 0 and 100.")
                                return
                            if field == "attendance" and not (0 <= val <= 100):
                                st.error("Attendance must be between 0 and 100.")
                                return
                            if field == "assignment_score" and not (0 <= val <= 100):
                                st.error("Assignment score must be between 0 and 100.")
                                return
                            if field == "midterm_marks" and not (0 <= val <= 100):
                                st.error("Midterm marks must be between 0 and 100.")
                                return
                            if field == "previous_marks" and not (0 <= val <= 100):
                                st.error("Previous marks must be between 0 and 100.")
                                return
                            updated.loc[updated["roll"] == selected_roll, field] = val
                        except ValueError:
                            st.error("Invalid numeric value.")
                            return
                    elif field == "study_hours":
                        try:
                            val = float(value)
                            if not (0 <= val <= 24):
                                st.error("Study hours must be between 0 and 24.")
                                return
                            updated.loc[updated["roll"] == selected_roll, "study_hours"] = val
                        except ValueError:
                            st.error("Invalid numeric value.")
                            return
                    try:
                        database.save_students(updated)
                        st.success("Student updated successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update student: {e}")

    elif action == "Delete Student":
        if student.empty:
            st.info("No student records found.")
            return

        st.subheader("Delete Student")
        roll_options = {f"{row['roll']} - {row['name']}": row["roll"] for _, row in student.iterrows()}
        selected_label = st.selectbox("Select Student to Delete", list(roll_options.keys()))
        selected_roll = roll_options[selected_label]
        confirm = st.checkbox(f"Confirm deletion of roll number {selected_roll}")

        if st.button("Delete", type="primary"):
            if confirm:
                updated = student[student["roll"] != selected_roll].reset_index(drop=True)
                try:
                    database.save_students(updated)
                    st.success("Student deleted successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete student: {e}")
            else:
                st.warning("Please confirm the deletion.")

    elif action == "Search Students":
        st.subheader("Search Students")
        query = st.text_input("Search by roll number or name")
        if query:
            results = search_students(student, query)
            if results.empty:
                st.info("No matching students found.")
            else:
                st.dataframe(results, use_container_width=True)

    elif action == "Filter Students":
        st.subheader("Filter Students")
        filter_type = st.selectbox("Filter By", ["Grade", "Attendance Category", "At-Risk Students"])

        if filter_type == "Grade":
            grade = st.selectbox("Grade", ["A", "B", "C", "D", "F"])
            results = filter_by_grade(student, grade)
            if results.empty:
                st.info(f"No students found with grade {grade}.")
            else:
                st.dataframe(results, use_container_width=True)

        elif filter_type == "Attendance Category":
            category = st.selectbox("Category", ["Excellent", "Good", "Low"])
            results = filter_by_attendance_category(student, category)
            if results.empty:
                st.info(f"No students found in category {category}.")
            else:
                st.dataframe(results, use_container_width=True)

        elif filter_type == "At-Risk Students":
            results = get_at_risk_df(student)
            if results.empty:
                st.info("No at-risk students found.")
            else:
                st.dataframe(results, use_container_width=True)


def page_analytics():
    st.header("Student Analytics")
    student = get_student_df()

    if student.empty:
        st.info("No student records found.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Grade Distribution", "Rankings", "At-Risk Students",
        "Attendance Categories", "Descriptive Statistics", "Search & Filter"
    ])

    with tab1:
        st.subheader("Grade Distribution")
        grade_df = get_grade_distribution_df(student)
        if not grade_df.empty:
            st.bar_chart(grade_df.set_index("Grade"))
            st.dataframe(grade_df, use_container_width=True)
        else:
            st.info("No grade data available.")

    with tab2:
        st.subheader("Student Rankings")
        ranked = get_ranked_df(student)
        st.dataframe(ranked, use_container_width=True)

    with tab3:
        st.subheader("At-Risk Students")
        at_risk = get_at_risk_df(student)
        if at_risk.empty:
            st.info("No at-risk students found.")
        else:
            st.dataframe(at_risk, use_container_width=True)

    with tab4:
        st.subheader("Attendance Categories")
        att_cat = get_attendance_category_counts(student)
        if not att_cat.empty:
            st.bar_chart(att_cat.set_index("attendance_category"))
            st.dataframe(att_cat, use_container_width=True)
        else:
            st.info("Attendance data not available.")

    with tab5:
        st.subheader("Descriptive Statistics")
        stats = get_descriptive_stats(student)
        if stats:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Count", stats["count"])
            col2.metric("Mean", f"{stats['mean']:.2f}")
            col3.metric("Median", f"{stats['median']:.2f}")
            col4.metric("Std Dev", f"{stats['std']:.2f}")
            col5.metric("Q1", f"{stats['q1']:.2f}")
            col6.metric("Q3", f"{stats['q3']:.2f}")

    with tab6:
        st.subheader("Search Students")
        query = st.text_input("Enter roll number or name")
        if query:
            results = search_students(student, query)
            if results.empty:
                st.info("No matching students found.")
            else:
                st.dataframe(results, use_container_width=True)

        st.subheader("Filter Students")
        filter_type = st.selectbox("Filter By", ["Grade", "Attendance Category", "At-Risk Students"])

        if filter_type == "Grade":
            grade = st.selectbox("Select Grade", ["A", "B", "C", "D", "F"])
            results = filter_by_grade(student, grade)
            st.dataframe(results, use_container_width=True)
        elif filter_type == "Attendance Category":
            category = st.selectbox("Select Category", ["Excellent", "Good", "Low"])
            results = filter_by_attendance_category(student, category)
            st.dataframe(results, use_container_width=True)
        elif filter_type == "At-Risk Students":
            results = get_at_risk_df(student)
            st.dataframe(results, use_container_width=True)


def page_regression():
    st.header("Regression Lab")
    student = get_student_df()

    if st.button("Compare Regression Models", type="primary"):
        with st.spinner("Training and comparing models..."):
            comparison = analytics.compare_regression_models(student)
            if comparison is None:
                st.error(
                    "Cannot run regression: need at least 5 complete student records "
                    "with all academic features (attendance, study_hours, assignment_score, "
                    "midterm_marks, previous_marks)."
                )
            else:
                st.subheader("Model Comparison")
                st.dataframe(comparison, use_container_width=True)

                cv_results = analytics.cross_validate_regression_models(student)
                if cv_results is not None:
                    st.subheader("Cross-Validation Results")
                    st.dataframe(cv_results, use_container_width=True)
                else:
                    st.info("Cross-validation could not be performed with the current data.")

                st.caption(
                    "Note: No model is universally best. Performance depends on dataset size, "
                    "feature quality, and data distribution."
                )


def page_classification():
    st.header("Classification Lab")
    student = get_student_df()

    if st.button("Compare Classification Models", type="primary"):
        with st.spinner("Training and comparing classifiers..."):
            comparison = analytics.compare_classification_models(student)
            if comparison is None:
                st.error(
                    "Cannot run classification: need at least 5 complete student records with "
                    "both Pass and Fail examples."
                )
            else:
                st.subheader("Model Comparison")
                st.dataframe(comparison, use_container_width=True)

                trained = analytics.train_classification_model(student)
                if trained is not None:
                    cm = analytics.calculate_confusion_matrix(trained["y_test"], trained["y_pred"])
                    st.subheader("Confusion Matrix (Logistic Regression)")
                    cm_df = pd.DataFrame({
                        "Predicted Fail": [cm["tn"], cm["fn"]],
                        "Predicted Pass": [cm["fp"], cm["tp"]],
                    }, index=["Actual Fail", "Actual Pass"])
                    st.dataframe(cm_df, use_container_width=True)
                    st.caption("Fail = negative class (0), Pass = positive class (1)")

                cv_results = analytics.cross_validate_classification_models(student)
                if cv_results is not None:
                    st.subheader("Cross-Validation Results")
                    st.dataframe(cv_results, use_container_width=True)
                else:
                    st.info("Cross-validation could not be performed with the current data.")

                st.caption(
                    "Note: No model is universally best. Performance depends on dataset size, "
                    "feature quality, and data distribution."
                )


def page_prediction():
    st.header("Prediction Lab")
    student = get_student_df()

    if student.empty:
        st.info("No student records found.")
        return

    st.caption(
        "Predictions are model estimates. Feature importance and coefficients describe "
        "patterns learned by the model. They do not prove that a feature causes "
        "a student's performance to change."
    )

    tab1, tab2 = st.tabs(["Regression Prediction", "Classification Prediction"])

    with tab1:
        st.subheader("Predict Marks")
        model_name = st.selectbox(
            "Select Regression Model",
            ["Linear Regression", "Decision Tree", "Random Forest"],
        )

        with st.form("regression_prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                attendance = st.number_input("Attendance (0-100)", min_value=0, max_value=100, value=75)
                study_hours = st.number_input("Study Hours per Day (0-24)", min_value=0.0, max_value=24.0, value=5.0)
            with col2:
                assignment_score = st.number_input("Assignment Score (0-100)", min_value=0, max_value=100, value=70)
                midterm_marks = st.number_input("Midterm Marks (0-100)", min_value=0, max_value=100, value=70)
            with col3:
                previous_marks = st.number_input("Previous Marks (0-100)", min_value=0, max_value=100, value=70)
            submitted = st.form_submit_button("Predict Marks")

        if submitted:
            with st.spinner("Training model and predicting..."):
                result = train_specific_regression_model(student, model_name)
                if result is None:
                    st.error(
                        "Cannot train model: need at least 5 complete student records "
                        "with all academic features."
                    )
                else:
                    values = [attendance, study_hours, assignment_score, midterm_marks, previous_marks]
                    pred = analytics.predict_marks(result["model"], values)
                    if pred is None:
                        st.error("Prediction failed due to invalid input.")
                    else:
                        st.subheader("Predicted Marks")
                        st.metric("Marks", f"{pred:.1f} / 100")
                        st.caption(f"Model: {model_name} | Training records: {result['used_rows']}")

                        explanation = get_model_explanation(result["model"], model_name, "regression")
                        if explanation is not None:
                            st.subheader("Model Interpretation")
                            if explanation["type"] == "coefficients":
                                st.write("Model Coefficients")
                                coef_df = pd.DataFrame({
                                    "Feature": explanation["features"],
                                    "Coefficient": explanation["values"],
                                })
                                st.dataframe(coef_df, use_container_width=True)
                                st.caption(
                                    "A positive coefficient indicates the model's prediction increases "
                                    "as that feature increases, holding other features constant. "
                                    "A negative coefficient indicates the opposite. "
                                    "Coefficients describe the fitted model relationship and do not prove causation."
                                )
                            else:
                                st.write("Feature Importance")
                                imp_df = pd.DataFrame({
                                    "Feature": explanation["features"],
                                    "Importance": explanation["values"],
                                }).sort_values("Importance", ascending=False)
                                st.dataframe(imp_df, use_container_width=True)
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="skyblue")
                                ax.set_xlabel("Importance")
                                ax.set_title("Feature Importance")
                                st.pyplot(fig)
                                plt.close(fig)
                                st.caption(
                                    "Feature importance values describe patterns learned by the model. "
                                    "They do not prove that a feature causes a student's performance to change."
                                )

    with tab2:
        st.subheader("Predict Pass/Fail")
        model_name = st.selectbox(
            "Select Classification Model",
            ["Logistic Regression", "Decision Tree", "Random Forest"],
        )

        with st.form("classification_prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                attendance = st.number_input("Attendance (0-100)", min_value=0, max_value=100, value=75, key="cls_attendance")
                study_hours = st.number_input("Study Hours per Day (0-24)", min_value=0.0, max_value=24.0, value=5.0, key="cls_study_hours")
            with col2:
                assignment_score = st.number_input("Assignment Score (0-100)", min_value=0, max_value=100, value=70, key="cls_assignment_score")
                midterm_marks = st.number_input("Midterm Marks (0-100)", min_value=0, max_value=100, value=70, key="cls_midterm_marks")
            with col3:
                previous_marks = st.number_input("Previous Marks (0-100)", min_value=0, max_value=100, value=70, key="cls_previous_marks")
            submitted = st.form_submit_button("Predict Pass/Fail")

        if submitted:
            with st.spinner("Training model and predicting..."):
                result = train_specific_classification_model(student, model_name)
                if result is None:
                    st.error(
                        "Cannot train classifier: need at least 5 complete student records with "
                        "both Pass and Fail examples."
                    )
                else:
                    values = [attendance, study_hours, assignment_score, midterm_marks, previous_marks]
                    pred = analytics.predict_pass_fail(result["model"], values)
                    if pred is None:
                        st.error("Prediction failed due to invalid input.")
                    else:
                        st.subheader("Prediction")
                        st.metric("Predicted Outcome", pred)

                        if hasattr(result["model"], "predict_proba"):
                            try:
                                proba = result["model"].predict_proba([values])[0]
                                pass_proba = float(proba[1])
                                st.metric("Probability of Pass", f"{pass_proba * 100:.1f}%")
                            except Exception:
                                pass

                        st.caption(f"Model: {model_name} | Training records: {result['used_rows']}")

                        explanation = get_model_explanation(result["model"], model_name, "classification")
                        if explanation is not None:
                            st.subheader("Model Interpretation")
                            if explanation["type"] == "coefficients":
                                st.write("Model Coefficients")
                                coef_df = pd.DataFrame({
                                    "Feature": explanation["features"],
                                    "Coefficient": explanation["values"],
                                })
                                st.dataframe(coef_df, use_container_width=True)
                                st.caption(
                                    "Coefficients describe the fitted model relationship related to "
                                    "the predicted class. They do not prove that a feature causes "
                                    "a student's performance to change."
                                )
                            else:
                                st.write("Feature Importance")
                                imp_df = pd.DataFrame({
                                    "Feature": explanation["features"],
                                    "Importance": explanation["values"],
                                }).sort_values("Importance", ascending=False)
                                st.dataframe(imp_df, use_container_width=True)
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="skyblue")
                                ax.set_xlabel("Importance")
                                ax.set_title("Feature Importance")
                                st.pyplot(fig)
                                plt.close(fig)
                                st.caption(
                                    "Feature importance values describe patterns learned by the model. "
                                    "They do not prove that a feature causes a student's performance to change."
                                )


if page == PAGE_DASHBOARD:
    page_dashboard()
elif page == PAGE_STUDENT_MGMT:
    page_student_management()
elif page == PAGE_ANALYTICS:
    page_analytics()
elif page == PAGE_REGRESSION:
    page_regression()
elif page == PAGE_CLASSIFICATION:
    page_classification()
elif page == PAGE_PREDICTION:
    page_prediction()
