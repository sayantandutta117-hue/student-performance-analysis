import sqlite3
import pandas as pd
from pandas.errors import DatabaseError as PandasDatabaseError

DB_PATH = "students.db"

# Columns that may need to be added via migration
MIGRATION_COLUMNS = [
    ("study_hours", "REAL"),
    ("assignment_score", "INTEGER"),
    ("midterm_marks", "INTEGER"),
    ("previous_marks", "INTEGER"),
]


def init_db(db_path=DB_PATH):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    roll INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    marks INTEGER NOT NULL,
                    phone TEXT NOT NULL,
                    attendance INTEGER NOT NULL,
                    study_hours REAL,
                    assignment_score INTEGER,
                    midterm_marks INTEGER,
                    previous_marks INTEGER
                )
                """
            )
            conn.commit()
            _migrate_schema(conn)
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")


def _migrate_schema(conn):
    """
    Safely add new columns to an existing students table.
    This allows old databases to work without data loss.
    """
    try:
        cursor = conn.execute("PRAGMA table_info(students)")
        existing_columns = {row[1] for row in cursor.fetchall()}
    except sqlite3.Error:
        return

    for col_name, col_type in MIGRATION_COLUMNS:
        if col_name not in existing_columns:
            try:
                conn.execute(f"ALTER TABLE students ADD COLUMN {col_name} {col_type}")
                conn.commit()
            except sqlite3.Error:
                pass


def load_students(db_path=DB_PATH):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        _migrate_schema(conn)
        df = pd.read_sql_query(
            """
            SELECT name, roll, marks, phone, attendance,
                   study_hours, assignment_score, midterm_marks, previous_marks
            FROM students
            """,
            conn,
        )
    except (sqlite3.Error, PandasDatabaseError) as e:
        print(f"Database load error: {e}")
        return pd.DataFrame(columns=[
            "name", "roll", "marks", "phone", "attendance",
            "study_hours", "assignment_score", "midterm_marks", "previous_marks"
        ])
    finally:
        if conn:
            conn.close()

    expected_columns = [
        "name", "roll", "marks", "phone", "attendance",
        "study_hours", "assignment_score", "midterm_marks", "previous_marks"
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = pd.NA

    return df[expected_columns]


def save_students(student_df, db_path=DB_PATH):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM students")
            if not student_df.empty:
                expected_columns = [
                    "name", "roll", "marks", "phone", "attendance",
                    "study_hours", "assignment_score", "midterm_marks", "previous_marks"
                ]
                df = student_df.copy()
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = None
                df = df[expected_columns]
                # Replace pandas NA/NaN with None for SQLite compatibility
                df = df.where(pd.notna(df), None)
                conn.executemany(
                    """
                    INSERT INTO students
                    (name, roll, marks, phone, attendance,
                     study_hours, assignment_score, midterm_marks, previous_marks)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    df.values.tolist(),
                )
            conn.commit()
    except sqlite3.Error as e:
        print(f"Database save error: {e}")
        raise
