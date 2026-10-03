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
- Rank students by marks with explicit rank numbers
- Input validation for student names, marks, roll numbers, phone numbers, attendance, study hours, assignment scores, midterm marks, previous marks, and menu choices (prevents crashes from invalid input)
- Grade calculation (A, B, C, D, F):
  - A: 80-100
  - B: 70-79
  - C: 60-69
  - D: 50-59
  - F: 0-49
- Pass/Fail analysis with pass percentage
- Attendance tracking (0-100) with update support
- Study hours tracking (0-24 hours/day) with update support
- Assignment score tracking (0-100) with update support
- Midterm marks tracking (0-100) with update support
- Previous marks tracking (0-100) with update support
- Top 3 students report with rank, marks, and grade
- At-risk students report (marks < 40 OR attendance < 75) with reasons shown
- Attendance analysis with categories (Excellent: 90-100, Good: 75-89, Low: 0-74) and average marks
- Attendance vs Marks scatter plot with trend line and correlation coefficient
- Grade distribution counts with bar chart
- Individual student report by roll number
- Statistical summary: count, mean, median, mode, standard deviation, quartiles
- Search student by roll number (exact) or name (case-insensitive)
- Filter students by grade, attendance category, or at-risk status
- Centralized configuration for grade boundaries, pass mark, at-risk thresholds, and attendance categories
- Machine Learning: Linear Regression model predicts marks from multiple academic features
- Machine Learning: Compare multiple regression models (Linear, Decision Tree, Random Forest) with cross-validation
- SQLite persistence: student data survives application restarts (stored in `students.db`)

## Project Structure
- `main.py` — CLI menu, validation, analysis, and visualization functions
- `database.py` — SQLite persistence layer (init, load, save)
- `config.py` — Centralized academic thresholds (grades, pass mark, at-risk, attendance categories)
- `analytics.py` — Machine learning regression module (marks prediction)
- `test_validation.py` — Unit tests for all functionality
- `requirements.txt` — Python dependencies

## Technologies Used
- Python
- Pandas
- Numpy
- Matplotlib
- Scikit-learn

## How to Run
1. Clone this repository
2. Create a virtual environment (recommended):
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
3. Install dependencies using: pip install -r requirements.txt
4. Run: python main.py

## Data Persistence
- Student data is stored in a local SQLite database file: `students.db`
- The database and `students` table are created automatically on first run
- All add, update, and delete operations are persisted immediately
- Closing and reopening the application will reload existing students automatically
- Roll numbers are unique and cannot be duplicated
- Phone numbers are stored as text to preserve leading zeros

## Machine Learning - Marks Prediction
The application includes a Linear Regression model that predicts student marks from multiple academic features.

### What the model does
- **Target variable**: `marks` (the student's exam marks)
- **Features used**:
  - `attendance` (attendance percentage, 0-100)
  - `study_hours` (study hours per day, 0-24)
  - `assignment_score` (assignment score, 0-100)
  - `midterm_marks` (midterm exam marks, 0-100)
  - `previous_marks` (previous exam marks, 0-100)
- **Excluded features**: `name`, `roll`, `phone` are NOT used as ML features because they are not meaningful numeric predictors of academic performance

### How it works
1. The application validates that enough complete student records exist (minimum 5)
2. Records with missing academic features are excluded from training and a warning is shown
3. Data is split into training and testing sets (70% train, 30% test)
4. A Linear Regression model is trained on the training data
5. The model is evaluated on the held-out test data

### Evaluation metrics
- **MAE** (Mean Absolute Error): average absolute difference between predicted and actual marks
- **MSE** (Mean Squared Error): average squared difference
- **RMSE** (Root Mean Squared Error): square root of MSE, in the same units as marks
- **R²** (R-squared): how well the model explains variance (0 to 1, higher is better)

### Handling missing data
- Students from older database versions may have NULL values for the new academic fields
- The ML pipeline detects missing values and excludes incomplete records from training
- A warning message shows how many records were excluded
- The model trains only on complete records to avoid unreliable results

### Important limitations
- The model uses academic features as predictive indicators, not as proven causes of higher marks
- More student records and additional features would improve accuracy
- Predictions are estimates, not guaranteed outcomes
- The model should not be used for high-stakes decisions with very small datasets

### CLI usage
Select **"22. TRAIN MARKS PREDICTION MODEL"** from the menu to:
1. Train the model on current student data
2. View evaluation metrics
3. Optionally enter feature values to see a predicted mark

### Regression Model Comparison
Select **"23. COMPARE REGRESSION MODELS"** from the menu to compare multiple regression models on the same dataset.

#### Available models
- **Linear Regression** — basic linear model
- **Decision Tree Regression** — tree-based model
- **Random Forest Regression** — ensemble of decision trees

#### How comparison works
1. All models use the same cleaned dataset
2. All models use the same features:
   - `attendance`
   - `study_hours`
   - `assignment_score`
   - `midterm_marks`
   - `previous_marks`
3. All models use the same train/test split (70% train, 30% test, random_state=42)
4. Each model is trained independently and evaluated on the same held-out test set

#### Metrics compared
- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — R-squared

#### Cross-validation
The comparison also runs k-fold cross-validation (default 5-fold, reduced for small datasets):
- Reports RMSE mean and standard deviation
- Reports R² mean and standard deviation
- Uses the same random_state=42 for reproducibility

#### Important notes
- No model is automatically labeled as "best"
- Performance depends on dataset size, feature quality, and data distribution
- The same train/test split is used for fair comparison
- Cross-validation uses only the training data, not the test set

## Student Schema
The application stores the following fields for each student:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `name` | TEXT | - | Student name |
| `roll` | INTEGER | - | Unique roll number (primary key) |
| `marks` | INTEGER | 0-100 | Final exam marks |
| `phone` | TEXT | 10 digits | Contact phone number |
| `attendance` | INTEGER | 0-100 | Attendance percentage |
| `study_hours` | REAL | 0-24 | Study hours per day |
| `assignment_score` | INTEGER | 0-100 | Assignment score |
| `midterm_marks` | INTEGER | 0-100 | Midterm exam marks |
| `previous_marks` | INTEGER | 0-100 | Previous exam marks |

## Database Migration
- The application automatically migrates existing `students.db` files when new columns are added
- Old records without new academic fields are loaded with NULL values for those fields
- No data is deleted during migration
- Missing academic values are excluded from ML training with a warning message

## Running Tests
- Run all tests: python -m unittest test_validation.py -v
- Tests use a temporary SQLite database and do not modify `students.db`
