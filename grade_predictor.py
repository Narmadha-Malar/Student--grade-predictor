# ============================================
# STUDENT GRADE PREDICTOR - AI/ML Project
# By: Narmadha Malar K
# Tools: Python, Scikit-learn, Pandas
# ============================================

# STEP 1: Import Libraries
# Libraries = ready-made tools we can use directly

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Libraries loaded!")

# ============================================
# STEP 2: Create Student Data
# attendance = how many % classes attended
# midterm = midterm exam marks (out of 100)
# assignment = assignment marks (out of 100)
# result = 1 means Pass, 0 means Fail
# ============================================

data = {
    'attendance': [95, 85, 60, 45, 90, 30, 75, 55, 88, 40,
                   92, 70, 35, 80, 50, 65, 95, 25, 78, 88],

    'midterm':    [88, 76, 45, 30, 92, 20, 65, 40, 80, 25,
                   90, 60, 22, 72, 35, 55, 85, 15, 68, 84],

    'assignment': [90, 80, 50, 35, 88, 25, 70, 45, 85, 30,
                   92, 65, 28, 75, 40, 60, 88, 20, 72, 86],

    'result':     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
                   1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    # 1 = Pass, 0 = Fail
}

# Convert to DataFrame (like Excel table)
df = pd.DataFrame(data)

print(f"Total students: {len(df)}")
print(f"Pass: {df['result'].sum()}")
print(f"Fail: {len(df) - df['result'].sum()}")

# ============================================
# STEP 3: Separate Input and Output
# X = Input (attendance, midterm, assignment)
# y = Output (pass or fail)
# ============================================

X = df[['attendance', 'midterm', 'assignment']]  # Input columns
y = df['result']                                   # Output column

print("\nInput data ready!")

# ============================================
# STEP 4: Split Data into Train and Test
# 80% data = model will learn from this
# 20% data = we will test the model on this
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 20% for testing
    random_state=42    # fixed value so result is same every time
)

print(f"Training students: {len(X_train)}")
print(f"Testing students: {len(X_test)}")

# ============================================
# STEP 5: Train the Model
# We use Logistic Regression algorithm
# It is perfect for Pass/Fail predictions
# Model will learn the pattern from data
# ============================================

model = LogisticRegression()

# Train the model using training data
model.fit(X_train, y_train)

print("\nModel training done!")

# ============================================
# STEP 6: Check Accuracy
# We test on unseen students to check
# how good our model is
# ============================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.1f}%")

# ============================================
# STEP 7: Predict New Student Result
# Give any student details and
# model will say Pass or Fail
# ============================================

def predict_grade(attendance, midterm, assignment):
    # Create input as a table row
    student = pd.DataFrame([[attendance, midterm, assignment]],
                           columns=['attendance', 'midterm', 'assignment'])

    # Ask model to predict
    prediction = model.predict(student)[0]

    # Return result
    if prediction == 1:
        return "PASS - Student will pass! ✅"
    else:
        return "FAIL - Student needs help! ❌"

# Test with new students
print("\n--- Predicting New Students ---")

students = [
    (90, 85, 88),   # Good student
    (40, 25, 30),   # Weak student
    (75, 65, 70),   # Average student
    (20, 15, 20),   # Very weak student
]

for attendance, midterm, assignment in students:
    result = predict_grade(attendance, midterm, assignment)
    print(f"\nAttendance: {attendance}% | Midterm: {midterm} | Assignment: {assignment}")
    print(f"Prediction: {result}")

print("\nProject Complete!")
