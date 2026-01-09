import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data.csv")

# Clean column names (remove spaces, make consistent)
data.columns = data.columns.str.strip()

# DEBUG: show columns (safe to keep)
print("Columns found:", data.columns.tolist())

# ---- AUTO FIND FINAL GRADE COLUMN ----
# Look for column that contains 'G3' or 'final'
final_col = None
for col in data.columns:
    if col.lower() == "g3" or "final" in col.lower():
        final_col = col
        break

if final_col is None:
    raise Exception("Final grade column (G3) not found in dataset")

# Create Pass / Fail target
data["result"] = data[final_col].apply(lambda x: 1 if x >= 10 else 0)

# Use early grades as features (avoid leakage)
feature_cols = []
for col in data.columns:
    if col.lower() in ["g1", "g2"]:
        feature_cols.append(col)

if len(feature_cols) < 2:
    raise Exception("G1 and G2 columns not found")

X = data[feature_cols]
y = data["result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# ---- TEST WITH A NEW STUDENT ----
# Example student marks
new_student = pd.DataFrame([[10, 12]], columns=["G1", "G2"])
prediction = model.predict(new_student)


if prediction[0] == 1:
    print("Prediction for new student: PASS")
else:
    print("Prediction for new student: FAIL")