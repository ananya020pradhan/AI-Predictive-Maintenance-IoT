# ==============================
# AI Predictive Maintenance - main.py
# ==============================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("data/sensor_data.csv")

print("\nDataset Preview:")
print(df.head())

# ------------------------------
# 2. Handle Missing Values (ONLY numeric)
# ------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# ------------------------------
# 3. Encode Categorical Columns
# ------------------------------
df = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding Columns:")
print(df.columns)

# ------------------------------
# 4. Define Target Column
# ------------------------------
target_col = "faulty"

if target_col not in df.columns:
    raise Exception("Target column 'faulty' not found in dataset!")

X = df.drop(target_col, axis=1)
y = df[target_col]

# ------------------------------
# 5. Check Class Distribution
# ------------------------------
print("\nClass Distribution:")
print(y.value_counts())

# ------------------------------
# 6. Feature Scaling
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 7. Train-Test Split (IMPORTANT: stratify)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------
# 8. Model Training
# ------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------
# 9. Predictions
# ------------------------------
y_pred = model.predict(X_test)

# ------------------------------
# 10. Evaluation
# ------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# 11. Confusion Matrix (FIXED)
# ------------------------------
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:\n", cm)
print("Shape:", cm.shape)

# Create folder
os.makedirs("models", exist_ok=True)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("models/confusion_matrix.png")
plt.close()

# ------------------------------
# 12. Save Model & Scaler
# ------------------------------
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns, "models/features.pkl")

print("\nModel and Scaler saved successfully!")

# ------------------------------
# 13. Sample Prediction Test
# ------------------------------
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)[0]

print("\nSample Prediction:")

if prediction == 1:
    print("⚠️ Machine Failure Predicted!")
else:
    print("✅ Machine is Healthy")