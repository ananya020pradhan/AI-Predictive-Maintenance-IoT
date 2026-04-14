# src/train.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
print("📂 Loading dataset...")
df = pd.read_csv("data/sensor_data.csv")

print("\n🔍 Dataset Preview:")
print(df.head())

# -------------------------------
# Step 2: Data Cleaning
# -------------------------------
print("\n🧹 Cleaning data...")
df.fillna(df.mean(), inplace=True)

# -------------------------------
# Step 3: Features & Target
# -------------------------------
X = df.drop("failure", axis=1)
y = df["failure"]

# -------------------------------
# Step 4: Scaling
# -------------------------------
print("\n⚙️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 6: Model Training
# -------------------------------
print("\n🤖 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Step 7: Evaluation
# -------------------------------
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Step 8: Save Model & Scaler
# -------------------------------
print("\n💾 Saving model...")

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ model.pkl and scaler.pkl saved in /models folder")

# -------------------------------
# Step 9: Test Prediction
# -------------------------------
print("\n🔮 Testing prediction...")

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)[0]

if prediction == 1:
    print("⚠️ ALERT: Machine Failure Predicted!")
else:
    print("✅ Machine is Safe")