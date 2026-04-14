# 🔧 AI-Powered Predictive Maintenance System for IoT Devices

## 🚀 Overview

This project is an end-to-end Machine Learning system that predicts machine failures using IoT sensor data such as temperature, vibration, and pressure.

The system helps industries perform **predictive maintenance**, reducing downtime, improving efficiency, and saving costs.

---

## 🎯 Problem Statement

Traditional maintenance strategies:

* Reactive (after failure) ❌
* Scheduled (fixed intervals) ❌

This project provides:

* Predictive maintenance using AI ✅
* Early failure detection ✅
* Smart decision-making support ✅

---

## 🏭 Industry Relevance

This system can be applied in:

* Manufacturing plants
* Factories
* Power plants
* Automotive industry
* Aviation systems

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit (for deployment)
* Joblib (model saving)

---

## 📂 Project Structure

AI-Predictive-Maintenance-IoT/
│
├── data/              # Sensor dataset
├── models/            # Saved ML model & scaler
├── src/               # Training scripts
├── images/            # Output graphs
├── main.py            # Model training pipeline
├── app.py             # Streamlit web app
├── requirements.txt   # Dependencies
└── README.md

---

## 🔄 Workflow

1. Load sensor dataset
2. Data preprocessing & cleaning
3. Feature scaling
4. Model training (Random Forest)
5. Failure prediction
6. Visualization
7. Deployment using Streamlit

---

## 🧪 Dataset

The dataset simulates IoT sensor readings:

* Temperature
* Vibration
* Pressure
* Failure (Target: 0/1)

---

## ⚡ Installation & Setup

### Step 1: Clone Repository

git clone https://github.com/your-username/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT

---

### Step 2: Create Virtual Environment

python -m venv venv

Activate:
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

---

### Step 3: Install Dependencies

pip install -r requirements.txt

---

## ▶️ How to Run

### Step 1: Train Model

python main.py

This will generate:

* models/model.pkl
* models/scaler.pkl

---

### Step 2: Run Web App

streamlit run app.py

---

## 🌐 Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to Streamlit Cloud
3. Select your repository
4. Choose app.py
5. Click Deploy

---

## 📊 Features

* Machine failure prediction
* Real-time input sliders
* Probability-based output
* Alert system (⚠️ / ✅)
* Visualization charts

---

## 📸 Sample Outputs

* Confusion Matrix
* Prediction Graph
* Failure Alerts

(Add screenshots inside the /images folder)

---

## 🎓 Learning Outcomes

* End-to-end ML pipeline
* Feature engineering & scaling
* Model training & evaluation
* Model deployment
* GitHub project structuring

---

## 🔥 Future Improvements

* Real-time IoT data integration
* LSTM-based time-series prediction
* Cloud deployment (AWS/GCP)
* Live industrial dashboard

---

## 👨‍💻 Author

Ananya Pradhan



---

## ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 📢 Share it

---
