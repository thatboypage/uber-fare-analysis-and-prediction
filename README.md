# 🚕 Uber Fare Analysis and Prediction

This project analyzes and predicts Uber ride fares in New York City using historical ride data. It includes a complete machine learning pipeline — from raw data cleaning and feature engineering to model training, evaluation, and deployment using a Streamlit web app.

---

## 📁 Project Structure

```

Uber Fare Analysis and Prediction/
│
├── data/
│   ├── uber.csv                # Raw dataset from Kaggle
│   ├── uber\_cleaned.csv        # Cleaned dataset (intermediate)
│   └── uber\_cleaned\_final.csv  # Final dataset used for modeling
│
├── models/
│   └── xgb\_fare\_model.pkl      # Trained XGBoost model (serialized)
│
├── utils/
│   └── xgb\_fare\_model.json     # Model feature order (for app use)
│
├── notebooks/
│   └── uber\_fare\_prediction.ipynb   # Jupyter notebook with full analysis
│
├── Uber\_fare\_streamlitApp.py   # Streamlit app file
├── requirements.txt            # List of dependencies
└── README.md                   # Project overview and instructions

```

---

## 📊 Dataset Overview

The dataset contains Uber trip records in NYC between 2009 and 2015. Key columns include:

- `pickup_datetime`: Timestamp of ride
- `pickup_longitude`, `pickup_latitude`: Pickup coordinates
- `dropoff_longitude`, `dropoff_latitude`: Drop-off coordinates
- `passenger_count`: Number of passengers
- `fare_amount`: Fare charged (target variable)

---

## 🧹 Data Cleaning and Feature Engineering

Steps taken:

- Removed invalid coordinates and fare outliers
- Converted `pickup_datetime` to datetime format
- Engineered new features:
  - `distance` between pickup and dropoff points using Haversine formula
  - `is_day`, `is_night` based on time of day
  - `day_of_week` one-hot encoded (e.g., `day_Monday`, `day_Friday`)
  - `fare_per_km` and `fare_per_passenger` (created but **not used** in final model)

Final features used for prediction:
```

\['passenger\_count', 'distance', 'is\_day', 'is\_night',
'day\_Monday', 'day\_Tuesday', 'day\_Wednesday',
'day\_Thursday', 'day\_Friday', 'day\_Saturday', 'day\_Sunday']

````

---

## 🧠 Model Development

- Model used: **XGBoost Regressor**
- Target variable: **Log-transformed `fare_amount`**
- Train-test split: 80/20
- Evaluation metrics:
  - **MAE**: 0.1682
  - **MSE**: 0.0556
  - **RMSE**: 0.2359
  - **R²**: 0.8098

The model was trained directly, and tuned using basic hyperparameter optimization. Final model was saved as `xgb_fare_model.pkl`.

---

## 🌐 Streamlit Web App

The Streamlit app allows users to input trip details and receive predicted fare outputs. Inputs include:

- Pickup and dropoff coordinates (via map)
- Date and time of ride
- Passenger count

Features are auto-engineered in the app to match the trained model. The model's feature order is preserved using the saved JSON.

---

## 🚀 How to Run This Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/thatboypage/uber-fare-prediction.git
cd "Uber Fare Analysis and Prediction"
````

### 2️⃣ Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
streamlit run Uber_fare_streamlitApp.py
```

---

## 🔧 Deployment

You can deploy the app on **Streamlit Community Cloud**:

1. Push your code to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `Uber_fare_streamlitApp.py` as the app file
4. Set up secrets or environment variables if needed

---

## 📦 `requirements.txt` (Example)

Make sure this file includes all the necessary packages:

```
streamlit
xgboost
pandas
numpy
scikit-learn
joblib
```

---

## 🤝 Contributions

This is a personal project, but feel free to fork it, give feedback, or suggest improvements through pull requests!

---

## 📧 Contact

Built by **Richard Olamide Olanite**
For inquiries or collaborations, connect on [LinkedIn](www.linkedin.com/in/richard-olanite-55b4b0241) or \[[email@example.com](richardolanite@gmail.com)]

