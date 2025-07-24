import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, asin, sqrt
from datetime import datetime, time, date

# Load the pre-trained model
model = joblib.load('xgb_fare_model.pkl')

# Optional: Styling for the background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to calculate the distance between two coordinates using the Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Streamlit app title
st.title('🚖 Uber Fare Prediction App')
st.markdown('Predict the fare amount for an Uber ride based on distance and ride details. 📍')

# Sidebar inputs
st.sidebar.header("🛠 Input Ride Details")

pickup_longitude = st.sidebar.number_input('Pickup Longitude', value=-73.985428, format="%.6f")
pickup_latitude = st.sidebar.number_input('Pickup Latitude', value=40.748817, format="%.6f")
dropoff_longitude = st.sidebar.number_input('Dropoff Longitude', value=-73.985428, format="%.6f")
dropoff_latitude = st.sidebar.number_input('Dropoff Latitude', value=40.748817, format="%.6f")
passenger_count = st.sidebar.slider('Number of Passengers', min_value=1, max_value=6, value=1)
pickup_date = st.sidebar.date_input(
    "Pickup Date",
    value=date(2015, 1, 1),
    min_value=date(2009, 1, 1),
    max_value=date(2030, 12, 31)
)
pickup_time = st.sidebar.time_input("Pickup Time", value=time(datetime.now().hour, datetime.now().minute))

# Combine date and time into a single datetime object
pickup_datetime = datetime.combine(pickup_date, pickup_time)

# Calculate distance
distance = haversine(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)

# Extract time features
hour = pickup_datetime.hour
is_night = int(hour < 6 or hour >= 20)
is_day = int(6 <= hour < 20)

# One-hot encode day of the week
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_name = pickup_datetime.strftime('%A')
day_features = {f'day_{d}': int(day_name == d) for d in days}

# Prediction
if st.button("Predict Fare"):
    if distance == 0:
        st.warning("Pickup and dropoff locations are the same. Please enter different coordinates.")
    else:
        # Combine inputs
        input_features = {
            'passenger_count': passenger_count,
            'distance': distance,
            'is_night': is_night,
            'is_day': is_day
        }
        input_features.update(day_features)

        # Match exact order used in training
        expected_columns = ['passenger_count', 'is_night', 'is_day', 'distance',
                            'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday',
                            'day_Thursday', 'day_Tuesday', 'day_Wednesday']
        ordered_input = {col: input_features[col] for col in expected_columns}
        input_data = pd.DataFrame([ordered_input])

        # Predict
        predicted_fare_log = model.predict(input_data)[0]
        predicted_fare = np.exp(predicted_fare_log)

        st.success(f"💵 Fare Amount: ${predicted_fare:.2f}")
        st.write(f"📏 Distance: {distance:.2f} km")