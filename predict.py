import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Example input:
# [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
sample_data = np.array([[8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]])

sample_data_scaled = scaler.transform(sample_data)

prediction = model.predict(sample_data_scaled)

print("Predicted House Price:", prediction[0] * 100000, "USD")
