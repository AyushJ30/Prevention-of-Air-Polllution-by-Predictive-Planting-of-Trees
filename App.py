from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('pollution_model.pkl')

# Define plant data for recommendations
plant_data = pd.DataFrame({
    'Plant_Name': ['Neem Tree', 'Banyan Tree', 'Pine Tree', 'Eucalyptus'],
    'Pollutants_Absorbed': ['SO2, NO2, CO, PM', 'CO, SO2, PM', 'PM, NO2, SO2, CO', 'CO, SO2, O3'],
    'Absorption_Rate_Per_Plant': [0.8, 0.5, 1.2, 0.9]
})

# Function to recommend plants based on pollutant levels
def recommend_plants(pollutants):
    recommendations = []
    for _, row in plant_data.iterrows():
        absorbed_pollutants = row['Pollutants_Absorbed'].split(', ')
        matching_pollutants = [poll for poll in pollutants if poll in absorbed_pollutants]
        if matching_pollutants:
            recommendations.append({
                'Plant': row['Plant_Name'],
                'Matching_Pollutants': matching_pollutants,
                'Absorption_Rate': row['Absorption_Rate_Per_Plant'],
                'Plants_Required': max(pollutants.values()) / row['Absorption_Rate_Per_Plant']
            })
    return recommendations



# Route to predict pollution and recommend plants
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features for prediction
    features = [data['latitude'], data['longitude'], data['temperature_celsius'], data['wind_mph'],
                data['humidity'], data['pressure_mb'], data['air_quality_PM10'],
                data['air_quality_Carbon_Monoxide'], data['air_quality_Nitrogen_dioxide'],
                data['air_quality_Sulphur_dioxide'], data['air_quality_Ozone']]

    # Convert the input data to a numpy array
    input_data = np.array(features).reshape(1, -1)
    
    # Predict PM2.5 levels
    pm25_prediction = model.predict(input_data)[0]

    # Pollutants detected from input
    pollutants = {
        'PM': data['air_quality_PM10'],
        'CO': data['air_quality_Carbon_Monoxide'],
        'NO2': data['air_quality_Nitrogen_dioxide'],
        'SO2': data['air_quality_Sulphur_dioxide'],
        'O3': data['air_quality_Ozone']
    }

    # Get plant recommendations
    plant_recommendations = recommend_plants(pollutants)

    return jsonify({
        'PM2.5_prediction': pm25_prediction,
        'Plant_Recommendations': plant_recommendations
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
