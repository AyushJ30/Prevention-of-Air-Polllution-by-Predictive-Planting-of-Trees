import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset from the CSV file
data = pd.read_csv('GlobalWeatherRepository.csv')  # Replace with your CSV file path

# Check if there are any non-numeric columns
print(data.dtypes)

# Drop unnecessary time-related columns
data = data.drop(columns=['last_updated', 'sunrise', 'sunset', 'moonrise', 'moonset', 'last_updated_epoch', 'timezone'])

# Convert categorical columns to numeric using one-hot encoding
data = pd.get_dummies(data, columns=['country', 'location_name', 'condition_text', 'moon_phase', 'wind_direction'])

# Now that data is encoded, we can split the data into features (X) and target (y)
X = data.drop(columns=['air_quality_PM2.5'])  # Features
y = data['air_quality_PM2.5']  # Target (PM2.5 level)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(model, 'pollution_model.pkl')

print("Model trained and saved successfully!")
