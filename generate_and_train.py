import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

print("⏳ Generating synthetic training data...")

# 1. Setup Categories
soil_types = ['sandy', 'loamy', 'clay']
crop_types = ['wheat', 'maize', 'rice']

# 2. Create Mock Data (1000 rows)
data = pd.DataFrame({
    'Nitrogen (N)': np.random.uniform(0, 100, 1000),
    'Phosphorus (P)': np.random.uniform(0, 100, 1000),
    'Potassium (K)': np.random.uniform(0, 100, 1000),
    'Rainfall (mm)': np.random.uniform(200, 1000, 1000),
    'Temperature (C)': np.random.uniform(15, 35, 1000),
    'pH Level': np.random.uniform(5.5, 8.5, 1000),
    'Area Size (ha)': np.random.uniform(1, 10, 1000),
    'Soil Type': np.random.choice(soil_types, 1000),
    'Crop Type': np.random.choice(crop_types, 1000)
})

# 3. Create Target Variables (The "Answers")
data['Optimal N'] = np.random.uniform(50, 150, 1000)
data['Optimal P'] = np.random.uniform(30, 90, 1000)
data['Optimal K'] = np.random.uniform(30, 100, 1000)

# 4. Prepare Data for Training
# We use drop_first=True to match the logic in your app.py
data = pd.get_dummies(data, columns=['Soil Type', 'Crop Type'], drop_first=True)

# 5. Define Input (X) and Output (y)
X = data.drop(columns=['Optimal N', 'Optimal P', 'Optimal K'])
y = data[['Optimal N', 'Optimal P', 'Optimal K']]

# 6. Train the Model
print("🧠 Training the model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Save the file
joblib.dump(model, 'fertilizer_model.pkl')
print("✅ Success! 'fertilizer_model.pkl' has been created.")