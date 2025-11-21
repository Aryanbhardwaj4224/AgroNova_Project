import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load your data
csv_file = 'generated_soil_data.csv'
try:
    data = pd.read_csv(csv_file)
    print("✅ CSV loaded successfully!")
    
    # FIX: Strip whitespace from column names (fixes 'Soil Type ' vs 'Soil Type' issues)
    data.columns = data.columns.str.strip()
    print(f"🧐 Columns found: {data.columns.tolist()}")

except FileNotFoundError:
    print(f"❌ Error: '{csv_file}' not found. Please make sure the file is in the AgroNova_Project folder.")
    exit()

# 2. Rename columns if they are different (Standardizing)
# This maps common abbreviations to the full names your app expects
rename_map = {
    'N': 'Nitrogen (N)', 
    'P': 'Phosphorus (P)', 
    'K': 'Potassium (K)',
    'Rain': 'Rainfall (mm)',
    'Temp': 'Temperature (C)',
    'pH': 'pH Level',
    'Area': 'Area Size (ha)'
}
data.rename(columns=rename_map, inplace=True)

# 3. Check for required columns again
required_col = 'Soil Type'
if required_col not in data.columns:
    print(f"\n❌ CRITICAL ERROR: Column '{required_col}' still not found.")
    print(f"   Please look at the 'Columns found' list above and rename your CSV header to '{required_col}'.")
    exit()

# 4. Prepare Data
print("⚙️ Processing data...")
if data['Soil Type'].dtype == 'object':
    data = pd.get_dummies(data, columns=['Soil Type', 'Crop Type'], drop_first=True)

# 5. Align Columns to match app.py expectation
features_to_keep = [
    'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
    'Rainfall (mm)', 'Temperature (C)', 'pH Level', 'Area Size (ha)'
]
# Add the dummy columns that were created
dummy_cols = [col for col in data.columns if 'Soil Type_' in col or 'Crop Type_' in col]
features_to_keep.extend(dummy_cols)

X = data[features_to_keep]
y = data[['Optimal N', 'Optimal P', 'Optimal K']]

# 6. Train
print("🧠 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Save
joblib.dump(model, 'fertilizer_model.pkl')
print("✅ Success! Model saved as 'fertilizer_model.pkl'")