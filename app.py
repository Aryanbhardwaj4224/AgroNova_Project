import os
import joblib
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, request

app = Flask(__name__)

# --- CONFIGURATIONS ---
MODEL_FILE = 'fertilizer_model.pkl'

# Configure Gemini (Chatbot)
# For deployment, set the 'GEMINI_API_KEY' in Render Environment Variables.
# For local testing, you can paste your key below, but DO NOT share the file if you do.
GEMINI_API_KEY = "AIzaSyCncUuQhQWQXjWmCoNjduYg98eFzjJmQbQ"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# --- LOAD MODEL ---
fertilizer_model = None
if os.path.exists(MODEL_FILE):
    fertilizer_model = joblib.load(MODEL_FILE)
    print(f"✅ Loaded {MODEL_FILE}")
else:
    print(f"⚠️ WARNING: {MODEL_FILE} not found. Please run generate_and_train.py first.")

# --- ROUTES ---

@app.route('/')
def home():
    # Renders the Main Landing Page
    return render_template('index.html')

@app.route('/resources')
def resources():
    # FAQ page
    return render_template('resources.html')

@app.route('/optimizer')
def optimizer():
    # Fertilizer Form
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not fertilizer_model:
        return render_template('output.html', error="Model not loaded. Run generate_and_train.py on backend.")

    try:
        # 1. Get data from form
        soil_type = request.form.get('soil_type')
        crop_type = request.form.get('crop_type')
        
        # Get numeric inputs
        features = [
            float(request.form['nitrogen']),
            float(request.form['phosphorus']),
            float(request.form['potassium']),
            float(request.form['rainfall']),
            float(request.form['temperature']),
            float(request.form['ph']),
            float(request.form['area'])
        ]

        # 2. Encoding Logic
        # This MUST match exactly how the model was trained in generate_and_train.py
        
        # Soil Encoding (Alphabetical order: Clay, Loamy, Sandy -> Drop First (Clay) -> [Loamy, Sandy])
        soil_encoded = [
            1 if soil_type == 'loamy' else 0, 
            1 if soil_type == 'sandy' else 0
        ]

        # Crop Encoding (Alphabetical order: Maize, Rice, Wheat -> Drop First (Maize) -> [Rice, Wheat])
        crop_encoded = [
            1 if crop_type == 'rice' else 0,
            1 if crop_type == 'wheat' else 0
        ]
        
        # Combine all features
        final_features = np.array(features + soil_encoded + crop_encoded).reshape(1, -1)

        # 3. Predict
        prediction = fertilizer_model.predict(final_features)
        n, p, k = prediction[0]

        return render_template('output.html', 
                               predicted_n=round(n, 2), 
                               predicted_p=round(p, 2), 
                               predicted_k=round(k, 2))

    except Exception as e:
        return f"Error during prediction: {str(e)}"

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    response_text = None
    user_input = ""
    
    if request.method == "POST":
        user_input = request.form.get("user_input")
        # Check if input exists and API key is set (not the placeholder)
        if user_input and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            try:
                response = gemini_model.generate_content(user_input)
                response_text = response.text
            except Exception as e:
                response_text = f"Error: {str(e)}"
        elif GEMINI_API_KEY == "AIzaSyCncUuQhQWQXjWmCoNjduYg98eFzjJmQbQ":
            response_text = "API Key not configured. Please set the GEMINI_API_KEY."

    return render_template("chatbot.html", user_input=user_input, response=response_text)

if __name__ == '__main__':
    app.run(debug=True, port=5000)