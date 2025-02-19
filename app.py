from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import pandas as pd

# Create a Flask application instance
app = Flask(__name__)

# Enable CORS for the app
CORS(app)

# Load the pre-trained machine learning model
model = joblib.load('water_quality_model.pkl')

# Recommendation function
def get_recommendation(sensor_data):
    recommendations = []
    if sensor_data['Ammonia Level'] > 3.0:
        recommendations.append({
            "issue": "High ammonia level detected.",
            "recommendation": "Consider adding biological filters or reducing organic waste input."
        })
    if sensor_data['Dissolved Oxygen'] < 5.0:
        recommendations.append({
            "issue": "Low dissolved oxygen detected.",
            "recommendation": "Install aerators to improve oxygen levels."
        })
    if not (6.5 <= sensor_data['pH Level'] <= 8.5):
        recommendations.append({
            "issue": "pH level is outside the optimal range.",
            "recommendation": "Adjust pH using buffers like sodium carbonate or acidic solutions."
        })
    if sensor_data['Temperature'] < 10 or sensor_data['Temperature'] > 35:
        recommendations.append({
            "issue": "Temperature is outside the optimal range.",
            "recommendation": "Control temperature using cooling or heating systems."
        })

    if not recommendations:
        recommendations.append({
            "issue": "All parameters are within acceptable ranges.",
            "recommendation": "Continue monitoring to maintain optimal conditions."
        })

    return recommendations

# Define a route to handle water quality prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data], columns=['Temperature', 'pH Level', 'Ammonia Level', 'Dissolved Oxygen'])
        prediction = model.predict(input_data)
        classes = {0: "Poor", 1: "Average", 2: "Good"}
        result = classes[int(prediction[0])]
        recommendations = get_recommendation(data)
        return jsonify({
            "water_quality": result,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Health check route
@app.route('/', methods=['GET'])
def home():
    return "Fish Pond Simulator Backend is Running!"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
