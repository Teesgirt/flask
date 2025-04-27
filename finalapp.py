from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load models
diabetes_model = pickle.load(open('diabetes1.pkl', 'rb'))
heart_model = pickle.load(open('heart1.pkl', 'rb'))
# kidney_model = pickle.load(open('kidney.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to Disease Prediction API"

# Symptom-Based Disease Prediction API
@app.route('/predict_from_symptoms', methods=['POST'])
def predict_from_symptoms():
    try:
        data = request.form  # Use form data

        symptoms_text = data.get('symptoms', '')
        severity = int(data.get('severity', 5))
        temperature = float(data.get('temperature', 98.6))

        if not symptoms_text:
            return jsonify({'error': 'Symptoms are required'})

        symptoms = [s.strip().lower() for s in symptoms_text.split(',')]

        diabetes_symptoms = ['frequent urination', 'excessive thirst', 'extreme hunger', 'unexplained weight loss', 'fatigue', 'blurred vision']
        heart_disease_symptoms = ['chest pain', 'shortness of breath', 'pain in neck', 'jaw pain', 'pain in back', 'fatigue']

        diabetes_score = sum(1 for symptom in symptoms if symptom in diabetes_symptoms)
        heart_score = sum(1 for symptom in symptoms if symptom in heart_disease_symptoms)

        # Modify score based on severity and temperature
        if severity >= 7:
            heart_score += 2
        if temperature > 100.4:
            diabetes_score += 1

        if diabetes_score > heart_score:
            result = "Likely Diabetes"
        elif heart_score > diabetes_score:
            result = "Likely Heart Disease"
        else:
            result = "Symptoms unclear, please consult a doctor."

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
