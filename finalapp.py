from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load your models
diabetes_model = pickle.load(open('diabetes1.pkl', 'rb'))
kidney_model = pickle.load(open('kidney1.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to Disease Prediction API"

# Diabetes Prediction
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.form

        Pregnancies = float(data.get('Pregnancies'))
        Glucose = float(data.get('Glucose'))
        BloodPressure = float(data.get('BloodPressure'))
        SkinThickness = float(data.get('SkinThickness'))
        Insulin = float(data.get('Insulin'))
        BMI = float(data.get('BMI'))
        DiabetesPedigreeFunction = float(data.get('DiabetesPedigreeFunction'))
        Age = float(data.get('Age'))

        # Feature engineering
        NewBMI_Underweight = 0
        NewBMI_Overweight = 0
        NewBMI_Obesity_1 = 0
        NewBMI_Obesity_2 = 0
        NewBMI_Obesity_3 = 0
        NewInsulinScore_Normal = 0
        NewGlucose_Low = 0
        NewGlucose_Normal = 0
        NewGlucose_Overweight = 0
        NewGlucose_Secret = 0

        if BMI <= 18.5:
            NewBMI_Underweight = 1
        elif 18.5 < BMI <= 24.9:
            pass
        elif 24.9 < BMI <= 29.9:
            NewBMI_Overweight = 1
        elif 29.9 < BMI <= 34.9:
            NewBMI_Obesity_1 = 1
        elif 34.9 < BMI <= 39.9:
            NewBMI_Obesity_2 = 1
        elif BMI > 39.9:
            NewBMI_Obesity_3 = 1

        if 16 <= Insulin <= 166:
            NewInsulinScore_Normal = 1

        if Glucose <= 70:
            NewGlucose_Low = 1
        elif 70 < Glucose <= 99:
            NewGlucose_Normal = 1
        elif 99 < Glucose <= 126:
            NewGlucose_Overweight = 1
        elif Glucose > 126:
            NewGlucose_Secret = 1

        input_query = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                 BMI, DiabetesPedigreeFunction, Age, NewBMI_Underweight,
                                 NewBMI_Overweight, NewBMI_Obesity_1, NewBMI_Obesity_2,
                                 NewBMI_Obesity_3, NewInsulinScore_Normal, NewGlucose_Low,
                                 NewGlucose_Normal, NewGlucose_Overweight, NewGlucose_Secret]])

        prediction = diabetes_model.predict(input_query)[0]

        return jsonify({'diabetes_prediction': str(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Kidney Disease Prediction
@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    try:
        data = request.form

        features = [
            'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
            'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
            'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
            'potassium', 'haemoglobin', 'packed_cell_volume',
            'white_blood_cell_count', 'red_blood_cell_count', 'hypertension',
            'diabetes_mellitus', 'coronary_artery_disease', 'appetite',
            'peda_edema', 'aanemia'
        ]

        input_data = []
        for feature in features:
            input_data.append(float(data.get(feature)))

        input_query = np.array([input_data])

        prediction = kidney_model.predict(input_query)[0]

        return jsonify({'kidney_prediction': str(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
