from flask import Flask, render_template, request, flash
import joblib
import numpy as np
import random
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Load trained model with error handling
try:
    model = joblib.load("logreg.pkl")
except FileNotFoundError:
    print("Warning: Model file not found. Using dummy predictions.")
    model = None

# Mapping numeric prediction to stage
stage_map = {
    0: 'NORMAL',
    1: 'HYPERTENSION (Stage-1)',
    2: 'HYPERTENSION (Stage-2)',
    3: 'HYPERTENSIVE CRISIS'
}

# Medical-grade color mapping
color_map = {
    0: '#108981',  # Normal green
    1: '#F59E0B',  # Stage 1 amber
    2: '#F97316',  # Stage 2 orange
    3: '#EF4444'   # Crisis red
}

# Recommendations
recommendations = {
    0: {
        'title': 'Normal Blood Pressure',
        'description': 'Your cardiovascular risk is normal.',
        'actions': [
            'Maintain current healthy lifestyle',
            'Regular physical activity (150 minutes/week)',
            'Balanced, low-sodium diet',
            'Annual BP monitoring'
        ],
        'priority': 'Low Risk'
    },
    1: {
        'title': 'Stage 1 Hypertension',
        'description': 'Mild elevation detected requiring lifestyle modifications.',
        'actions': [
            'Consult healthcare provider',
            'DASH diet plan',
            'Increase physical activity',
            'Monitor BP bi-weekly'
        ],
        'priority': 'Moderate Risk'
    },
    2: {
        'title': 'Stage 2 Hypertension',
        'description': 'Significant hypertension requiring medical intervention.',
        'actions': [
            'Consult physician urgently',
            'Likely medication therapy',
            'Daily BP monitoring',
            'Lifestyle modification counseling'
        ],
        'priority': 'High Risk'
    },
    3: {
        'title': 'Hypertensive Crisis',
        'description': 'CRITICAL: Requires emergency medical care.',
        'actions': [
            'Seek immediate medical attention',
            'Call 911 if symptomatic',
            'Monitor for stroke/heart attack signs',
            'Prepare medication list'
        ],
        'priority': 'EMERGENCY'
    }
}

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            required_fields = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication',
                           'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
                           'Whendiagnosed', 'Systolic', 'Diastolic', 'ControlledDiet']
        
        form_data = {}
        for field in required_fields:
            value = request.form.get(field)
            if not value or value == "":
                flash(f"Please complete all required fields: {field.replace('_',' ')}", "error")
                return render_template('index.html')
            form_data[field] = value
        
        try:
            encoded = [
                0 if form_data['Gender'] == 'Male' else 1,
                {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4}[form_data['Age']],
                1 if form_data['History'] == 'Yes' else 0,
                1 if form_data['Patient'] == 'Yes' else 0,
                1 if form_data['TakeMedication'] == 'Yes' else 0,
                {'Mild': 0, 'Moderate': 1, 'Severe': 2}[form_data['Severity']],
                1 if form_data['BreathShortness'] == 'Yes' else 0,
                1 if form_data['VisualChanges'] == 'Yes' else 0,
                1 if form_data['NoseBleeding'] == 'Yes' else 0,
                {'<1 Year': 1, '1-5 Years': 2, '>5 Years': 3}[form_data['Whendiagnosed']],
                {'100-110': 0, '111-120': 1, '121-130': 2, '130+': 3}[form_data['Systolic']],
                {'70-80': 0, '81-90': 1, '91-100': 2, '100+': 3}[form_data['Diastolic']],
                1 if form_data['ControlledDiet'] == 'Yes' else 0
            ]
        except KeyError as e:
            flash(f"Invalid selection detected: {str(e)}","error")
            return render_template('index.html')
        
        scaled_encoded = encoded.copy()
        scaled_encoded[1] = (encoded[1]-1)/3
        scaled_encoded[5] = encoded[5]/2
        scaled_encoded[9] = (encoded[9]-1)/2
        scaled_encoded[10] = encoded[10]/3
        scaled_encoded[11] = encoded[11]/3
        
        input_array = np.array(scaled_encoded).reshape(1, -1)
        
        if model is not None:
            prediction = int(model.predict(input_array)[0])
            try:
                confidence = max(model.predict_proba(input_array)[0])*100
            except:
                confidence = 85.0
        else:
            prediction = random.randint(0, 3)
            confidence = 87.5
            flash("Demo Mode: Using simulated AI prediction for demonstration", "info")
        
        # Correct mapping
        risk_class_map = {
            0: 'low-risk',
            1: 'moderate-risk',
            2: 'high-risk',
            3: 'critical-risk'
        }
        
        result_text = stage_map[prediction]
        result_color = color_map[prediction]
        result_recommendation = recommendations[prediction]
        risk_class = risk_class_map[prediction]
        
        # Debug print to verify
        print(f"Prediction: {prediction}, Risk Class: {risk_class}, Text: {result_text}")
        
        return render_template('index.html',
                               prediction_text=result_text,
                               result_color=result_color,
                               confidence=confidence,
                               recommendation=result_recommendation,
                               risk_class=risk_class,
                               form_data=form_data)
    
    except Exception as e:
        flash("System error occurred. Please try again or contact technical support.", "error")
        print(f"Error details: {str(e)}")
        return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)
