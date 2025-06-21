from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('index1.html')

@app.route('/Home', methods=['GET'])
def my_home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Binary encodings
        gender = 0 if request.form.get('gender') == 'FEMALE' else 1
        car = 1 if request.form.get('car') == 'YES' else 0
        realestate = 1 if request.form.get('realestate') == 'YES' else 0
        income = float(request.form.get('income'))

        # Normalize and encode NAME_INCOME_TYPE
        income_type_raw = request.form.get('income_type')
        income_type_simplified = {
            'Commercial associate': 'Working',
            'State servant': 'Working',
            'Working': 'Working',
            'Pensioner': 'Pensioner',
            'Student': 'Student'
        }[income_type_raw]
        income_type_map = {
            'Pensioner': 0,
            'Student': 1,
            'Working': 2
        }
        income_type = income_type_map[income_type_simplified]

        # Normalize and encode NAME_EDUCATION_TYPE
        education_raw = request.form.get('education')
        education_simplified = {
            'Secondary / secondary special': 'secondary',
            'Lower secondary': 'secondary',
            'Higher education': 'Higher education',
            'Incomplete higher': 'Higher education',
            'Academic degree': 'Academic degree'
        }[education_raw]
        education_map = {
            'Academic degree': 0,
            'Higher education': 1,
            'secondary': 2
        }
        education = education_map[education_simplified]

        # Normalize and encode NAME_FAMILY_STATUS
        family_status_raw = request.form.get('family_status')
        family_status_simplified = {
            'Single / not married': 'Single',
            'Separated': 'Single',
            'Widow': 'Single',
            'Civil marriage': 'Married',
            'Married': 'Married'
        }[family_status_raw]
        family_status_map = {
            'Married': 0,
            'Single': 1
        }
        family_status = family_status_map[family_status_simplified]

        # Normalize and encode NAME_HOUSING_TYPE
        housing_type_raw = request.form.get('housing_type')
        housing_type_simplified = {
            'House / apartment': 'House / apartment',
            'Municipal apartment': 'House / apartment',
            'Rented apartment': 'House / apartment',
            'Office apartment': 'House / apartment',
            'Co-op apartment': 'House / apartment',
            'With parents': 'With parents'
        }[housing_type_raw]
        housing_type_map = {
            'House / apartment': 0,
            'With parents': 1
        }
        housing_type = housing_type_map[housing_type_simplified]

        # Float features
        days_birth = float(request.form.get('days_birth'))
        days_employed = float(request.form.get('days_employed'))
        family_members = float(request.form.get('family_members'))

        # Loan features
        emi_paid_off = int(request.form.get('emi_paid_off'))
        emi_past_dues = int(request.form.get('emi_past_dues'))
        loans = int(request.form.get('loans'))

        # Assemble feature vector
        input_data = pd.DataFrame([[
            gender, car, realestate, income,
            income_type, education, family_status,
            housing_type, days_birth, days_employed,
            family_members, emi_paid_off, emi_past_dues, loans
        ]], columns=[
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL',
            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
            'CNT_FAM_MEMBERS', 'paid_off', '#_of_pastdues', 'no_loan'
        ])

        # Prediction
        prediction = model.predict(input_data)[0]
        result_text = "✅ Approved" if prediction == 1 else "❌ Not Approved"

        return render_template('result.html', prediction=result_text)

    except Exception as e:
        return f"<h3>❌ Error during prediction: {str(e)}</h3>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True)
