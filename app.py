from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_data(form_data):
    # Lakukan preprocessing pada data
    # Misalnya, konversi data ke tipe yang sesuai, label encoding, dll.
    gender = form_data.get('gender', 'default_value')
    month = form_data.get('month', 'default_value')
    claim = form_data.get('claim', 'default_value')
    umbrella = form_data.get('umbrella', 'default_value')
    witnesses = form_data.get('witnesses', 'default_value')
    auto_year = form_data.get('year', 'default_value')
    bodily_injuries = form_data.get('injuries', 'default_value')
    number_of_vehicles_involved = form_data.get('involved', 'default_value')
    education_level_mapping = {'Associate': 0, 'College': 1, 'High School': 2, 'JD': 3, 'MD': 4, 'Masters': 5, 'PhD': 6}
    incident_type_mapping = {'Multi-vehicle Collision': 0, 'Parked Car': 1, 'Single Vehicle Collision': 2, 'Vehicle Theft': 3}
    collision_type_mapping = {'Front Collision': 0, 'Rear Collision': 1, 'Side Collision': 2}
    incident_severity_mapping = {'Major Damage': 0, 'Minor Damage': 1, 'Total Loss': 2, 'Trivial Damage': 3}
    authorities_contacted_mapping = {'Ambulance': 0, 'Fire': 1, 'None': 2, 'Other': 3, 'Police': 4}
    police_report_mapping = {'NO': 0, 'YES': 1}

    processed_data = {
        'age': int(form_data['age']),
        'insured_sex': 1 if gender == 'MALE' else 0,
        'months_as_customer': int(month) if month.isdigit() else 0,
        'insured_education_level': education_level_mapping.get(form_data['insured_education_level'], -1),
        'incident_type': incident_type_mapping.get(form_data['incident_type'], -1),
        'collision_type': collision_type_mapping.get(form_data['collision_type'], -1),
        'incident_severity': incident_severity_mapping.get(form_data['incident_severity'], -1),
        'authorities_contacted': authorities_contacted_mapping.get(form_data['authorities_contacted'], -1),
        'police_report_available': police_report_mapping.get(form_data['police_report_available'], -1),
        'vehicle_claim': int(claim) if claim.isdigit() else 0,
        'umbrella_limit': int(umbrella) if umbrella.isdigit() else 0,
        'witnesses': int(witnesses) if witnesses.isdigit() else 0,
        'auto_year': int(auto_year) if auto_year.isdigit() else 0,
        'bodily_injuries': int(bodily_injuries) if bodily_injuries.isdigit() else 0,
        'number_of_vehicles_involved': int(number_of_vehicles_involved) if number_of_vehicles_involved.isdigit() else 0
    }
    return processed_data

def index_to_label(index):
    label = {
        0 : "Safe",
        1 : "Fraud",
    }
    return label[index]

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/PredictionPage")
def predictionpage():
    return render_template('PredictionPage.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            data = {
                'age': request.form.get('age'),
                'insured_sex': request.form.get('gender'),
                'months_as_customer': request.form.get('month'),
                'insured_education_level': request.form.get('edc'),
                'incident_date': request.form.get('date'),
                'incident_type': request.form.get('type'),
                'collision_type': request.form.get('collision'),
                'incident_severity': request.form.get('saverity'),
                'authorities_contacted': request.form.get('contacted'),
                'number_of_vehicles_involved': request.form.get('involved'),
                'bodily_injuries': request.form.get('injuries'),
                'witnesses': request.form.get('witesses'),
                'police_report_available': request.form.get('police'),
                'umbrella_limit': request.form.get('umbrella'),
                'auto_year': request.form.get('year'),
                'vehicle_claim': request.form.get('claim')
            }
            
            processed_data = preprocess_data(data)

            # Correct the order of columns as per the trained model's expectations
            columns = ['months_as_customer', 'age', 'umbrella_limit', 'insured_sex', 'insured_education_level', 'incident_date', 'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted', 'police_report_available', 'vehicle_claim', 'auto_year', 'bodily_injuries', 'number_of_vehicles_involved', 'witnesses']
            df = pd.DataFrame([processed_data], columns=columns)
            
            # Menggunakan predict_proba untuk mendapatkan probabilitas
            probabilities = model.predict_proba(df)
            confidence_score = probabilities[0][1] 
            
            prediction = model.predict(df)
            result = index_to_label(prediction[0])
            result = str(result)
            
            return render_template('PredictionPage.html', prediction_result = result, confidence_score=confidence_score)

        except Exception as e:
            print("Error occurred:", e)
            # Optionally, return a different template with the error message
            return render_template('error_page.html', error=str(e))
    else:
        return render_template('PredictionPage.html')


if __name__ == '__main__':
    app.run(debug=True)