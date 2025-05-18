from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Define upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Load the dataset
            data = pd.read_csv(filepath)

            # Store the first 1000 rows in session
            session['data'] = data.head(100).to_dict(orient='records')  # Convert to a list of dictionaries
            
            return render_template('display_data.html', data=session['data'])

        return "Invalid file type", 400

    return render_template('upload.html')

@app.route('/display')
def display():
    # Redirect to predict page if data is not present
    if 'data' not in session:
        return redirect(url_for('upload'))
    return render_template('display_data.html', data=session['data'])

@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    if request.method == 'POST':
        try:
            # Extract and validate form values
            step = request.form.get('step')
            type_ = request.form.get('type')
            amount = request.form.get('amount')
            oldbalanceOrg = request.form.get('oldbalanceOrg')
            newbalanceOrig = request.form.get('newbalanceOrig')
            oldbalanceDest = request.form.get('oldbalanceDest')
            newbalanceDest = request.form.get('newbalanceDest')

            # Check if any of the fields are missing
            if not all([step, type_, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]):
                raise ValueError("Missing fields in the form submission")

            # Convert form values to float
            form_values = [
                float(step),
                float(type_),
                float(amount),
                float(oldbalanceOrg),
                float(newbalanceOrig),
                float(oldbalanceDest),
                float(newbalanceDest)
            ]
            
            # Convert to numpy array and reshape for prediction
            features = [np.array(form_values)]
            
            # Check if the features array has the correct shape
            if len(features[0]) != model.n_features_in_:
                raise ValueError(f"Expected {model.n_features_in_} features, got {len(features[0])}")

            # Predict
            prediction = model.predict(features)

            # Check prediction
            res = "FRAUD" if prediction[0] == 1 else "NOT FRAUD"

        except Exception as e:
            res = f"Error: {str(e)}"
        
        return render_template('result.html', res=res)
    
    return render_template('predict.html')

@app.route('/clear')
def clear():
    session.pop('data', None)  # Clear the stored data
    return redirect(url_for('upload'))

if __name__ == '__main__':
    app.run(debug=True)
