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

            # Store the first 100 rows in session
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
            step = float(request.form.get('step'))
            type_ = float(request.form.get('type'))
            amount = float(request.form.get('amount'))
            oldbalanceOrg = float(request.form.get('oldbalanceOrg'))
            newbalanceOrig = float(request.form.get('newbalanceOrig'))
            oldbalanceDest = float(request.form.get('oldbalanceDest'))
            newbalanceDest = float(request.form.get('newbalanceDest'))

            # Convert to numpy array and reshape for prediction
            features = np.array([[step, type_, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])
            
            # Predict
            prediction = model.predict(features)
            risk_score = model.predict_proba(features)[0][1] * 100  # Get risk score as a percentage

            # Check prediction
            res = "FRAUD" if prediction[0] == 1 else "NOT FRAUD"

            # Example indicators (you can replace these with your logic)
            fraud_indicators = []
            if amount > 1000:
                fraud_indicators.append("Large transaction amount.")
            if oldbalanceOrg < 100:
                fraud_indicators.append("Old balance is unusually low.")

            # User behavior analysis (simple example)
            behavior_analysis = "This transaction is significantly different from your usual patterns."

        except Exception as e:
            res = f"Error: {str(e)}"
            risk_score = None
            fraud_indicators = []
            behavior_analysis = ""

        return render_template('result.html', res=res, risk_score=risk_score,
                               amount=amount, type_=type_, 
                               oldbalanceOrg=oldbalanceOrg, newbalanceOrig=newbalanceOrig,
                               oldbalanceDest=oldbalanceDest, newbalanceDest=newbalanceDest,
                               fraud_indicators=fraud_indicators,
                               behavior_analysis=behavior_analysis)

    return render_template('predict.html')

@app.route('/clear')
def clear():
    session.pop('data', None)  # Clear the stored data
    return redirect(url_for('upload'))

if __name__ == '__main__':
    app.run(debug=True)
