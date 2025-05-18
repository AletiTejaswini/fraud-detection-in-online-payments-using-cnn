from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os
app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

@app.route('/')
def index():
    return render_template('index1.html')
@app.route('/about')
def about():
    return render_template('about.html')



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

            # Prepare the data for prediction
            features = data[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
            predictions = model.predict(features)

            # Add predictions to the DataFrame
            data['prediction'] = ["FRAUD" if pred == 1 else "NOT FRAUD" for pred in predictions]

            # Save results to a new CSV file
            result_file = os.path.join(UPLOAD_FOLDER, 'predictions.csv')
            data.to_csv(result_file, index=False)

            return render_template('result.html', res=f"Predictions saved to {result_file}")

        return "Invalid file type", 400

    # If GET request, show the upload form
    return render_template('upload.html')

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
            
            # Debug: Print the features
            print(f"Form values: {form_values}")

            # Convert to numpy array and reshape for prediction
            features = [np.array(form_values)]
            
            # Debug: Print the features shape
            print(f"Features shape: {np.shape(features[0])}")

            # Check if the features array has the correct shape
            if len(features[0]) != model.n_features_in_:
                raise ValueError(f"Expected {model.n_features_in_} features, got {len(features[0])}")

            # Predict
            prediction = model.predict(features)

            # Debug: Print the prediction
            print(f"Prediction: {prediction}")

            # Check prediction
            res = "FRAUD" if prediction[0] == 0 else "NOT FRAUD"

        except Exception as e:
            # Debug: Print the error
            print(f"Error: {str(e)}")
            res = f"Error: {str(e)}"
        
        return render_template('result.html', res=res)
    
    # If GET request, show the form
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
