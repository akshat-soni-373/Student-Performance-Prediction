from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__, template_folder='.')

# Load the model (ensure the model.pkl is in the same directory as app.py)
try:
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        model = pickle.load(open(model_path, 'rb'))
    else:
        raise FileNotFoundError("The model.pkl file does not exist.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the model is loaded
        if model is None:
            print("Model is not loaded. Prediction can't be made.")
            return "Model is not loaded properly. Please check the model file."

        # Get the form data
        hours_studied = request.form['Hours_Studied']
        previous_scores = request.form['Previous_Scores']

        # Log the form inputs
        print(f"Received inputs - Hours Studied: {hours_studied}, Previous Scores: {previous_scores}")

        # Check if the inputs are valid numbers
        try:
            hours_studied = float(hours_studied)
            previous_scores = float(previous_scores)
        except ValueError:
            return "Invalid input values. Please enter valid numbers for both fields."

        # Prepare the features array for prediction
        final_features = np.array([[hours_studied, previous_scores]])

        # Log the features array
        print(f"Features array: {final_features}")

        # Make prediction using the model
        prediction = model.predict(final_features)
        output = prediction[0]

        # Log the output prediction
        print(f"Prediction: {output}")

        # Return the result to be displayed on the same page
        return render_template('index.html', prediction_text=f'Predicted Performance Index: {output:.2f}')
    except Exception as e:
        # Log the error
        print(f"Error during prediction: {str(e)}")
        return f"An error occurred during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
