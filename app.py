from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the saved model (make sure the model.pkl is in the same directory as app.py)
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
    try:
        # Test model loading
        if model is None:
            return "Model not loaded properly. Please check the model file."
        
        return render_template('index.html')
    except Exception as e:
        return f"An error occurred while rendering the page: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return "Model is not loaded. Prediction can't be made."
        
        # Get input from the form
        hours_studied = request.form.get('Hours_Studied')
        previous_scores = request.form.get('Previous_Scores')

        # Log the form data for debugging
        print(f"Received inputs: Hours_Studied = {hours_studied}, Previous_Scores = {previous_scores}")

        # Validate input data
        if not hours_studied or not previous_scores:
            return "Both 'Hours Studied' and 'Previous Scores' must be provided."

        try:
            hours_studied = float(hours_studied)
            previous_scores = float(previous_scores)
        except ValueError:
            return "Invalid input: Please provide valid numeric values for 'Hours Studied' and 'Previous Scores'."

        # Prepare the features array for prediction
        final_features = np.array([[hours_studied, previous_scores]])

        # Make prediction using the loaded model
        prediction = model.predict(final_features)
        output = prediction[0]

        # Log prediction result
        print(f"Prediction result: {output}")

        return render_template('index.html', prediction_text=f'Predicted Performance Index: {output:.2f}')
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return f"An error occurred during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
