from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the saved model (make sure the model.pkl is in the same directory as app.py)
try:
    model_path = os.path.join(os.getcwd(), 'model.pkl')
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
        if model is None:
            return render_template('index.html', error_message="Model not loaded properly. Please check the model file.")
        
        return render_template('index.html')
    except Exception as e:
        return render_template('index.html', error_message=f"An error occurred while rendering the page: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', error_message="Model is not loaded. Prediction can't be made.")
        
        hours_studied = request.form.get('Hours_Studied')
        previous_scores = request.form.get('Previous_Scores')

        # Validate input data
        if not hours_studied or not previous_scores:
            return render_template('index.html', error_message="Both 'Hours Studied' and 'Previous Scores' must be provided.")

        try:
            hours_studied = float(hours_studied)
            previous_scores = float(previous_scores)
        except ValueError:
            return render_template('index.html', error_message="Invalid input: Please provide valid numeric values for 'Hours Studied' and 'Previous Scores'.")

        # Prepare the features array for prediction
        final_features = np.array([[hours_studied, previous_scores]])

        # Make prediction using the loaded model
        prediction = model.predict(final_features)
        output = prediction[0]

        return render_template('index.html', prediction_text=f'Predicted Performance Index: {output:.2f}')
    
    except Exception as e:
        return render_template('index.html', error_message=f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=False)
