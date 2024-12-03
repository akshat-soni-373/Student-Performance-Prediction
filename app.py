from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model (make sure the model.pkl is in the same directory as app.py)
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    try:
        # Test model loading
        if model is None:
            return "Model not loaded properly. Please check the model file."
        
        # If model is loaded, proceed with rendering the template
        return render_template('index.html')
    except Exception as e:
        return f"An error occurred while rendering the page: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form
        hours_studied = float(request.form['Hours_Studied'])
        previous_scores = float(request.form['Previous_Scores'])

        # Prepare the features array for prediction
        final_features = np.array([[hours_studied, previous_scores]])

        # Make prediction using the loaded model
        if model:
            prediction = model.predict(final_features)
            output = prediction[0]
        else:
            output = "Model is not available."

        # Return the result to the user (rendering the result in the HTML)
        return render_template('index.html', prediction_text=f'Predicted Performance Index: {output:.2f}')

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
