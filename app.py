from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app instance
app = Flask(__name__)

# Load the saved model (ensure the model.pkl is in the same directory as app.py)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    # Render the index.html page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        hours_studied = float(request.form['Hours_Studied'])
        previous_scores = float(request.form['Previous_Scores'])

        # Prepare the feature array for prediction
        final_features = np.array([[hours_studied, previous_scores]])

        # Make the prediction using the loaded model
        prediction = model.predict(final_features)

        # Extract the predicted performance index (continuous value)
        output = prediction[0]

        # Render the result in the HTML template
        return render_template('index.html', prediction_text=f'Predicted Performance Index: {output:.2f}')

    except Exception as e:
        # Handle errors and show a meaningful message
        return render_template('index.html', prediction_text=f"Error: {e}")

# Start the Flask app with debug mode enabled (only for local development)
if __name__ == "__main__":
    app.run(debug=True)
