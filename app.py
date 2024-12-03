from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model_path = 'model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None
    print("Error: model.pkl not found.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    error_message = None

    if request.method == 'POST':
        try:
            if model is None:
                error_message = "Model not loaded. Please check the server setup."
            else:
                # Extract input data from form
                hours_studied = float(request.form['Hours_Studied'])
                previous_scores = float(request.form['Previous_Scores'])

                # Prepare input data for prediction
                input_features = np.array([[hours_studied, previous_scores]])
                prediction = model.predict(input_features)

                # Format prediction output
                prediction_text = f"Predicted Performance Index: {prediction[0]:.2f}"
        except Exception as e:
            error_message = f"An error occurred during prediction: {str(e)}"

    # Render the template with variables
    return render_template('index.html', prediction_text=prediction_text, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
