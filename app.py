from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model (make sure the model.pkl is in the same directory as app.py)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    hours_studied = float(request.form['Hours_Studied'])
    previous_scores = float(request.form['Previous_Scores'])

    # Prepare the features array for prediction
    final_features = np.array([[hours_studied, previous_scores]])

    # Make prediction using the loaded model
    prediction = model.predict(final_features)

    # Output the predicted performance index (continuous value)
    output = prediction[0]

    # Return the result to the user (rendering the result in the HTML)
    return render_template('index.html', prediction_text=f'Predicted Performance Index: {output:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
