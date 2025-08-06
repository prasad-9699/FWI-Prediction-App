from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create Flask application
application = Flask(__name__)
app = application

# Load Ridge regressor and standard scaler from pickle files
ridge_model = pickle.load(open(r'D:\ML\implementation\models\ridge.pkl', 'rb'))
standard_scaler = pickle.load(open(r'D:\ML\implementation\models\scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('home.html')  # Change to match the actual template filename

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get form data from the request
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = int(request.form.get("Classes"))  # Assuming categorical (0/1)
        Region = int(request.form.get("Region"))    # Assuming categorical (0/1)

        # Create a NumPy array for model input
        new_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Scale the data
        scaled_data = standard_scaler.transform(new_data)

        # Predict
        result = ridge_model.predict(scaled_data)[0]

        return render_template('home.html', result=result)

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
