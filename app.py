from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Home route that serves the form
@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        try:
            # Retrieve form data and convert to float
            feature1 = float(request.form.get('feature1'))
            feature2 = float(request.form.get('feature2'))
            feature3 = float(request.form.get('feature3'))
            feature4 = float(request.form.get('feature4'))
            
            # Prepare the data for prediction
            input_data = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)
            prediction = model.predict(input_data)
            species = {0: "setosa", 1: "versicolor", 2: "virginica"}
            predicted_species = species.get(prediction[0], "Unknown")
            
            result = f"Predicted Species: {predicted_species}"
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('index.html', result=result)

# API endpoint for JSON-based predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data')
        if data is None:
            return jsonify({'error': 'No data provided. Expected JSON key "data".'}), 400
        
        input_data = np.array(data).reshape(1, -1)
        prediction = model.predict(input_data)
        species = {0: "setosa", 1: "versicolor", 2: "virginica"}
        predicted_species = species.get(prediction[0], "Unknown")
        return jsonify({'prediction': predicted_species})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
