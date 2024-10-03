import numpy as np
from app import app
from flask import render_template, request, jsonify
from app.predictors import RegressionPredictor, CNNPredictor

@app.route('/', methods=['GET'])
def index():
    """
    Load models in advance using the Singleton Pattern to ensure they are ready 
    when the index page is rendered.
    """
    # Initialize the predictors
    RegressionPredictor()
    CNNPredictor()

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Process the incoming data for prediction. The incoming data should be in 
    the format of an image array. Reverse the image colors and normalize it 
    before passing it to the models for predictions.
    """
    # Retrieve input data from the request (expected to be JSON format)
    input_data = np.array(request.json)

    # Reverse (white background, black digit -> black background, white digit) 
    # and normalize the image
    input_data = (255 - input_data) / 255.0

    # Ensure the input is shaped correctly for the models
    # Assuming models expect input in shape (1, 28, 28, 1) for CNN
    input_data = input_data.reshape(1, 28, 28, 1)

    # Get predictions from both models
    result_of_regression = RegressionPredictor().predict(input_data)
    result_of_convolutional = CNNPredictor().predict(input_data)

    # Return the predictions as JSON response
    return jsonify(data=[result_of_regression, result_of_convolutional])
