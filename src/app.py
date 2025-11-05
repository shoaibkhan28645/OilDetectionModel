

import os
print("Importing Flask...")
from flask import Flask, render_template, request
print("Importing OilDetectionPredictor...")
from predict import OilDetectionPredictor
print("Imports complete.")

app = Flask(__name__)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the model once
print("Loading model...")
model_path = '../models/oil_detection_transfer_learning_20250901_033505.h5'
predictor = OilDetectionPredictor(model_path)
print("Model loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No image selected')

    image = request.files['image']

    if image.filename == '':
        return render_template('index.html', error='No image selected')

    if image:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Predict the image
        result = predictor.predict_single_image(image_path, show_image=False)

        return render_template('result.html', result=result, image_path=image_path)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)

