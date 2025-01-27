import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pandas as pd


# Load the nutrition data
data = pd.read_csv("indian_dishes_nutrients.csv")  # Ensure this file exists in the same directory

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and label encoder
model = load_model('model.h5')  # Replace with your trained image model path
label_encoder = pickle.load(open('model.pkl', 'rb'))  # Label encoder for decoding predictions

# Define allowed extensions for image files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('donorpage3.html')  # Ensure this HTML file exists in the `templates` folder

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    name = request.form.get('name')  # Get user name or food name
    quantity = float(request.form.get('quantity'))  # Get quantity
    date = request.form.get('date')  # Get date

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        try:
            # Process the image file
            image = Image.open(file).convert('RGB')
            image = image.resize((128, 128))  # Resize to match model input size
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Predict using the model
            prediction_index = np.argmax(model.predict(image_array))
            predicted_class = label_encoder.inverse_transform([prediction_index])[0]
            
            # Retrieve the nutrients for the predicted food
            if predicted_class in data['Food'].values:
                nutrients = data[data['Food'] == predicted_class].iloc[:, 1:].to_dict('records')[0]
                nutrient_info = "\n".join([f"{key}: {value}" for key, value in nutrients.items()])
            else:
                nutrient_info = "Nutritional information not available."

            # Render the prediction result
            return render_template(
                "donorpage3.html",
                prediction_text=f"Predicted Food: {predicted_class}",
                nutrient_info=f"Nutritional Info:\n{nutrient_info}\n\n"
                              f"Food Name: {name}\n"
                              f"Quantity: {quantity} kg\n"
                              f"Date: {date}"
            )
        except Exception as e:
            return jsonify({"error": f"Error processing the image: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type. Please upload a valid image."}), 400


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    API endpoint for predicting food category and returning nutrition data from an uploaded image.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        try:
            # Process the image file
            image = Image.open(file).convert('RGB')  # Convert to RGB if grayscale
            image = image.resize((128, 128))  # Resize to match the model's expected input size
  
            image_array = img_to_array(image) / 255.0  # Normalize pixel values
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            # Predict using the model
            prediction_index = np.argmax(model.predict(image_array))
            predicted_class = label_encoder.inverse_transform([prediction_index])[0]

            # Retrieve the nutrients for the predicted food
            if predicted_class in data['Food'].values:
                nutrients = data[data['Food'] == predicted_class].iloc[:, 1:].to_dict('records')[0]
                return jsonify({
                    "predicted_food": predicted_class,
                    "nutrients": nutrients
                })
            else:
                return jsonify({
                    "predicted_food": predicted_class,
                    "error": "Nutritional information not available."
                })
        except Exception as e:
            return jsonify({"error": f"Error processing the image: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type. Please upload a valid image."}), 400







if __name__ == "__main__":
    app.run(debug=True)
