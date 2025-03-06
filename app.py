from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('fracture_detection_model.h5')

# Define the class labels
class_labels = [
    'avulsion fracture', 'Comminuted fracture', 'Fracture Dislocation',
    'Greenstick fracture', 'Hairline Fracture', 'Impacted fracture',
    'Longitudinal fracture', 'Oblique fracture', 'Pathological fracture', 'Spiral Fracture'
]

# Define a function to prepare the image for prediction
def prepare_image(image_path):
    img = load_img(image_path, target_size=(200, 200))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]
    return img_array

# Route to handle the upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the file
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        
        # Prepare the image and make prediction
        img_array = prepare_image(image_path)
        predictions = model.predict(img_array)
        
        # Get the predicted class and its probability
        predicted_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_index]
        confidence = predictions[0][predicted_index] * 100  # Convert to percentage
        
        # Render the result template with prediction, confidence, and image
        return render_template('result.html', label=predicted_class, confidence=confidence, image_path=url_for('uploaded_file', filename=file.filename))

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Run the app
if __name__ == '__main__':
    # Ensure the uploads folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
