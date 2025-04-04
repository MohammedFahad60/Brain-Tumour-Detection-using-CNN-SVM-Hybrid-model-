import cv2
import numpy as np
import joblib
import base64
import os
import logging
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model, Sequential

app = Flask(__name__)

# Set up logging
log_messages = []

class ListHandler(logging.Handler):
    def emit(self, record):
        log_messages.append(self.format(record))

handler = ListHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)

# Define the correct path to the model files
model_dir = 'D:/Study Material/Engineering/5th sem/Mini Project/Tumour detection/tumor_detection/model'
cnn_model_path = os.path.join(model_dir, 'cnn_model.h5')
svm_model_path = os.path.join(model_dir, 'cnn_svm_model.pkl')

# Load the trained models
cnn_model = load_model(cnn_model_path)
svm = joblib.load(svm_model_path)

# Extract the feature extractor part of the CNN
feature_extractor = Sequential(cnn_model.layers[:-2])

def extract_features(image):
    # Resize the image to the CNN input size
    resized_image = cv2.resize(image, (64, 64)).reshape(1, 64, 64, 1) / 255.0
    # Use the CNN model to extract features
    features = feature_extractor.predict(resized_image).flatten()
    logging.debug(f'Extracted features: {features}')
    return features

def detect_tumor(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = extract_features(gray_image).reshape(1, -1)
    prediction = svm.predict(features)
    logging.debug(f'Prediction: {prediction}')

    if prediction[0] == 1:
        # Tumor detected, highlight the tumor area

        # Apply Gaussian blur and thresholding to isolate the tumor
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, thresh = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # **Filter Contours Based on Area and Shape**
        min_contour_area = 1
        max_contour_area = 5000
        filtered_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_contour_area < area < max_contour_area:
                # Check for aspect ratio (optional)
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:  # Roughly tumor-like shape
                    filtered_contours.append(c)

        # Create a blank image for the segmented image (with borders)
        segmented_image = image.copy()

        # Draw contours on the segmented image (with graph-like border)
        cv2.drawContours(segmented_image, filtered_contours, -1, (0, 255, 0), 2)  # Green borders

        # Calculate tumor-affected area percentage using filtered contours
        tumor_area = sum(cv2.contourArea(c) for c in filtered_contours)
        total_area = gray_image.shape[0] * gray_image.shape[1]
        tumor_percentage = (tumor_area / total_area) * 100
        logging.debug(f'Tumor percentage: {tumor_percentage:.2f}%')

        return prediction[0], segmented_image, tumor_percentage
    else:
        # No tumor detected, return the original image
        return prediction[0], image, 0.0




def stage(prediction):
    return 'Malignant' if prediction == 1 else 'Benign'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    log_messages.clear()
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})

    try:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image file'})

        result, processed_image, tumor_percentage = detect_tumor(image)
        result_text = 'Tumor Detected' if result == 1 else 'No Tumor'
        tumor_stage = stage(result)
        cancer_chance = f"{min(max(tumor_percentage, 10), 95):.2f}%" if result == 1 else "0%"

        _, original_img_encoded = cv2.imencode('.png', image)
        original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')

        _, processed_img_encoded = cv2.imencode('.png', processed_image)
        processed_img_base64 = base64.b64encode(processed_img_encoded).decode('utf-8')
        
        

        return jsonify({
            'diagnosis': result_text,
            'tumor_stage': tumor_stage,
            'tumor_percentage': f"{tumor_percentage:.2f}%",
            'cancer_chance': cancer_chance,
            'original_image': original_img_base64,
            'image': processed_img_base64,
            'logs': log_messages
        })

    except Exception as e:
        logging.error(f'Error processing image: {e}')
        return jsonify({'error': str(e), 'logs': log_messages})
    
    
    
if __name__ == '__main__':
    app.run(debug=True)

