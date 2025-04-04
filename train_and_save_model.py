import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Define paths
dataset_path = 'D:/Study Material/Engineering/5th sem/Mini Project/Tumour detection/tumor_detection/brain_tumor_dataset'
model_dir = 'D:/Study Material/Engineering/5th sem/Mini Project/Tumour detection/tumor_detection/model'
os.makedirs(model_dir, exist_ok=True)

def load_data(dataset_path):
    images = []
    labels = []
    for label, subdir in enumerate(['no_tumor', 'tumor']):
        subdir_path = os.path.join(dataset_path, subdir)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (64, 64))
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_data(dataset_path)

# Normalize the image data
X = X / 255.0

# Convert labels to categorical for CNN
y_cnn = to_categorical(y)

# Split the dataset into training and test sets
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X, y_cnn, test_size=0.2, random_state=42)

# Define a CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Output layer for two classes
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(X_train_cnn[..., np.newaxis], y_train_cnn, epochs=10, batch_size=32, validation_split=0.2)

# Use the CNN to extract features
feature_extractor = Sequential(cnn_model.layers[:-2])  # Remove the final dense layers
X_features = feature_extractor.predict(X[..., np.newaxis])

# Flatten the features for the SVM
X_features_flattened = X_features.reshape(X_features.shape[0], -1)

# Split features into training and test sets
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_features_flattened, y, test_size=0.2, random_state=42
)

# Create a pipeline with a standard scaler and SVM classifier
pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

# Train the SVM model
pipeline.fit(X_train_svm, y_train_svm)

# Define the model paths
CNN_MODEL_PATH = os.path.join(model_dir, 'cnn_model.h5')
SVM_MODEL_PATH = os.path.join(model_dir, 'cnn_svm_model.pkl')

# Save the trained CNN and SVM models to files
cnn_model.save(CNN_MODEL_PATH)
joblib.dump(pipeline, SVM_MODEL_PATH)


print("\nCNN and SVM models trained and saved successfully.")

