import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

class VerseDetector:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = load_model("models/saved_model.h5")
        print("Model loaded successfully.")

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        return np.expand_dims(image, axis=0)

    def detect_verse(self, preprocessed_image):
        predictions = self.model.predict(preprocessed_image)
        return predictions.argmax()