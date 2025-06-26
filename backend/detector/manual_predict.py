import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

MODEL_PATH = "/home/akshita-bindal/Desktop/new_manual/website/backend/manual_classifier_mobilenetv2.h5"
IMG_SIZE = (128, 128)
THRESHOLD = 0.5

# Load the model once at server start
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image_file(file):
    try:
        image = Image.open(file).convert("RGB")
        image = image.resize(IMG_SIZE)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)[0][0]
        print(f"Raw prediction value: {prediction}")
        # Return both the boolean and the raw value for debugging
        return {
            'contains_manual': bool(prediction >= THRESHOLD),
            'raw_prediction': float(prediction)
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None