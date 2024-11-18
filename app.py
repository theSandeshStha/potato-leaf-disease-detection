import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

MODEL_PATH = "potato.h5"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
CONFIDENCE_THRESHOLD = 0.6

model = None

app = Flask(__name__)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from '{MODEL_PATH}'.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise SystemExit("Failed to load model. Exiting application.")


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Preprocesses the input image for model inference.

    Parameters:
        img (PIL.Image.Image): Input image.

    Returns:
        np.ndarray: Preprocessed image array ready for model prediction.
    """
    img = img.resize((256, 256))
    img_array = np.array(img.convert("RGB")) / 255.0
    return np.expand_dims(img_array, axis=0)


def is_leaf_like(image: Image.Image) -> bool:
    """
    Determines if the input image resembles a leaf based on green color detection.

    Parameters:
        image (PIL.Image.Image): Input image.

    Returns:
        bool: True if the image is leaf-like, otherwise False.
    """
    img_array = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask) / (mask.size * 255)
    return green_ratio > 0.3


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handles both GET and POST requests. Renders the main HTML page or performs
    image classification based on the POST request.
    """
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            img = Image.open(file.stream)
            if not is_leaf_like(img):
                return jsonify({"error": "Input image does not resemble a leaf."}), 400

            img_array = preprocess_image(img)
            predictions = model.predict(img_array)
            confidence = np.max(predictions[0])
            predicted_class_index = np.argmax(predictions[0])

            if confidence < CONFIDENCE_THRESHOLD:
                return jsonify({
                    "predicted_class": "Uncertain or Invalid Input",
                    "confidence": round(confidence * 100, 2)
                })

            predicted_class = CLASS_NAMES[predicted_class_index]

            return jsonify({
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2)
            })

        except Exception as e:
            app.logger.error(f"Error processing image: {e}")
            return jsonify({"error": "Error processing image"}), 500

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
