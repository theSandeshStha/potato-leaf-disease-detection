import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

model = None
class_names = ["Early Blight", "Late Blight", "Healthy"]

model_path = "potato.h5"
model = tf.keras.models.load_model(model_path)

def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img.convert("RGB"))
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def is_leaf_like(image):
 
    img_array = np.array(image.convert("RGB"))
 
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
 
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
 
    mask = cv2.inRange(hsv, lower_green, upper_green)
 
    green_ratio = np.sum(mask) / (mask.size * 255)
    return green_ratio > 0.3

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
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

            confidence_threshold = 0.6
            if confidence < confidence_threshold:
                return jsonify({
                    "predicted_class": "Uncertain or Invalid Input",
                    "confidence": confidence * 100
                })

            predicted_class = class_names[predicted_class_index]

            return jsonify({
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2)
            })

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"error": "Error processing image"}), 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)