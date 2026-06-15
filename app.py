from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# load model
model = tf.keras.models.load_model("malaria_model.keras")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

@app.route("/")
def home():
    return {"message": "AfyaLens API Running"}

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"].read()

    img = preprocess(file)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = "Parasitized"
    else:
        result = "Uninfected"

    return jsonify({
        "result": result,
        "confidence": float(prediction)
    })

if __name__ == "__main__":
    app.run()
