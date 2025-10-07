from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json
from sentence_transformers import SentenceTransformer
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_DIR = "intent_oos_model"  

clf = joblib.load(os.path.join(MODEL_DIR, "intent_oos_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
sbert_model = SentenceTransformer(os.path.join(MODEL_DIR, "sbert_model"))

with open(os.path.join(MODEL_DIR, "oos_config.json")) as f:
    config = json.load(f)

best_th = config["best_threshold"]
oos_name = config["oos_name"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    vec = sbert_model.encode([text])
    probs = clf.predict_proba(vec)[0]
    max_prob = float(probs.max())
    pred_label = clf.predict(vec)[0]
    intent = label_encoder.inverse_transform([pred_label])[0]

    if max_prob < best_th:
        intent = oos_name

    return jsonify({
        "intent": intent,
        "confidence": round(max_prob, 3)
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Intent Classification + OOS API is running!"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
