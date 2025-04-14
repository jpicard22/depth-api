from flask import Flask, request, jsonify
import os
from PIL import Image
import torch
import torchvision.transforms as T
import io
import base64

app = Flask(__name__)

print(f"✅ PORT Railway = {os.environ.get('PORT')}", file=sys.stderr)


# 📌 Chargement du modèle MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Bienvenue sur l'API de profondeur MiDaS !"})

@app.route("/", methods=["POST"])
def predict_depth():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400

    img_file = request.files["image"]
    img = Image.open(img_file).convert("RGB")

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    # Conversion du tenseur en image
    prediction_np = prediction.numpy()
    prediction_img = Image.fromarray((prediction_np / prediction_np.max() * 255).astype('uint8'))

    # Encodage en base64 pour retour JSON
    buffered = io.BytesIO()
    prediction_img.save(buffered, format="PNG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"depth_map": encoded_img})

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)