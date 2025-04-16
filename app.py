from flask import Flask, request, send_file, jsonify
from PIL import Image
import torch
import numpy as np
import io
import cv2

app = Flask(__name__)

# 🔥 Chargement du modèle MiDaS depuis torch.hub (pas besoin de fichier .pt local)
model_type = "DPT_Large"  # ce modèle fonctionne bien avec 'transforms.dpt_transform'
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

# 🎯 Transformations d’image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

@app.route('/', methods=['POST'])
def depth_map():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    img_file = request.files['image']
    image = Image.open(img_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)
        depth = prediction.squeeze().cpu().numpy()

    # 🧠 Normalisation de la carte de profondeur
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    depth_img = (depth_normalized * 255).astype(np.uint8)

    # 💾 Conversion en PNG
    _, buffer = cv2.imencode(".png", depth_img)
    img_bytes = io.BytesIO(buffer.tobytes())

    return send_file(img_bytes, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
