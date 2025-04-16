from flask import Flask, request, jsonify
# Assurez-vous que 'generate_depth_map' renvoie le chemin de l'image générée
from depth import generate_depth_map
import os
import uuid
import base64

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/", methods=["POST"])
def generate_depth():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Sauvegarder l'image
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(PROCESSED_FOLDER, f"{uuid.uuid4().hex}_depth.jpg") # Nom de fichier différent pour la profondeur
    file.save(input_path)

    # Générer la carte de profondeur
    try:
        generate_depth_map(input_path, output_path)

        # Lire l'image de profondeur et l'encoder en base64
        with open(output_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Supprimer les fichiers temporaires
        os.remove(input_path)
        os.remove(output_path)

        return jsonify({"processed_image": encoded_string}), 200
    except Exception as e:
        # Supprimer les fichiers en cas d'erreur
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Railway fournit PORT
    app.run(host='0.0.0.0', port=port)