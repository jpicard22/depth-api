from flask import Flask, request, jsonify
from depth import generate_depth_map  # Assurez-vous que c'est le bon nom de votre fonction
import os
import uuid
import base64
import logging

logging.basicConfig(level=logging.DEBUG) # Activer les logs de débogage

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/", methods=["POST"])
def generate_depth():
    logging.info("Requête POST reçue.")
    if 'image' not in request.files:
        logging.error("Aucune image fournie.")
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == "":
        logging.error("Nom de fichier vide.")
        return jsonify({"error": "Empty filename"}), 400

    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(PROCESSED_FOLDER, f"{uuid.uuid4().hex}_depth.jpg")
    logging.info(f"Sauvegarde de l'image vers : {input_path}")
    try:
        file.save(input_path)
        logging.info(f"Image sauvegardée avec succès.")

        logging.info(f"Génération de la carte de profondeur vers : {output_path}")
        generate_depth_map(input_path, output_path)
        logging.info(f"Carte de profondeur générée avec succès.")

        with open(output_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        logging.info("Image de profondeur encodée en base64.")

        os.remove(input_path)
        os.remove(output_path)
        logging.info("Fichiers temporaires supprimés.")

        return jsonify({"processed_image": encoded_string}), 200

    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)