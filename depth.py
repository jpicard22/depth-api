from flask import Flask, request, jsonify, send_file
import os
import uuid
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return "✅ API MiDaS Depth fonctionne."

@app.route("/", methods=["POST"])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide."}), 400

    # Noms de fichiers uniques
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.jpg")
    output_path = os.path.join(PROCESSED_FOLDER, f"{unique_id}.png")

    try:
        # Sauvegarde du fichier uploadé
        file.save(input_path)

        # Appel du script depth.py avec subprocess
        result = subprocess.run(
            ["python3", "depth.py", input_path, output_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return jsonify({"error": "Erreur du script", "details": result.stderr}), 500

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": "Exception côté serveur", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
