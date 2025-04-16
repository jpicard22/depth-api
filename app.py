from flask import Flask, request, send_file
import os
import subprocess
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/depth", methods=["POST"])
def depth():
    if 'image' not in request.files:
        return "Aucune image envoyée", 400

    img_file = request.files['image']
    img_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{img_id}.jpg")
    output_path = os.path.join(PROCESSED_FOLDER, f"{img_id}_depth.png")

    img_file.save(input_path)

    try:
        result = subprocess.run(
            ['python', 'depth.py', input_path, output_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return f"Erreur lors du traitement : {result.stderr}", 500

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        return f"Erreur serveur : {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)