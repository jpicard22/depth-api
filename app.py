from flask import Flask, request, jsonify
import os
import base64
import uuid
from depth import generate_depth_map

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Route pour tester l'API (GET) ou générer la carte de profondeur (POST)
    """
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == "":
            return jsonify({'error': 'Empty filename'}), 400

        image_name = f"{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[-1].lower()}"
        input_path = os.path.join(UPLOAD_FOLDER, image_name)
        output_name = f"{uuid.uuid4().hex}_depth.png"
        output_path = os.path.join(PROCESSED_FOLDER, output_name)

        try:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            os.makedirs(PROCESSED_FOLDER, exist_ok=True)
            file.save(input_path)

            generate_depth_map(input_path, output_path)

            with open(output_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            os.remove(input_path)
            os.remove(output_path)

            return jsonify({'processed_image': encoded_string}), 200

        except Exception as e:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            return jsonify({'error': str(e)}), 500
    else:
        return 'API OK'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)