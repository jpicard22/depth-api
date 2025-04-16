import os
import base64
from flask import Flask, request, jsonify
from depth import generate_depth_map

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """
    Route pour tester l'API
    """
    return 'API OK'

@app.route('/generate', methods=['POST'])
def generate():
    """
    Route pour générer la carte de profondeur à partir de l'image envoyée et la renvoyer en base64.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    input_path = 'input.jpg'
    output_path = 'output.png'

    try:
        file.save(input_path)
        generate_depth_map(input_path, output_path)

        # Lire l'image de profondeur et l'encoder en base64
        with open(output_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Supprimer les fichiers temporaires
        os.remove(input_path)
        os.remove(output_path)

        return jsonify({'processed_image': encoded_string}), 200

    except Exception as e:
        # Supprimer les fichiers en cas d'erreur
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)