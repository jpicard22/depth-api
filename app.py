from flask import Flask, request, jsonify
import os
import base64
import uuid
from depth import generate_depth_map  # Assure-toi que cette fonction est correcte et importée.

app = Flask(__name__)

# Dossiers pour les images
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])  # Autoriser les méthodes GET et POST
def index():
    """
    Route pour tester l'API (GET) ou générer la carte de profondeur (POST)
    """
    if request.method == 'POST':
        # Vérifie si une image est incluse dans la requête
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == "":
            return jsonify({'error': 'Empty filename'}), 400

        # Génère un nom unique pour l'image
        image_name = f"{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[-1].lower()}"
        input_path = os.path.join(UPLOAD_FOLDER, image_name)
        output_name = f"{uuid.uuid4().hex}_depth.png"
        output_path = os.path.join(PROCESSED_FOLDER, output_name)

        try:
            # Sauvegarde le fichier image dans le dossier "uploads"
            file.save(input_path)

            # Génération de la carte de profondeur
            generate_depth_map(input_path, output_path)

            # Ouverture de l'image générée pour la convertir en base64
            with open(output_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            # Suppression des fichiers temporaires après traitement
            os.remove(input_path)
            os.remove(output_path)

            # Retourner l'image générée en base64 dans la réponse
            return jsonify({'processed_image': encoded_string}), 200

        except Exception as e:
            # Suppression des fichiers en cas d'erreur
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            return jsonify({'error': str(e)}), 500
    else:
        # Test de l'API
        return 'API OK'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
