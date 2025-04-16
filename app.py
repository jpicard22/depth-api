from flask import Flask, request, jsonify
import os
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
    Route pour générer la carte de profondeur à partir de l'image envoyée
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400  # Vérification de la présence d'une image

    # Sauvegarde de l'image envoyée
    file = request.files['image']
    input_path = 'input.jpg'  # Chemin temporaire pour l'image
    output_path = 'output.png'  # Chemin de la sortie (carte de profondeur)
    file.save(input_path)  # Sauvegarde de l'image

    try:
        # Génération de la carte de profondeur
        generate_depth_map(input_path, output_path)

        # Retour de la réponse avec le statut de succès
        return jsonify({'status': 'success', 'output_path': output_path})
    except Exception as e:
        # Retour de la réponse avec l'erreur éventuelle
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Récupération du port de Railway ou utilisation du port par défaut (8080)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)  # Lancer l'API Flask sur toutes les interfaces réseau
