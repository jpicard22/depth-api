from flask import Flask, render_template, request, send_from_directory
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Créer le dossier uploads s'il n'existe pas
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    depth_image = None

    if request.method == "POST":
        if "image" not in request.files:
            return "Aucun fichier sélectionné", 400
        
        file = request.files["image"]
        if file.filename == "":
            return "Aucun fichier sélectionné", 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Générer la carte de profondeur
        depth_path = os.path.join(app.config["UPLOAD_FOLDER"], "depth_" + file.filename)
        subprocess.run(["python", "depth.py", file_path, depth_path])

        uploaded_image = file.filename
        depth_image = "depth_" + file.filename

    return render_template("index.html", uploaded_image=uploaded_image, depth_image=depth_image)

if __name__ == "__main__":
    app.run(debug=True)
