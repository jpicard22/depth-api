from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route("/depth", methods=["POST"])
def depth():
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    # Lance le script Python en arrière-plan
    command = ["python3", "depth.py", filename]
    subprocess.Popen(command)

    return jsonify({"message": f"Traitement lancé pour {filename}"})


@app.route("/check/<filename>", methods=["GET"])
def check_file(filename):
    output_path = os.path.join("public", "processed", filename)
    exists = os.path.exists(output_path)
    return jsonify({"ready": exists})


if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=10000)
