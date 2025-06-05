# backend/routes.py

import os
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from backend.file_utils import allowed_file
from backend.tasks import train_model
from backend.config import UPLOAD_FOLDER, MODEL_FOLDER, LOG_FOLDER

api = Blueprint('api', __name__)

# 🧠 Existing Train Endpoint
@api.route("/api/train", methods=["POST"])
def train_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    model_type = request.form.get("model_type", "default")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = train_model(filepath, model_type)
        return jsonify(result), 200

    return jsonify({"error": "Invalid file type"}), 400

# ✅ DOWNLOAD ENDPOINT for model or log
@api.route("/api/download/<folder>/<filename>", methods=["GET"])
def download_file(folder, filename):
    if folder not in ["trained_models", "logs", "uploads"]:
        return jsonify({"error": "Invalid folder"}), 400

    directory = {
        "trained_models": MODEL_FOLDER,
        "logs": LOG_FOLDER,
        "uploads": UPLOAD_FOLDER
    }.get(folder)

    if not os.path.exists(os.path.join(directory, filename)):
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(directory, filename, as_attachment=True)

# ✅ METADATA: List all files in each folder
@api.route("/api/metadata", methods=["GET"])
def metadata():
    def list_files(folder):
        return sorted([os.path.basename(f) for f in glob.glob(f"{folder}/*")])

    return jsonify({
        "datasets": list_files(UPLOAD_FOLDER),
        "models": list_files(MODEL_FOLDER),
        "logs": list_files(LOG_FOLDER)
    }), 200
