# backend/routes.py

import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from backend.file_utils import allowed_file
from backend.tasks import train_model
from backend.config import UPLOAD_FOLDER

api = Blueprint('api', __name__)

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
