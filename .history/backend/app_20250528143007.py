from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from parsers import parse_file
from train_handler import train_model

UPLOAD_FOLDER = 'backend/uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'txt', 'pdf', 'docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uid = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uid + "_" + filename)
        file.save(filepath)

        # Parse the file
        parsed_data, file_type = parse_file(filepath)

        # Train model
        result = train_model(parsed_data, file_type, filename)

        return jsonify(result), 200

    return jsonify({"error": "Unsupported file type"}), 400

@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('backend/models', filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs('backend/uploads', exist_ok=True)
    os.makedirs('backend/models', exist_ok=True)
    os.makedirs('backend/logs', exist_ok=True)
    app.run(debug=True)
