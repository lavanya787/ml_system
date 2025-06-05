from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
import uuid
import joblib
from werkzeug.utils import secure_filename
import json
from datetime import datetime
# Parsers
import docx
import PyPDF2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LOGS_FOLDER = 'logs'
os.makedirs(LOGS_FOLDER, exist_ok=True)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
PREVIEW_FOLDER = 'previews'

for folder in [UPLOAD_FOLDER, MODEL_FOLDER, PREVIEW_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['PREVIEW_FOLDER'] = PREVIEW_FOLDER

ALLOWED_EXTENSIONS = {'csv', 'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_file(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()

    if ext == 'csv':
        return pd.read_csv(filepath)
    elif ext == 'txt':
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return pd.DataFrame([line.strip().split(',') for line in lines[1:]], columns=lines[0].strip().split(','))
    elif ext == 'docx':
        raise NotImplementedError("Structured DOCX parsing not implemented.")
    elif ext == 'pdf':
        raise NotImplementedError("Structured PDF parsing not implemented.")
    else:
        raise ValueError("Unsupported file type")

@app.route('/api/train', methods=['POST'])
def train_model():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    model_type = request.form.get('model_type', 'random_forest')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        saved_filename = f"{file_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)

        try:
            df = parse_file(filepath)

            # ✅ File validation
            if df.shape[0] < 5 or df.shape[1] < 2:
                return jsonify({"error": "Insufficient data for training"}), 400

            # Save dataset preview
            preview_path = os.path.join(app.config['PREVIEW_FOLDER'], f"{file_id}_preview.csv")
            df.head(10).to_csv(preview_path, index=False)

            df = df.apply(pd.to_numeric, errors='ignore')  # Try to cast if possible
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            if model_type == "random_forest":
                model = RandomForestClassifier()
            elif model_type == "logistic_regression":
                model = LogisticRegression(max_iter=1000)
            else:
                return jsonify({"error": "Unsupported model type"}), 400

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # ✅ Save model
            model_path = os.path.join(app.config['MODEL_FOLDER'], f"{file_id}_{model_type}.pkl")
            joblib.dump(model, model_path)

            return jsonify({
                "message": "Model trained successfully",
                "accuracy": accuracy,
                "model_download_url": f"/api/download/model/{file_id}_{model_type}.pkl",
                "preview_download_url": f"/api/download/preview/{file_id}_preview.csv"
            })

        except NotImplementedError as e:
            return jsonify({"error": str(e)}), 501
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Unsupported file type"}), 400
def save_training_log(file_id, original_filename, model_type, accuracy, dataset_shape):
    log_data = {
        "file_id": file_id,
        "original_filename": original_filename,
        "model_type": model_type,
        "accuracy": accuracy,
        "num_rows": dataset_shape[0],
        "num_columns": dataset_shape[1],
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }

    log_path = os.path.join(LOGS_FOLDER, f"{file_id}_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)

    return log_path

# Inside /api/train after successful training and saving model:
save_training_log(
    file_id=file_id,
    original_filename=filename,
    model_type=model_type,
    accuracy=accuracy,
    dataset_shape=df.shape
)
@app.route('/api/download/model/<filename>', methods=['GET'])
def download_model(filename):
    return send_from_directory(app.config['MODEL_FOLDER'], filename, as_attachment=True)

@app.route('/api/download/preview/<filename>', methods=['GET'])
def download_preview(filename):
    return send_from_directory(app.config['PREVIEW_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
