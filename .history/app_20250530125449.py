from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
import uuid
import json
import traceback

from config import *
from utils.file_parser import (
    allowed_file, parse_file, extract_text_docx, extract_text_pdf, extract_text_image
)
from utils.nlp_utils import summarize_text, extract_keywords, extract_named_entities
from utils.pdf_generator import generate_pdf_report, generate_pdf_summary
from utils.model_utils import train_model
from utils.logger import save_training_log

# --- Setup ---
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ANALYSIS_FOLDER = 'analysis'
MODEL_FOLDER = 'models'
PREVIEW_FOLDER = 'previews'
LOGS_FOLDER = 'logs'

for folder in [UPLOAD_FOLDER, MODEL_FOLDER, PREVIEW_FOLDER, LOGS_FOLDER, ANALYSIS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['PREVIEW_FOLDER'] = PREVIEW_FOLDER
app.config['ANALYSIS_FOLDER'] = ANALYSIS_FOLDER


    log_path = os.path.join(LOGS_FOLDER, f"{file_id}_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)

# --- Routes ---
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
            if df.shape[0] < 5 or df.shape[1] < 2:
                return jsonify({"error": "Insufficient data for training"}), 400

            preview_path = os.path.join(PREVIEW_FOLDER, f"{file_id}_preview.csv")
            df.head(10).to_csv(preview_path, index=False)

            df = df.apply(pd.to_numeric, errors='ignore')
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

            model_path = os.path.join(MODEL_FOLDER, f"{file_id}_{model_type}.pkl")
            joblib.dump(model, model_path)

            save_training_log(file_id, filename, model_type, accuracy, df.shape)

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

    return jsonify({"error": "Unsupported file type"}), 400

@app.route('/api/analyze-doc', methods=['POST'])
def analyze_doc():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        saved_filename = f"{file_id}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(filepath)

        try:
            ext = filename.rsplit('.', 1)[1].lower()

            if ext != 'docx':
                return jsonify({"error": "Only .docx supported for heading-style NLP analysis"}), 400

            doc = docx.Document(filepath)
            sections = extract_sections_by_style(doc)

            analysis = {}
            for heading, content in sections.items():
                if len(content) > 30:
                    summary = summarizer(content, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                else:
                    summary = content
                analysis[heading] = summary

            json_path = os.path.join(LOGS_FOLDER, f"{file_id}_analysis.json")
            with open(json_path, 'w') as f:
                json.dump(analysis, f, indent=4)

            pdf_path = os.path.join(LOGS_FOLDER, f"{file_id}_summary.pdf")
            generate_pdf_summary(analysis, pdf_path)

            return jsonify({
                "message": "Analysis completed",
                "sections": list(analysis.keys()),
                "json_log_url": f"/api/download/log/{file_id}_analysis.json",
                "pdf_summary_url": f"/api/download/log/{file_id}_summary.pdf"
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unsupported file type"}), 400
@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
    file.save(saved_path)

    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'docx':
        extracted = extract_text_docx(saved_path)
    elif ext == 'pdf':
        extracted = extract_text_pdf(saved_path)
    elif ext in ['png', 'jpg', 'jpeg']:
        extracted = extract_text_image(saved_path)
    else:
        return jsonify({"error": "Unsupported format"}), 400

    full_text = '\n'.join([text for _, text in extracted])
    summary = summarize_text(full_text)
    keywords = extract_keywords(full_text)
    entities = extract_named_entities(full_text)
    pdf_path = generate_pdf_report(file_id, extracted, summary, keywords, entities)

    analysis_log = {
        "file_id": file_id,
        "filename": filename,
        "summary": summary,
        "keywords": keywords,
        "named_entities": entities,
        "extracted_sections": extracted,
        "report_pdf": f"/api/download/report/{file_id}_summary.pdf",
        "timestamp": datetime.utcnow().isoformat()
    }

    log_path = os.path.join(app.config['ANALYSIS_FOLDER'], f"{file_id}_log.json")
    with open(log_path, 'w') as f:
        json.dump(analysis_log, f, indent=4)

    return jsonify(analysis_log)

@app.route('/api/download/<folder>/<filename>')
def download_file(folder, filename):
    if folder not in ['model', 'preview', 'log', 'analysis']:
        return jsonify({'error': 'Invalid folder'}), 400
    folder_map = {
        'model': MODEL_FOLDER,
        'preview': PREVIEW_FOLDER,
        'log': LOGS_FOLDER,
        'analysis': ANALYSIS_FOLDER
    }
    return send_from_directory(folder_map[folder], filename, as_attachment=True)

# --- Main ---
if __name__ == '__main__':
    app.run(debug=True)
