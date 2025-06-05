from flask import Flask, request, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename

# Optional parsers
import docx
import PyPDF2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        doc = docx.Document(filepath)
        text = [p.text for p in doc.paragraphs if p.text.strip()]
        # Add logic to parse structured data if needed
        raise NotImplementedError("DOCX parser needs structured format handling.")
    elif ext == 'pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
            # Add logic to parse structured data if needed
        raise NotImplementedError("PDF parser needs structured format handling.")
    else:
        raise ValueError("Unsupported file type.")

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = parse_file(filepath)
            # Convert to numeric if needed (basic assumption)
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

            return jsonify({
                "message": "Model trained successfully",
                "accuracy": accuracy
            })

        except NotImplementedError as e:
            return jsonify({"error": str(e)}), 501
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unsupported file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)
