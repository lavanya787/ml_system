from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from supabase import create_client, Client
import bcrypt
import uuid
from datetime import datetime
import re
import os
from werkzeug.utils import secure_filename
import pandas as pd
import os
import uuid
import json
import joblib
import faiss

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from auth_routes import auth_bp, login_required
from backend.utils.file_parser import (
    allowed_file, parse_file, extract_text_docx, extract_text_pdf, extract_text_image
)
from backend.utils.nlp_utils import summarize_text, extract_keywords, extract_named_entities
from backend.utils.pdf_generator import generate_pdf_report, generate_pdf_summary
from backend.utils.model_utils import train_model
from backend.utils.logger import save_training_log

app = Flask(__name__)
CORS(app)

# Supabase setup
SUPABASE_URL = "https://uctyxchurvievzvhthru.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVjdHl4Y2h1cnZpZXZ6dmh0aHJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU0ODk0MDIsImV4cCI6MjA2MTA2NTQwMn0.YytXX-q4QDO_vY9f1e_P-UWc6v6860kcsbe_bTZVgCI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
app.config['LOGS_FOLDER'] = LOGS_FOLDER

# In-memory chat history
conversation_store = {}
models={}
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to validate UUID
def is_valid_uuid(val):
    uuid_pattern = re.compile(
        r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    )
    return bool(uuid_pattern.match(val))
def save_faiss_index(file_id, texts):
    embeddings = embedding_model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f"{ANALYSIS_FOLDER}/{file_id}.index")
    with open(f"{ANALYSIS_FOLDER}/{file_id}.json", 'w') as f:
        json.dump(texts, f)

def load_faiss_index(file_id):
    try:
        index = faiss.read_index(f"{ANALYSIS_FOLDER}/{file_id}.index")
        with open(f"{ANALYSIS_FOLDER}/{file_id}.json", 'r') as f:
            texts = json.load(f)
        return index, texts
    except:
        return None, None

def generate_answer_with_context(question, chunks):
    context = "\n".join(chunks)
    return f"Answer based on: {context[:300]}...\n\n[Mock Answer] for: {question}"

# --- Routes ---
# Serve the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Check if user already exists
    user = supabase.table('users').select('*').eq('email', email).execute()
    print(f"Checking if user exists with email {email}: {user.data}")  # Debug log
    if user.data:
        return jsonify({'message': 'User already exists'}), 400

    # Hash password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Generate unique user ID
    user_id = str(uuid.uuid4())
    print(f"Generated user_id for registration: {user_id}")  # Debug log
    if not is_valid_uuid(user_id):
        print(f"ERROR: Generated user_id is not a valid UUID: {user_id}")
        return jsonify({'message': 'Internal server error: Invalid user_id generated'}), 500

    # Store user in Supabase
    registration_time = datetime.utcnow().isoformat()
    new_user = {
        'user_id': user_id,
        'username': username,
        'email': email,
        'password': hashed_password,
        'registration_time': registration_time
    }
    print(f"Storing user in Supabase: {new_user}")  # Debug log
    try:
        supabase.table('users').insert(new_user).execute()
    except Exception as e:
        print(f"Error inserting user into Supabase: {e}")
        return jsonify({'message': 'Error storing user in database'}), 500


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # Fetch user from Supabase
    user = supabase.table('users').select('*').eq('email', email).execute()
    print(f"Supabase response for user lookup: {user.data}")  # Debug log
    if not user.data:
        return jsonify({'message': 'User not found'}), 404

    user_data = user.data[0]
    # Verify password
    if not bcrypt.checkpw(password.encode('utf-8'), user_data['password'].encode('utf-8')):
        return jsonify({'message': 'Incorrect password'}), 401

    # Update last login time
    login_time = datetime.utcnow().isoformat()
    try:
        supabase.table('users').update({'last_login': login_time}).eq('email', email).execute()
    except Exception as e:
        print(f"Error updating last login time in Supabase: {e}")
        return jsonify({'message': 'Error updating login time'}), 500

    # Redirect to Streamlit app with user_id as a query parameter
    user_id = user_data['user_id']
    print(f"Retrieved user_id for login: {user_id}")  # Debug log
    if not is_valid_uuid(user_id):
        print(f"ERROR: Retrieved user_id is not a valid UUID: {user_id}")
        return jsonify({'message': 'Internal server error: Invalid user_id retrieved'}), 500

    streamlit_url = f"http://localhost:8501/?user_id={user_id}"
    print(f"Redirecting to Streamlit app after login: {streamlit_url}")  # Debug log
    return jsonify({'message': 'Login successful', 'redirect': streamlit_url}), 200
# --- File Upload Routes ---
@app.route("/api/upload", methods=["POST"])
def upload_file():
    """File upload endpoint"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        saved_filename = f"{file_id}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
        file.save(file_path)
        
        # For text-based files, create FAISS index for chat functionality
        if filename.lower().endswith(('.txt', '.pdf', '.docx')):
            try:
                if filename.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                elif filename.lower().endswith('.pdf'):
                    content = extract_text_pdf(file_path)
                    chunks = [text for _, text in content if text.strip()]
                elif filename.lower().endswith('.docx'):
                    content = extract_text_docx(file_path)
                    chunks = [text for _, text in content if text.strip()]
                
                if chunks:
                    save_faiss_index(file_id, chunks)
            except Exception as e:
                print(f"Error creating index: {e}")
        
        return jsonify({"file_id": file_id, "filename": saved_filename}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

# --- Chat Routes ---
@app.route("/api/chat", methods=["POST"])
def chat_with_file():
    """Chat with uploaded file using RAG"""
    try:
        data = request.get_json()
        file_id = data.get("file_id")
        question = data.get("question")

        if not file_id or not question:
            return jsonify({"error": "file_id and question required"}), 400

        # Load conversation history
        history = conversation_store.get(file_id, [])
        
        # Load FAISS index
        index, texts = load_faiss_index(file_id)
        if not index:
            return jsonify({"error": "No searchable content found for this file"}), 404

        # Search for relevant chunks
        question_vec = embedding_model.encode([question])
        D, I = index.search(question_vec, 3)
        top_chunks = [texts[i] for i in I[0] if i < len(texts)]

        # Generate answer
        answer = generate_answer_with_context(question, top_chunks)
        
        # Update conversation history
        history.append({"question": question, "answer": answer})
        conversation_store[file_id] = history

        return jsonify({"answer": answer, "history": history}), 200
    except Exception as e:
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

# --- File Analysis Routes ---
@app.route("/api/preview", methods=["POST"])
def preview_file():
    """Preview CSV file data"""
    try:
        data = request.json
        filename = data.get("filename")
        
        if not filename:
            return jsonify({"error": "Filename required"}), 400
            
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        df = parse_file(file_path)
        preview_path = os.path.join(PREVIEW_FOLDER, f"{filename}_preview.csv")
        df.head(10).to_csv(preview_path, index=False)
        
        return jsonify({
            "columns": df.columns.tolist(),
            "preview_path": preview_path,
            "shape": df.shape
        }), 200
    except Exception as e:
        return jsonify({"error": f"Preview error: {str(e)}"}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    """Analyze document and extract information"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(saved_path)

        # Extract text based on file type
        ext = filename.rsplit('.', 1)[1].lower()
        if ext == 'docx':
            extracted = extract_text_docx(saved_path)
        elif ext == 'pdf':
            extracted = extract_text_pdf(saved_path)
        elif ext in ['png', 'jpg', 'jpeg']:
            extracted = extract_text_image(saved_path)
        else:
            return jsonify({"error": "Unsupported format"}), 400

        # Process extracted text
        full_text = '\n'.join([text for _, text in extracted])
        
        if not full_text.strip():
            return jsonify({"error": "No text could be extracted from the file"}), 400
            
        summary = summarize_text(full_text)
        keywords = extract_keywords(full_text)
        entities = extract_named_entities(full_text)
        
        # Generate PDF report
        pdf_path = generate_pdf_report(file_id, extracted, summary, keywords, entities)

        # Create analysis log
        analysis_log = {
            "file_id": file_id,
            "filename": filename,
            "summary": summary,
            "keywords": keywords,
            "named_entities": entities,
            "extracted_sections": len(extracted),
            "report_pdf": f"/api/download/analysis/{file_id}_summary.pdf",
            "timestamp": datetime.utcnow().isoformat()
        }

        # Save analysis log
        log_path = os.path.join(app.config['ANALYSIS_FOLDER'], f"{file_id}_log.json")
        with open(log_path, 'w') as f:
            json.dump(analysis_log, f, indent=4)

        return jsonify(analysis_log), 200
    except Exception as e:
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500

# --- Machine Learning Routes ---
@app.route('/api/train', methods=['POST'])
def train_model():
    """Train machine learning model"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    model_type = request.form.get('model_type', 'random_forest')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        saved_filename = f"{file_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)

        # Parse file
        df = parse_file(filepath)
        if df.shape[0] < 5 or df.shape[1] < 2:
            return jsonify({"error": "Insufficient data for training"}), 400

        # Create preview
        preview_path = os.path.join(PREVIEW_FOLDER, f"{file_id}_preview.csv")
        df.head(10).to_csv(preview_path, index=False)

        # Prepare data (assuming last column is target)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Handle non-numeric data
        X = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        if model_type == "random_forest":
            model = RandomForestClassifier(random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            return jsonify({"error": "Unsupported model type"}), 400

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save model
        model_path = os.path.join(MODEL_FOLDER, f"{file_id}_{model_type}.pkl")
        joblib.dump(model, model_path)

        # Store model info for predictions
        models[file_id] = {
            'model': model,
            'columns': X.columns.tolist(),
            'model_type': model_type
        }

        # Save training log
        save_training_log(file_id, filename, model_type, accuracy, df.shape)

        return jsonify({
            "message": "Model trained successfully",
            "file_id": file_id,
            "accuracy": accuracy,
            "model_download_url": f"/api/download/model/{file_id}_{model_type}.pkl",
            "preview_download_url": f"/api/download/preview/{file_id}_preview.csv"
        }), 200

    except Exception as e:
        return jsonify({"error": f"Training error: {str(e)}"}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using trained model"""
    try:
        json_data = request.json
        file_id = json_data.get("file_id")
        features = json_data.get("features")

        if not file_id or file_id not in models:
            return jsonify({"error": "Model not found or not trained"}), 400
        
        if not features or not isinstance(features, dict):
            return jsonify({"error": "Features must be provided as a dictionary"}), 400

        model_info = models[file_id]
        model = model_info['model']
        expected_features = model_info['columns']

        # Check for missing features
        missing_features = [f for f in expected_features if f not in features]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Prepare input
        input_vector = [features[f] for f in expected_features]
        input_df = pd.DataFrame([input_vector], columns=expected_features)

        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get prediction probability if available
        try:
            proba = model.predict_proba(input_df)[0].tolist()
        except:
            proba = None

        return jsonify({
            "prediction": str(prediction),
            "probability": proba
        }), 200

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# --- Utility Routes ---
@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Record user feedback"""
    try:
        data = request.get_json()
        file_id = data.get("file_id")
        question = data.get("question")
        answer = data.get("answer")
        feedback_type = data.get("feedback")  # 'positive' or 'negative'

        log = {
            "file_id": file_id,
            "question": question,
            "answer": answer,
            "feedback": feedback_type,
            "timestamp": datetime.utcnow().isoformat()
        }

        feedback_path = os.path.join(LOGS_FOLDER, f"{file_id}_feedback.json")
        with open(feedback_path, "a") as f:
            f.write(json.dumps(log) + "\n")

        return jsonify({"message": "Feedback recorded"}), 200
    except Exception as e:
        return jsonify({"error": f"Feedback error: {str(e)}"}), 500

@app.route('/api/download/<folder>/<filename>')
def download_file(folder, filename):
    """Download files from various folders"""
    folder_map = {
        'model': MODEL_FOLDER,
        'preview': PREVIEW_FOLDER,
        'log': LOGS_FOLDER,
        'analysis': ANALYSIS_FOLDER
    }
    
    if folder not in folder_map:
        return jsonify({'error': 'Invalid folder'}), 400
    
    try:
        return send_from_directory(folder_map[folder], filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/logout', methods=['POST'])
def logout():
    """User logout"""
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'}), 200

# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)