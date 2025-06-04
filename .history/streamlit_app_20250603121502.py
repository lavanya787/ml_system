import streamlit as st
import pandas as pd
import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from supabase import create_client, Client
import os
from datetime import datetime
import chardet
import uuid

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Supabase credentials
SUPABASE_URL = "https://uctyxchurvievzvhthru.supabase.co"
SUPABASE_KEY = "your_supabase_key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Validate UUID
def is_valid_uuid(val):
    pattern = r'^[0-9a-fA-F\-]{36}$'
    return re.fullmatch(pattern, val) is not None

# Preprocessing steps
def general_preprocessing(text):
    if isinstance(text, bytes):
        encoding = chardet.detect(text)['encoding']
        text = text.decode(encoding, errors='replace')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def preprocess_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = " ".join([page.extract_text() or "" for page in reader.pages])
    return general_preprocessing(text)

def preprocess_excel(file):
    df = pd.read_excel(file).fillna('').dropna(how='all').dropna(axis=1, how='all')
    df.columns = [c.lower().strip() for c in df.columns]
    return general_preprocessing(df.to_string(index=False))

def preprocess_csv(file):
    df = pd.read_csv(file).fillna('')
    df.columns = [c.lower().strip() for c in df.columns]
    return general_preprocessing(df.to_string(index=False))

def preprocess_text(file):
    text = file.read().decode('utf-8', errors='replace')
    return general_preprocessing(text)

# --- Streamlit UI ---
st.title("File Upload Interface")

# Extract user_id from query
query_params = st.query_params
user_id = query_params.get("user_id", "")
if isinstance(user_id, list): user_id = user_id[0] if user_id else ""

if not user_id or not is_valid_uuid(user_id):
    st.error("Invalid or missing User ID.")
    st.markdown("[Go to Login/Register](http://localhost:5000)", unsafe_allow_html=True)
    st.stop()

uploaded_file = st.file_uploader("Upload your file (CSV, Excel, PDF, Text)", type=["csv", "xlsx", "xls", "pdf", "txt"])

if uploaded_file:
    file_id = str(uuid.uuid4())
    file_type = uploaded_file.type
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{uploaded_file.name}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess based on file type
    if file_type == "application/pdf":
        preprocessed_data = preprocess_pdf(uploaded_file)
    elif file_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        preprocessed_data = preprocess_excel(uploaded_file)
    elif file_type == "text/csv":
        preprocessed_data = preprocess_csv(uploaded_file)
    elif file_type == "text/plain":
        preprocessed_data = preprocess_text(uploaded_file)
    else:
        st.error("Unsupported file type!")
        st.stop()

    # Save preprocessed data
    os.makedirs("preprocessed", exist_ok=True)
    cleaned_filename = uploaded_file.name.rsplit('.', 1)[0] + "_preprocessed.csv"
    storage_path = f"preprocessed/{cleaned_filename}"
    df_clean = pd.DataFrame({"cleaned_text": [preprocessed_data]})
    df_clean.to_csv(storage_path, index=False)

    # Store in Supabase
    timestamp = datetime.utcnow().isoformat()

    try:
        supabase.table("file_uploads").insert({
            "file_id": file_id,
            "user_id": user_id,
            "file_name": uploaded_file.name,
            "file_path": file_path,
            "upload_time": timestamp
        }).execute()

        supabase.table("data_preprocessing_logs").insert({
            "preprocess_id": str(uuid.uuid4()),
            "file_id": file_id,
            "user_id": user_id,
            "status": "completed",
            "preprocess_time": timestamp
        }).execute()

        supabase.table("datasets").insert({
            "dataset_id": str(uuid.uuid4()),
            "file_id": file_id,
            "user_id": user_id,
            "dataset_name": cleaned_filename,
            "storage_path": storage_path,
            "created_at": timestamp
        }).execute()

        # ✅ Success Message only
        st.success("✅ File uploaded successfully!")

    except Exception as e:
        st.error(f"Upload failed: {e}")
        st.stop()

# --- Minimal message box ---
user_message = st.text_input("Your Message:")
if user_message:
    st.write("📩 Message received.")
