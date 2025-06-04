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
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVjdHl4Y2h1cnZpZXZ6dmh0aHJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU0ODk0MDIsImV4cCI6MjA2MTA2NTQwMn0.YytXX-q4QDO_vY9f1e_P-UWc6v6860kcsbe_bTZVgCI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helper Functions ---

def is_valid_uuid(val):
    pattern = r'^[0-9a-fA-F\-]{36}$'
    return re.fullmatch(pattern, val) is not None

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
st.set_page_config(page_title="Data Upload and Preprocess", layout="centered")
st.title("📤 File Upload Interface")

# Get query parameter
query_params = st.query_params
user_id = query_params.get("user_id", "")
if isinstance(user_id, list): user_id = user_id[0] if user_id else ""

# --- Logout ---
with st.sidebar:
    st.header("🔐 Session")
    if st.button("🚪 Logout"):
        st.experimental_set_query_params()
        st.success("You have been logged out.")
        st.markdown("[Return to Login](http://localhost:5000)", unsafe_allow_html=True)
        st.stop()

# --- User Validation ---
if not user_id or not is_valid_uuid(user_id):
    st.error("Invalid or missing User ID.")
    st.markdown("[Go to Login/Register](http://localhost:5000)", unsafe_allow_html=True)
    st.stop()

# --- Upload Section ---
uploaded_file = st.file_uploader("📎 Upload your file", type=["csv", "xlsx", "xls", "pdf", "txt"])

if uploaded_file:
    file_id = str(uuid.uuid4())
    file_type = uploaded_file.type
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{uploaded_file.name}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess file
    try:
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
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

    # Save preprocessed data
    os.makedirs("preprocessed", exist_ok=True)
    cleaned_filename = uploaded_file.name.rsplit('.', 1)[0] + "_preprocessed.csv"
    storage_path = f"preprocessed/{cleaned_filename}"
    df_clean = pd.DataFrame({"cleaned_text": [preprocessed_data]})
    df_clean.to_csv(storage_path, index=False)

    # Upload metadata to Supabase
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

        st.success("✅ File uploaded and processed successfully!")

    except Exception as e:
        st.error(f"Error saving to Supabase: {e}")
        st.stop()

# --- Optional User Message ---
user_message = st.text_input("💬 Any comments or messages?")
if user_message:
    st.info(f"📩 Message received: {user_message}")
