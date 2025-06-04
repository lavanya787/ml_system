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

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Supabase setup
SUPABASE_URL = "https://uctyxchurvievzvhthru.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVjdHl4Y2h1cnZpZXZ6dmh0aHJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU0ODk0MDIsImV4cCI6MjA2MTA2NTQwMn0.YytXX-q4QDO_vY9f1e_P-UWc6v6860kcsbe_bTZVgCI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Preprocessing functions
def general_preprocessing(text):
    # Encoding handling (detect and convert to UTF-8)
    if isinstance(text, bytes):
        result = chardet.detect(text)
        encoding = result['encoding']
        text = text.decode(encoding, errors='replace')
    
    # Remove noise (extra spaces, special characters)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def preprocess_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    
    # Handle layout issues (basic cleanup)
    text = re.sub(r'\n+', ' ', text)
    return general_preprocessing(text)

def preprocess_excel(file):
    df = pd.read_excel(file)
    # Drop empty rows/columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    # Handle missing values (replace NaN with empty string)
    df = df.fillna('')
    # Standardize headers
    df.columns = [col.lower().strip() for col in df.columns]
    # Convert to text
    text = df.to_string(index=False)
    return general_preprocessing(text)

def preprocess_csv(file):
    df = pd.read_csv(file)
    # Handle missing values
    df = df.fillna('')
    # Normalize column names
    df.columns = [col.lower().strip() for col in df.columns]
    # Convert to text
    text = df.to_string(index=False)
    return general_preprocessing(text)

def preprocess_text(file):
    text = file.read().decode('utf-8', errors='replace')
    # Clean non-UTF characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return general_preprocessing(text)

# Streamlit app
st.title("File Upload and Preprocessing")

# Get user_id from query parameters
query_params = st.query_params
user_id = query_params.get("user_id", [""])[0]
print(f"Received user_id: {user_id}")  # Debug log

if not user_id:
    st.error("User ID not found. Please log in or register first.")
else:
    st.write(f"Logged in as User ID: {user_id}")
    uploaded_file = st.file_uploader("Upload a file (CSV, Excel, PDF, Text)", type=['csv', 'xlsx', 'xls', 'pdf', 'txt'])

    if uploaded_file:
        # Generate a unique file_id
        file_id = str(uuid.uuid4())
        
        # Save the uploaded file temporarily
        file_path = f"uploads/{uploaded_file.name}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Preprocess based on file type
        file_type = uploaded_file.type
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
            preprocessed_data = None
        
        if preprocessed_data:
            # Store preprocessed data (e.g., in a file or directly in DB)
            storage_path = f"preprocessed/{uploaded_file.name}.txt"
            os.makedirs("preprocessed", exist_ok=True)
            with open(storage_path, "w", encoding="utf-8") as f:
                f.write(preprocessed_data)
            
            # Log preprocessing metadata in data_preprocessing_logs
            preprocess_time = datetime.utcnow().isoformat()
            preprocess_id = str(uuid.uuid4())
            file_data = {
                "preprocess_id": preprocess_id,
                "file_id": file_id,
                "user_id": user_id,
                "status": "completed",
                "preprocess_time": preprocess_time
            }
            supabase.table("data_preprocessing_logs").insert(file_data).execute()
            
            # Store dataset info in datasets table
            dataset_id = str(uuid.uuid4())
            dataset_data = {
                "dataset_id": dataset_id,
                "file_id": file_id,
                "user_id": user_id,
                "dataset_name": uploaded_file.name,
                "storage_path": storage_path,
                "created_at": preprocess_time
            }
            supabase.table("datasets").insert(dataset_data).execute()
            
            # Optionally, log file upload in file_uploads table (if needed)
            upload_data = {
                "file_id": file_id,
                "user_id": user_id,
                "file_name": uploaded_file.name,
                "file_path": file_path,
                "upload_time": preprocess_time
            }
            supabase.table("file_uploads").insert(upload_data).execute()
            
            st.success(f"File {uploaded_file.name} preprocessed and stored successfully!")
            st.write("Preprocessed data preview:")
            st.text(preprocessed_data[:500])  # Show first 500 characters