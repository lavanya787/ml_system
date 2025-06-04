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
SUPABASE_KEY = "your_actual_supabase_key_here"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Function to validate UUID
def is_valid_uuid(val):
    uuid_pattern = re.compile(
        r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    )
    return bool(uuid_pattern.match(val))

# Preprocessing functions
def general_preprocessing(text):
    if isinstance(text, bytes):
        result = chardet.detect(text)
        encoding = result['encoding']
        text = text.decode(encoding, errors='replace')
    
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def preprocess_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return general_preprocessing(text)

def preprocess_excel(file):
    df = pd.read_excel(file)
    df = df.dropna(how='all').dropna(axis=1, how='all')
    df = df.fillna('')
    df.columns = [col.lower().strip() for col in df.columns]
    return general_preprocessing(df.to_string(index=False))

def preprocess_csv(file):
    df = pd.read_csv(file)
    df = df.fillna('')
    df.columns = [col.lower().strip() for col in df.columns]
    return general_preprocessing(df.to_string(index=False))

def preprocess_text(file):
    text = file.read().decode('utf-8', errors='replace')
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return general_preprocessing(text)

# Chatbot logic
def chatbot_response(user_message, preprocessed_data=None):
    if preprocessed_data:
        return f"Got your message: '{user_message}'. I’ve already processed the file."
    return f"Got your message: '{user_message}'. Please upload a file first."

# Streamlit app
st.set_page_config(page_title="Chatbot Upload", layout="centered")
st.title("📄 Upload File and Chat 💬")

# Validate user_id
query_params = st.query_params
user_id = query_params.get("user_id", "")
if isinstance(user_id, list):
    user_id = user_id[0] if user_id else ""

if not user_id or not is_valid_uuid(user_id):
    st.error("Invalid or missing User ID. Please log in again.")
    st.markdown("[🔐 Go to Login](http://localhost:5000)", unsafe_allow_html=True)
    st.stop()

# Logout button
st.markdown("[🚪 Logout](http://localhost:5000)", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("📎 Upload a CSV, Excel, PDF, or TXT file", type=['csv', 'xlsx', 'xls', 'pdf', 'txt'])

preprocessed_data = None
if uploaded_file:
    file_id = str(uuid.uuid4())
    file_type = uploaded_file.type
    file_path = f"uploads/{uploaded_file.name}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess
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

    if preprocessed_data:
        preprocess_time = datetime.utcnow().isoformat()
        storage_path = f"preprocessed/{uploaded_file.name}.txt"
        os.makedirs("preprocessed", exist_ok=True)
        with open(storage_path, "w", encoding="utf-8") as f:
            f.write(preprocessed_data)

        try:
            supabase.table("file_uploads").insert({
                "file_id": file_id,
                "user_id": user_id,
                "file_name": uploaded_file.name,
                "file_path": file_path,
                "upload_time": preprocess_time
            }).execute()

            supabase.table("data_preprocessing_logs").insert({
                "preprocess_id": str(uuid.uuid4()),
                "file_id": file_id,
                "user_id": user_id,
                "status": "completed",
                "preprocess_time": preprocess_time
            }).execute()

            supabase.table("datasets").insert({
                "dataset_id": str(uuid.uuid4()),
                "file_id": file_id,
                "user_id": user_id,
                "dataset_name": uploaded_file.name,
                "storage_path": storage_path,
                "created_at": preprocess_time
            }).execute()

            st.success("✅ File uploaded and processed successfully!")
        except Exception as e:
            st.error(f"❌ Error saving data: {e}")
            st.stop()

# Chat section
st.header("💬 Ask the Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_reply = chatbot_response(user_input, preprocessed_data)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
