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
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Supabase setup
SUPABASE_URL = "https://uctyxchurvievzvhthru.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVjdHl4Y2h1cnZpZXZ6dmh0aHJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU0ODk0MDIsImV4cCI6MjA2MTA2NTQwMn0.YytXX-q4QDO_vY9f1e_P-UWc6v6860kcsbe_bTZVgCI"
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
    text = re.sub(r'[^\w\s.]', '', text)
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
    text = re.sub(r'\n+', ' ', text)
    return general_preprocessing(text)

def preprocess_excel(file):
    df = pd.read_excel(file)
    df = df.dropna(how='all').dropna(axis=1, how='all')
    df = df.fillna('')
    df.columns = [col.lower().strip() for col in df.columns]
    text = df.to_string(index=False)
    return general_preprocessing(text)

def preprocess_csv(file):
    df = pd.read_csv(file)
    df = df.fillna('')
    df.columns = [col.lower().strip() for col in df.columns]
    text = df.to_string(index=False)
    return general_preprocessing(text)

def preprocess_text(file):
    text = file.read().decode('utf-8', errors='replace')
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return general_preprocessing(text)
def smart_clean(  text):
    return ' '.join(re.sub(r'[^\w\s.]', '', word) for word in text.split())

text = smart_clean(text)


# Simple chatbot response function
def chatbot_response(user_message, preprocessed_data=None):
    if preprocessed_data:
        return f"Received your message: '{user_message}'. I have processed your uploaded file. Here's a preview of the preprocessed data: {preprocessed_data[:100]}..."
    return f"Received your message: '{user_message}'. Please upload a file to proceed."

# Streamlit app
st.title("Chatbot")

# Get user_id from query parameters
query_params = st.query_params
st.write(f"Debug: Full query params: {query_params}")  # Debug log
user_id = query_params.get("user_id", "")
if isinstance(user_id, list):
    user_id = user_id[0] if user_id else ""
st.write(f"Debug: Extracted user_id: '{user_id}'")  # Debug log

# Validate user_id
if not user_id or not is_valid_uuid(user_id):
    st.error("Invalid or missing User ID. Please log in or register again.")
    st.markdown("[Go to Login/Register](http://localhost:5000)", unsafe_allow_html=True)
    st.stop()
else:
    st.write(f"Logged in as User ID: {user_id}")

    # Add logout button
    if st.button("Logout"):
        st.markdown("<meta http-equiv='refresh' content='0;url=http://localhost:5000'>", unsafe_allow_html=True)
        st.stop()

    # File upload section
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Upload a file (CSV, Excel, PDF, Text)", type=['csv', 'xlsx', 'xls', 'pdf', 'txt'])

    preprocessed_data = None
    if uploaded_file:
        file_id = str(uuid.uuid4())
        file_path = f"uploads/{uploaded_file.name}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
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
            storage_path = f"preprocessed/{uploaded_file.name}.txt"
            os.makedirs("preprocessed", exist_ok=True)
            with open(storage_path, "w", encoding="utf-8") as f:
                f.write(preprocessed_data)
            
            preprocess_time = datetime.utcnow().isoformat()

            # Step 1: Insert into file_uploads first (since it’s referenced by other tables)
            upload_data = {
                "file_id": file_id,
                "user_id": user_id,
                "file_name": uploaded_file.name,
                "file_path": file_path,
                "upload_time": preprocess_time
            }
            try:
                supabase.table("file_uploads").insert(upload_data).execute()
                st.write(f"Debug: Successfully inserted into file_uploads with file_id: {file_id}")
            except Exception as e:
                st.error(f"Error logging file upload: {e}")
                st.stop()

            # Step 2: Insert into data_preprocessing_logs (after file_uploads)
            preprocess_id = str(uuid.uuid4())
            file_data = {
                "preprocess_id": preprocess_id,
                "file_id": file_id,
                "user_id": user_id,
                "status": "completed",
                "preprocess_time": preprocess_time
            }
            try:
                supabase.table("data_preprocessing_logs").insert(file_data).execute()
                st.write(f"Debug: Successfully inserted into data_preprocessing_logs with preprocess_id: {preprocess_id}")
            except Exception as e:
                st.error(f"Error logging preprocessing data: {e}")
                st.stop()
            
            # Step 3: Insert into datasets (after file_uploads)
            dataset_id = str(uuid.uuid4())
            dataset_data = {
                "dataset_id": dataset_id,
                "file_id": file_id,
                "user_id": user_id,
                "dataset_name": uploaded_file.name,
                "storage_path": storage_path,
                "created_at": preprocess_time
            }
            try:
                supabase.table("datasets").insert(dataset_data).execute()
                st.write(f"Debug: Successfully inserted into datasets with dataset_id: {dataset_id}")
            except Exception as e:
                st.error(f"Error storing dataset: {e}")
                st.stop()
            
            st.success(f"File {uploaded_file.name} preprocessed and stored successfully!")
            st.write("Preprocessed data preview:")
            st.text(preprocessed_data[:500])

    # Chatbot section
    st.header("Chat with the Bot")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate bot response
        bot_response = chatbot_response(user_input, preprocessed_data)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        
        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(bot_response)