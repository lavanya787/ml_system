import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import spacy
import re
from textblob import TextBlob
import spacy
import pdfplumber
import difflib
from flask import Flask, request, jsonify
from flask import render_template, redirect, url_for
from supabase import create_client, Client
import bcrypt, uuid, re
from datetime import datetime


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.domain_detector import DomainDetector
from utils.data_processor import DataProcessor, GenericDomainConfig
from utils.model_handler import ModelHandler
from utils.llm_manager import LLMManager
from utils.query_handler import QueryHandler
from utils.logger import Logger
from utils.preprocessing import Preprocessor
from utils.file_upload import file_upload_interface

# Load spaCy model for optional entity extraction
nlp = spacy.load("en_core_web_sm")

DOMAINS = [
    'customer_support', 'entertainment', 'gaming', 'legal', 'marketing',
    'logistics', 'manufacturing', 'real_estate', 'agriculture', 'energy',
    'hospitality', 'automobile', 'telecommunications', 'government',
    'food_beverage', 'it_services', 'event_management', 'insurance',
    'retail', 'hr_resource'
]

st.set_page_config(page_title="Multi-Domain ML System", page_icon="🧠", layout="wide")

# Supabase credentials
SUPABASE_URL = "https://uctyxchurvievzvhthru.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVjdHl4Y2h1cnZpZXZ6dmh0aHJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU0ODk0MDIsImV4cCI6MjA2MTA2NTQwMn0.YytXX-q4QDO_vY9f1e_P-UWc6v6860kcsbe_bTZVgCI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class MultiDomainMLSystem:
    def __init__(self):
        self.logger = Logger()
        self.preprocessor = Preprocessor()
        self.model_handler = ModelHandler()
        self.detector = DomainDetector(logger=self.logger)
        self.processor = DataProcessor(logger=self.logger)
        self.llm_manager = LLMManager(logger=self.logger)
        self.model_handler = ModelHandler(logger=self.logger)
        self.query_handler = QueryHandler(logger=self.logger)
        self.domain_config = GenericDomainConfig(logger=self.logger)
        # Initialize session state variables
        st.session_state.setdefault('dataset_uploaded', False)
        st.session_state.setdefault('detected_domain', None)
        st.session_state.setdefault('processed_data', None)
        st.session_state.setdefault('raw_data', None)
        st.session_state.setdefault('models', {})  # store models per domain or general
        st.session_state.setdefault('user_id','')
        st.session_state.setdefault('current_session_queries', [])
        st.session_state.setdefault('suggested_input', '')  # For handling suggestions
        st.session_state.setdefault('logged_in', False)
        st.session_state.setdefault('username', None)
    def run(self):
        st.title("🧠 Multi-Domain ML System")
        st.markdown("### Automatically detects and analyzes datasets from multiple domains")

        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a section", [
            "Dataset Upload & Detection", "Model Training",
            "Data Analysis","Intelligent Query",
            "Predictions", "System Logs"
        ])

        if page == "Dataset Upload & Detection":
            self.upload_and_detect_page()
        elif page == "Model Training":
            self.model_training_page()
        elif page == "Data Analysis":
            self.data_analysis_page()
        elif page == "Intelligent Query":
            self.intelligent_query_page()
        elif page == "Query History":
            self.query_history_page()
        elif page == "Predictions":
            self.predictions_page()
        elif page == "System Logs":
            self.system_logs_page()
        elif page == "Logout":
            self.logout()  
            st.success("Logged out successfully.")
    def read_uploaded_file(self, uploaded_file):
        import io
        import pdfplumber
        import docx

        file_type = uploaded_file.name.split('.')[-1].lower()

        try:
            if file_type == "csv":
                return pd.read_csv(uploaded_file)
            elif file_type in ["xlsx", "xls"]:
                return pd.read_excel(uploaded_file)
            elif file_type == "txt":
                text = uploaded_file.read().decode("utf-8")
                return pd.DataFrame({"Text": text.splitlines()})
            elif file_type == "pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                return pd.DataFrame({"Text": [text]})
            elif file_type == "docx":
                doc = docx.Document(uploaded_file)
                text = "\n".join([p.text for p in doc.paragraphs])
                return pd.DataFrame({"Text": [text]})
            else:
                return None
        except Exception as e:
            self.logger.log_error(f"Error reading uploaded file: {str(e)}")
            return None
 
    def upload_and_detect_page(self):
        st.subheader("📤 Upload Dataset")
        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=["csv", "xlsx", "xls", "txt", "docx", "doc", "pdf"],
            accept_multiple_files=True)

        if uploaded_files:
            # Initialize session dict if not present
            if 'user_uploads' not in st.session_state:
                st.session_state['user_uploads'] = {}

            for uploaded_file in uploaded_files:
                st.markdown(f"#### 📁 Processing: `{uploaded_file.name}`")
                data = self.read_uploaded_file(uploaded_file)

                if data is None:
                    self.logger.log_error(f"Unsupported or unreadable file: {uploaded_file.name}")
                    st.error(f"❌ Could not process file: {uploaded_file.name}")
                    continue

                self.logger.log_info(f"Dataset loaded successfully! Shape: {data.shape}")
                st.success(f"✅ Dataset loaded! Shape: {data.shape}")

                # Auto-detect domain
                detected = self.detector.detect_domain(data)
                st.write("🧠 Domain detection result:", detected)
                
                detected_domain = detected.get('domain', None)
                confidence = detected.get('confidence', 0)

                if detected_domain:
                    st.info(f"Detected domain: **{detected_domain}** (confidence: {confidence:.2f})")
                else:
                    st.warning("Could not confidently detect a domain.")
                st.write("📊 Uploaded data columns:", list(data.columns))

                # Process domain-specific data if detected
                processed_data = self.processor.process_domain_data(data, detected_domain) if detected_domain else data

                # Save each file's data separately in session state
                st.session_state['user_uploads'][uploaded_file.name] = {
                    'raw_data': data,
                    'domain': detected_domain,
                    'processed_data': processed_data
                }

                # Show preview of this file's processed data
                st.dataframe(processed_data.head())

                # Run NLP entity extraction if text column found
                text_col = None
                for col in data.columns:
                    if 'desc' in col.lower() or 'text' in col.lower():
                        text_col = col
                        break

                if text_col:
                    st.write(f"🔍 Running NLP on column: `{text_col}`")
                    sample_texts = data[text_col].dropna().astype(str).head(5)
                    for text in sample_texts:
                        doc = nlp(text)
                        ents = [(ent.text, ent.label_) for ent in doc.ents]
                        st.write(f"**Text**: {text}")
                        st.write(f"**Entities**: {ents}")
                        st.markdown("---")

            # Optionally: Save last uploaded dataset to shortcut session vars
            last_file = uploaded_files[-1]
            last_data = st.session_state['user_uploads'][last_file.name]['raw_data']
            last_domain = st.session_state['user_uploads'][last_file.name]['domain']
            last_processed = st.session_state['user_uploads'][last_file.name]['processed_data']

            st.session_state['raw_data'] = last_data
            st.session_state['detected_domain'] = last_domain
            st.session_state['processed_data'] = last_processed
            st.session_state['dataset_uploaded'] = True  # mark dataset uploaded

    def data_analysis_page(self):
        st.header("📈 Data Analysis")
        
        if not st.session_state.get('dataset_uploaded', False):
            st.warning("⚠️ Please upload a dataset first in the 'Dataset Upload & Detection' section.")
            return

        data = st.session_state['processed_data']
        detected_domain = st.session_state.get('detected_domain')
        
        if detected_domain:
            st.info(f"📊 Analysis for **{detected_domain.replace('_', ' ').title()}** domain")

        # Basic statistics
        st.subheader("📋 Dataset Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types**")
            dtype_df = pd.DataFrame(data.dtypes).reset_index()
            dtype_df.columns = ['Column', 'Type']
            st.dataframe(dtype_df)
        
        with col2:
            st.write("**Missing Values**")
            missing_df = pd.DataFrame(data.isnull().sum()).reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if not missing_df.empty:
                st.dataframe(missing_df)
            else:
                st.write("No missing values found! ✅")

        # Statistical summary
        st.subheader("📊 Statistical Summary")
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.dataframe(numeric_data.describe())
        else:
            st.warning("No numeric columns found for statistical analysis.")

        # Visualizations
        st.subheader("📈 Visualizations")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X-axis", numeric_cols, key="x_axis")
            with col2:
                y_col = st.selectbox("Select Y-axis", numeric_cols, key="y_axis")
            
            if x_col != y_col:
                fig = px.scatter(data, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        if numeric_cols:
            st.subheader("📊 Distribution Analysis")
            selected_col = st.selectbox("Select column for distribution", numeric_cols, key="dist_col")
            fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        if not numeric_data.empty:
            st.subheader("🔗 Correlation Heatmap")
            corr = numeric_data.corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    def model_training_page(self):
        st.header("🛠️ Model Training")
        if st.session_state.get('processed_data') is None:
            st.warning("Please upload a dataset first in the 'Dataset Upload & Detection' section.")
            return

        domain = st.session_state.get('detected_domain')
        if not domain:
            st.warning("Please upload a dataset with a detectable domain first.")
            return

        data = st.session_state['processed_data']

        st.write(f"Training model for domain: **{domain}**")

        # For demonstration, trigger model training on button press
        if st.button("Train Model"):
            try:
                with st.spinner("Training model..."):
                    model = self.model_handler.train_model(data, domain)
                    # Save the model in session state
                    st.session_state['models'][domain] = model
                    st.success(f"Model trained successfully for domain '{domain}'!")
                    self.logger.log_info(f"Model trained for domain: {domain}")
            except Exception as e:
                st.error(f"Model training failed: {str(e)}")
                self.logger.log_error(f"Model training failed: {str(e)}")

    def intelligent_query_page(self):
        st.header("💡 Intelligent Query")
        data = st.session_state.get('raw_data')
        if data is None:
            st.warning("Upload and process a dataset first.")
            return

        if 'chat_log' not in st.session_state:
            st.session_state.chat_log = []

        if 'pending_query' in st.session_state:
            st.session_state['chat_input'] = st.session_state['pending_query']
            del st.session_state['pending_query']

        # === Semantic Inference ===
        def infer_column_type(col: str) -> str:
            col_l = col.lower()
            if "name" in col_l: return "name"
            if any(x in col_l for x in ["roll", "id", "code", "emp", "reg"]): return "id"
            if "gender" in col_l: return "gender"
            if "age" in col_l: return "age"
            if "percent" in col_l: return "percentage"
            if "attend" in col_l: return "attendance"
            if "extra" in col_l and "curric" in col_l: return "extracurricular"
            if any(x in col_l for x in ["mark", "score", "math", "phys", "chem", "bio", "comp", "eng"]): return "subject"
            return "other"

        def suggest_fields_from_query(question: str, all_columns: list, top_n: int = 5) -> list:
            question_words = re.findall(r'\w+', question.lower())
            scored = []

            for col in all_columns:
                col_clean = col.lower().replace("_", " ")
                score = sum(1 for word in question_words if word in col_clean)
                if score > 0:
                    scored.append((col, score))
                else:
                    # Fuzzy match fallback
                    for word in question_words:
                        if difflib.get_close_matches(word, col_clean.split(), cutoff=0.8):
                            scored.append((col, 1))
                            break

            scored = sorted(scored, key=lambda x: -x[1])
            return [col for col, _ in scored[:top_n]]

        # Match fields using fuzzy logic
        def get_relevant_fields(question: str, columns: list) -> list:
            question_words = re.findall(r'\w+', question.lower())
            relevant = []
            for col in columns:
                role = infer_column_type(col)
                col_clean = col.lower().replace("_", " ")
                col_words = col_clean.split()
                # Check if column name or its words are directly in question
                if any(word in question_words for word in col_words):
                    relevant.append(col)
                # Allow fuzzy match ONLY for subject/percentage (not age, gender, etc.)
                elif role in ["subject", "percentage"]:
                    if any(difflib.get_close_matches(w, question_words, cutoff=0.85) for w in col_words):
                        relevant.append(col)
            return relevant

        def format_summary(sid, row, fields, all_num_cols):
            name_col = next((col for col in row.index if infer_column_type(col) == "name"), None)
            name = row.get(name_col, "Unknown")
        # Build response parts for each requested field
            parts = []
            used_fields= set()
            # What fields were actually asked for
            explicitly_requested = set(fields)
            for col in fields:
                if col in used_fields or col not in row.index:
                    continue
                used_fields.add(col)
                val = row[col]
                role = infer_column_type(col)
                pretty = col.replace("_", " ").title()
                if pd.isna(val):
                    parts.append(f"has no data for {pretty}")
                elif role == "subject":
                    parts.append(f"scored {val} in {pretty}")
                elif role == "age":
                    parts.append(f"is {val} years old")
                elif role == "gender":
                    parts.append(f"is identified as {val}")
                elif role == "attendance":
                    parts.append(f"has an attendance of {val}%")
                elif role == "extracurricular":
                        if str(val).strip().lower() in ["yes", "1", "true"]:
                            parts.append("participated in extracurricular activities")
                        else:
                            parts.append("did not participate in extracurricular activities")
                elif "total" in col.lower() or "sum" in col.lower():
                    parts.append(f"has a total of **{val}** marks")
                else:
                    parts.append(f"has {pretty} as {val}")
            if not parts:
                return f"❌ **Error**: No valid data found for the requested fields for ID **{sid}**."
        
            return f"📊 **Student {sid}** ({name}): " + ", ".join(parts) + "."
        
        def classify(df):
            return {
                "text": df.select_dtypes(include='object').columns.tolist(),
                "num": df.select_dtypes(include='number').columns.tolist(),
                "cat": df.select_dtypes(include='category').columns.tolist()
            }

        # Show chat history
        for m in st.session_state.chat_log:
            icon = "👤" if m["role"] == "user" else "🤖"
            st.markdown(f"{icon} **{m['role'].capitalize()}:** {m['content']}")

        q = st.text_input("Ask your question here:", key="chat_input")
        if not q: return
        # Skip auto-correction for queries that look like they contain IDs or specific terms
        skip_correction = any(keyword in q.lower() for keyword in ['stu', 'emp', 'id', 'physics', 'chemistry', 'math', 'computer'])
    
        if not skip_correction:
            try:
                corrected = str(TextBlob(q).correct())
                if corrected.lower() != q.lower():
                    st.info(f"🔍 Did you mean: **{corrected}**")
                    q = corrected
            except: pass
        st.session_state.chat_log.append({"role": "user", "content": q})
        question = q.lower()
        resp, handled = "", False
        c = classify(data)
        # === Dynamic ID and Field Detection ===
        sid, sid_col = None, None
        question_words= re.findall(r'\w+', question.lower())
        for col in c["text"]:
            col_type = infer_column_type(col)
            if col_type in ["id", "name"]:  # Prioritize ID and name columns
                for val in data[col].dropna().astype(str):
                    val_lower = val.lower()
                    # Check for exact match or partial match in question
                    if val_lower in question or any(val_lower in word for word in question_words):
                        sid, sid_col = val, col
                        break
            if sid: break
                # If no ID found, try fuzzy matching
        if not sid:
            for col in c["text"]:
                for val in data[col].dropna().astype(str):
                    val_lower = val.lower()
                    matches = difflib.get_close_matches(val_lower, question_words, cutoff=0.6)
                    if matches:
                        sid, sid_col = val, col
                        break
                if sid:break

        # Only match numeric or subject-like fields that are relevant
        all_fields = c["num"] + c["text"]
        matched_fields = get_relevant_fields(question, all_fields)

        if sid:
            row = data[data[sid_col].astype(str).str.lower() == sid.lower()]
            if not row.empty:
            # fallback: if no matched_fields, show all subjects
                if not matched_fields:
                    matched_fields = [col for col in c["num"] if infer_column_type(col) == "subject"]
                resp = format_summary(sid, row.iloc[0], matched_fields, c["num"])
                handled = True

     # === Aggregations / Stats ===
        if not handled:
            for col in c["num"]:
                if col.lower() in question:
                    if "average" in question:
                        resp = f"Average {col}: {data[col].mean():.2f}"
                    elif "max" in question:
                        resp = f"Max {col}: {data[col].max()}"
                    elif "min" in question:
                        resp = f"Min {col}: {data[col].min()}"
                    elif any(k in question for k in ["top", "highest", "toppers", "rank"]):
                        matched_subjects = get_relevant_fields(question, c["num"])

                        if matched_subjects:
                            subject = matched_subjects[0]
                            top_rows = data.nlargest(5, subject)
        
                            name_col = next((col for col in data.columns if infer_column_type(col) == "name"), None)
                            id_col = next((col for col in data.columns if infer_column_type(col) == "id"), None)

                            ranks = ["🥇 1st", "🥈 2nd", "🥉 3rd", "4th", "5th"]
                            resp_lines = [f"📊 **Top 5 Performers in {subject}:**\n"]
        
                            for i, (_, row) in enumerate(top_rows.iterrows()):
                                name = row.get(name_col, "Unknown") if name_col else "Unknown"
                                roll = row.get(id_col, "N/A") if id_col else "N/A"
                                score = row[subject]
                                rank = ranks[i] if i < len(ranks) else f"{i+1}th"
                                resp_lines.append(f"{rank}: **{name}** (Roll No: {roll}) — scored {score} in {subject}")
        
                            resp = "\n".join(resp_lines)
                            handled = True

                    elif "below" in question:
                        val = int(''.join([c for c in question if c.isdigit()]))
                        filtered = data[data[col] < val]
                        st.dataframe(filtered)
                        resp = f"{len(filtered)} records below {val} in {col}"
                    handled = True
                    break

    # === Pass Count / Pass Percentage ===
        if not handled:
            if "how many" in question and "pass" in question:
                for col in c["num"]:
                    if col.lower() in question:
                        passed = data[data[col] >= 40]
                        resp = f"{len(passed)} students passed in {col}."
                        handled = True
                        break
            elif "pass percentage" in question:
                for col in c["num"]:
                    if col.lower() in question:
                        total = len(data)
                        passed = data[data[col] >= 40]
                        percent = (len(passed) / total) * 100
                        resp = f"Pass percentage in {col}: {percent:.2f}%"
                        handled = True
                        break
        if not handled:
                # Attempt fallback auto-question generation
            all_columns = data.columns.tolist()
            suggested_fields = suggest_fields_from_query(question, all_columns)
            sid_fallback, sid_col = None, None
            for col in data.select_dtypes(include='object').columns:
                for val in data[col].dropna().astype(str):
                    if val.lower() in question:
                        sid_fallback, sid_col = val, col
                        break
                if sid_fallback:
                    break

            if sid_fallback and suggested_fields:
                row = data[data[sid_col].astype(str).str.lower() == sid_fallback.lower()]
                if not row.empty:
                    st.info(f"🤖 Auto-completing your query using top matches: `{', '.join(suggested_fields)}`")
                    resp = format_summary(sid_fallback, row.iloc[0], suggested_fields)
                    handled = True
                else:
                    resp = "🤖 I found possible fields, but couldn't find a matching ID in the dataset."
            else:
                resp = "🤖 I'm not sure how to answer that. Try rephrasing or include a known ID/field."

        st.markdown(f"📋 {resp}")
        st.session_state.chat_log.append({"role": "assistant", "content": resp})

        def generate_suggestions(question: str, columns: list[str]) -> list[str]:
            q = question.lower()
            suggestions = []

            if "top" in q or "max" in q:
                suggestions.append("Who has the highest score?")
                suggestions.append("Top 5 performers in any subject")
            elif "average" in q:
                suggestions.append("Compare average of two subjects")
                suggestions.append("Show average attendance")
            elif "pass" in q:
                suggestions.append("How many failed in Maths")
                suggestions.append("Pass percentage in Chemistry")
            elif "compare" in q or "between" in q:
                suggestions.append("Compare Chemistry and Physics scores")
                suggestions.append("Which subject has higher average?")
            elif "attendance" in q:
                suggestions.append("Average of attendance")
                suggestions.append("Students with attendance below 75%")
            elif "score" in q or "marks" in q or "total" in q:
                suggestions.append("Show students with total marks below 300")
                suggestions.append("Top 3 students in total marks")
            else:
                suggestions += [
                "Top performers in any subject",
                "Show average attendance",
                "Pass percentage in Physics",
                "Who failed in Chemistry?"
            ]

            return suggestions[:5]
        domain_examples = generate_suggestions(question, c["num"] + c["text"])
        cols = st.columns(len(domain_examples))
        for i, e in enumerate(domain_examples):
            if cols[i].button(e, key=f"ex_{i}"):
                st.session_state['pending_query'] = e
                st.experimental_rerun()

    def predictions_page(self):
        st.header("📉 Predictions")
        
        if not st.session_state.get('models'):
            st.warning("⚠️ No trained models available. Please train a model first in the 'Model Training' section.")
            return

        data = st.session_state['processed_data']
        models = st.session_state['models']
        
        st.subheader("🔮 Generate Predictions")
        
        # Select model
        target_col = st.selectbox("Select Model (Target Column)", list(models.keys()), key="pred_target")
        
        if st.button("🎯 Generate Predictions", type="primary"):
            try:
                with st.spinner("Generating predictions..."):
                    model_info = models[target_col]
                    features = model_info['features']
                    
                    X = data[features].dropna()
                    predictions = model_info['model'].predict(X)
                    
                    # Create results dataframe
                    results_df = data.loc[X.index].copy()
                    results_df['Predictions'] = predictions
                    
                    st.success(f"✅ Generated {len(predictions)} predictions!")
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Sample Predictions**")
                        display_cols = [target_col, 'Predictions'] + features[:3]
                        st.dataframe(results_df[display_cols].head(10))
                    
                    with col2:
                        st.write("**Prediction Statistics**")
                        pred_stats = pd.DataFrame({
                            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                            'Value': [
                                np.mean(predictions),
                                np.median(predictions),
                                np.std(predictions),
                                np.min(predictions),
                                np.max(predictions)
                            ]
                        })
                        st.dataframe(pred_stats)
                    
                    # Visualization
                    st.subheader("📈 Prediction Distribution")
                    fig = px.histogram(x=predictions, title=f"Distribution of Predictions for {target_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Actual vs Predicted (if target column exists)
                    if target_col in results_df.columns:
                        fig2 = px.scatter(x=results_df[target_col], y=predictions, 
                                        title=f"Actual vs Predicted: {target_col}",
                                        labels={'x': f'Actual {target_col}', 'y': f'Predicted {target_col}'})
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    self.logger.log_info(f"Predictions generated for {target_col}")
                    
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                self.logger.log_error(f"Prediction error: {str(e)}")

    def system_logs_page(self):
        st.header("🧾 System Logs")
        
        st.info("📝 System logs and activity monitoring")
        
        # Session information
        st.subheader("📊 Session Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset Uploaded", "Yes" if st.session_state.get('dataset_uploaded') else "No")
        with col2:
            st.metric("Models Trained", len(st.session_state.get('models', {})))
        with col3:
            detected_domain = st.session_state.get('detected_domain', 'None')
            st.metric("Detected Domain", detected_domain.replace('_', ' ').title() if detected_domain else 'None')
        
        # Session state details
        if st.checkbox("Show detailed session state"):
            st.subheader("🔍 Session State Details")
            session_info = {
                'Dataset Uploaded': st.session_state.get('dataset_uploaded', False),
                'Detected Domain': st.session_state.get('detected_domain'),
                'Data Shape': st.session_state.get('processed_data').shape if st.session_state.get('processed_data') is not None else None,
                'Available Models': list(st.session_state.get('models', {}).keys()),
                'Session Keys': list(st.session_state.keys())
            }
            for key, value in session_info.items():
                st.write(f"**{key}:** {value}")
    def logout(self):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = ''
        st.success("Logged out successfully.")
        st.experimental_rerun()

if __name__ == "__main__":
    app = MultiDomainMLSystem()
    app.run()
