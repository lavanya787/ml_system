import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import spacy
import re
from textblob import TextBlob
from datetime import datetime
import docx
import pdfplumber
from typing import Optional
from datetime import datetime
import difflib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.domain_detector import DomainDetector
from utils.data_processor import DataProcessor, GenericDomainConfig
from utils.model_handler import ModelHandler
from utils.llm_manager import LLMManager
from utils.query_handler import QueryHandler
from utils.logger import Logger

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


class MultiDomainMLSystem:
    def __init__(self):
        self.logger = Logger()
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
    
    def run(self):
        st.title("🧠 Multi-Domain ML System")
        st.markdown("### Automatically detects and analyzes datasets from multiple domains")
        # Manual User ID Input
        self.handle_user_id_input()
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a section", [
            "Dataset Upload & Detection", "Model Training",
            "Data Analysis","Intelligent Query",
            "Predictions", "System Logs"
        ])

        # Domain selection removed; always auto-detect
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

    def handle_user_id_input(self):
        """Handle manual user ID input"""
        st.sidebar.title("User Configuration")
        user_id = st.sidebar.text_input("👤 Enter User ID:", value=st.session_state.get('user_id', ''))
        
        if user_id:
            st.session_state['user_id'] = user_id
            st.sidebar.success(f"✅ User ID set: {user_id}")
        else:
            st.sidebar.warning("⚠️ Please enter a User ID to continue")

    def read_uploaded_file(self, uploaded_files):
        file_type = uploaded_files.name.split('.')[-1].lower()

        try:
            if file_type == "csv":
                return pd.read_csv(uploaded_files, encoding='utf-8')
            elif file_type in ["xlsx", "xls"]:
                return pd.read_excel(uploaded_files)
            elif file_type == "txt":
                content = uploaded_files.read().decode(errors="ignore")
                return pd.DataFrame({"text": content.splitlines()})
            elif file_type == "docx":
                doc = docx.Document(uploaded_files)
                fullText = "\n".join([para.text for para in doc.paragraphs])
                return pd.DataFrame({"text": fullText.splitlines()})
            elif file_type == "doc":
                st.error("❌ .doc format not supported. Please convert to .docx.")
                return None
            elif file_type == "pdf":
                with pdfplumber.open(uploaded_files) as pdf:
                    text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                return pd.DataFrame({"text": text.splitlines()})
            else:
                st.error("❌ Unsupported file type.")
                return None
        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")
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

        def classify(df):
            return {
                "text": df.select_dtypes(include='object').columns.tolist(),
                "num": df.select_dtypes(include='number').columns.tolist(),
                "cat": [c for c in df.select_dtypes(include='object') if df[c].nunique() < 15],
            }

        def format_summary(sid, row, fields):
            name_col = next((c for c in row.index if 'name' in c.lower()), None)
            name = row.get(name_col, "Unknown")
            parts = [f"{'secured an overall percentage of' if 'percentage' in f.lower() else f'scored {row[f]} marks in {f.title()}'}" for f in fields]
            return f"The student with Roll No {sid.upper()}, named {name}, " + ", and ".join(parts) + "."

        for m in st.session_state.chat_log:
            icon = "👤" if m["role"] == "user" else "🤖"
            st.markdown(f"{icon} **{m['role'].capitalize()}:** {m['content']}")

    q = st.text_input("Ask your question here:", key="chat_input")
    if not q: return

    try:
        corrected = str(TextBlob(q).correct())
        if corrected.lower() != q.lower(): st.info(f"🔍 Did you mean: **{corrected}**"); q = corrected
    except: pass

    st.session_state.chat_log.append({"role": "user", "content": q})
    question = q.lower()
    c = classify(data)
    resp, handled = "", False
    domain = st.session_state.get("detected_domain", "generic")

    # === 1. Basic Info Retrieval ===
    sid, sid_col = None, None
    for col in c["text"]:
        for val in data[col].dropna().astype(str):
            if val.lower() in question: sid, sid_col = val, col; break
        if sid: break
    fields = [col for col in c["num"] if col.lower() in question]
    if sid and fields:
        row = data[data[sid_col].astype(str).str.lower() == sid.lower()]
        if not row.empty:
            resp, handled = format_summary(sid, row.iloc[0], fields), True

    # === 2. Stats / Top / Filter / Comparison ===
    if not handled:
        num_col = next((col for col in c["num"] if col.lower() in question), None)
        if "average" in question and num_col: resp = f"Average {num_col}: {data[num_col].mean():.2f}"; handled = True
        elif "max" in question and num_col: resp = f"Max {num_col}: {data[num_col].max()}"; handled = True
        elif "min" in question and num_col: resp = f"Min {num_col}: {data[num_col].min()}"; handled = True
        elif "top" in question and num_col:
            top = data.nlargest(5, num_col); resp = f"Top 5 in {num_col}:\n" + "\n".join(f"- {r}" for r in top[num_col].tolist()); handled = True
        elif "below" in question and num_col:
            val = int(''.join([c for c in question if c.isdigit()])); filtered = data[data[num_col] < val]
            st.dataframe(filtered); resp = f"{len(filtered)} records found with {num_col} below {val}."; handled = True

    # === 3. Derived / NLP / Fallback ===
    if not handled and "pass" in question:
        col = next((c for c in c["num"] if c.lower() in question), None)
        if col: data['status'] = np.where(data[col] >= 40, "Pass", "Fail"); st.dataframe(data[[col, 'status']])
        resp, handled = f"Pass/Fail status based on {col}.", True

    if not handled: resp = "🤖 I'm not sure how to answer that. Try rephrasing or use IDs/subjects."

    st.markdown(f"📋 {resp}")
    st.session_state.chat_log.append({"role": "assistant", "content": resp})

    # === Suggestions ===
    domain_examples = {
        "education": ["Top students in Physics", "Average in Computer", "Pass/fail in Chemistry"],
        "retail": ["Sales above 10000", "Top products", "Revenue by category"],
        "generic": ["What is the max value?", "Show a bar chart"]
    }
    ex = domain_examples.get(domain, domain_examples["generic"])
    cols = st.columns(len(ex))
    for i, e in enumerate(ex):
        if cols[i].button(e, key=f"ex_{i}"): st.session_state['chat_input'] = e; st.rerun()

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

if __name__ == "__main__":
    app = MultiDomainMLSystem()
    app.run()