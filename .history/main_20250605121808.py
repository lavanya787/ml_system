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

        if 'raw_data' not in st.session_state or st.session_state['raw_data'] is None:
            st.warning("Please upload and process a dataset first.")
            return

        data = st.session_state['raw_data']

        if 'chat_log' not in st.session_state:
            st.session_state.chat_log = []

        for msg in st.session_state.chat_log:
            icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"{icon} **{msg['role'].capitalize()}:** {msg['content']}")

        query = st.text_input("Ask your question here:", key="chat_input")
        if not query:
            return

        # ✅ TextBlob Auto-Correction
        try:
            corrected = str(TextBlob(query).correct())
            if corrected.lower() != query.lower():
                st.info(f"🔍 Did you mean: **{corrected}**")
                query = corrected
        except Exception as e:
            self.logger.log_info(f"TextBlob correction failed: {e}")

    st.session_state.chat_log.append({"role": "user", "content": query})
    question = query.lower()
    response = ""
    handled = False

    try:
        text_cols = data.select_dtypes(include='object').columns.tolist()
        num_cols = data.select_dtypes(include='number').columns.tolist()
        all_cols = data.columns.tolist()

        # === 1. Basic Info Retrieval ===
        entity_id = None
        subject_fields = []

        for col in text_cols:
            for val in data[col].dropna().unique():
                if str(val).lower() in question:
                    entity_id = str(val)
                    entity_col = col
                    break
            if entity_id:
                break

        for col in num_cols:
            if col.lower() in question:
                subject_fields.append(col)

        if entity_id and subject_fields:
            filtered = data[data[entity_col].astype(str).str.lower() == entity_id.lower()]
            if not filtered.empty:
                row = filtered.iloc[0]
                response = format_student_summary(entity_id, row, subject_fields)
                handled = True

        # === 2. Top/Bottom Performers ===
        if not handled and any(k in question for k in ["top", "highest", "best", "lowest", "bottom"]):
            n = 5
            subject = next((col for col in num_cols if col.lower() in question), None)
            if subject:
                if "top" in question or "highest" in question or "best" in question:
                    top_students = data.nlargest(n, subject)
                    response = f"🏆 Top {n} students in {subject}:\n"
                else:
                    top_students = data.nsmallest(n, subject)
                    response = f"🔻 Bottom {n} students in {subject}:\n"

                name_col = next((col for col in text_cols if "name" in col.lower()), None)
                for i, row in top_students.iterrows():
                    name = row.get(name_col, "Unknown")
                    response += f"- {name} ({subject}: {row[subject]})\n"
                handled = True

        # === 3. Statistical/Analytical Questions ===
        if not handled and any(k in question for k in ["average", "mean", "max", "min"]):
            op_map = {"average": np.mean, "mean": np.mean, "max": np.max, "min": np.min}
            op_type = next((k for k in op_map if k in question), None)
            col = next((c for c in num_cols if c.lower() in question), None)
            if op_type and col:
                result = op_map[op_type](data[col])
                response = f"📊 The {op_type} value in {col} is {result:.2f}."
                handled = True

        # === 4. Comparative Questions ===
        if not handled and "more" in question or "higher" in question:
            ids = [val for val in data[text_cols[0]].dropna().astype(str) if val.lower() in question]
            sub = next((col for col in num_cols if col.lower() in question), None)
            if len(ids) == 2 and sub:
                rows = data[data[text_cols[0]].astype(str).str.lower().isin([i.lower() for i in ids])]
                if len(rows) == 2:
                    values = rows[[text_cols[0], sub]].values.tolist()
                    winner = max(values, key=lambda x: x[1])
                    response = f"📌 {winner[0]} scored higher in {sub} with {winner[1]} marks."
                    handled = True

        # === 5. Categorical Analysis ===
        if not handled and ("count" in question or "how many" in question):
            for col in text_cols:
                if col.lower() in question:
                    counts = data[col].value_counts()
                    response = f"📊 Category distribution in {col}:\n"
                    for idx, val in counts.items():
                        response += f"- {idx}: {val}\n"
                    handled = True
                    break

        # === 6. Custom Filtering ===
        if not handled and ("above" in question or "below" in question):
            col = next((c for c in num_cols if c.lower() in question), None)
            operator = ">" if "above" in question else "<"
            value = int("".join([s for s in question if s.isdigit()]))
            if col:
                filtered = data.query(f"{col} {operator} {value}")
                response = f"🔍 {len(filtered)} students found where {col} {operator} {value}."
                st.dataframe(filtered)
                handled = True

        # === 7. Data Science-Oriented ===
        if not handled and "correlation" in question:
            cols = [col for col in num_cols if col.lower() in question]
            if len(cols) >= 2:
                corr = data[cols[0]].corr(data[cols[1]])
                response = f"📈 Correlation between {cols[0]} and {cols[1]} is {corr:.2f}."
                handled = True

        # === 8. Visualization-Driven ===
        if not handled and ("chart" in question or "plot" in question):
            x = next((col for col in all_cols if "x axis" in question or col.lower() in question), None)
            y = next((col for col in num_cols if col.lower() in question), None)
            if x and y:
                fig = px.bar(data, x=x, y=y, title=f"{y} vs {x}")
                st.plotly_chart(fig)
                response = f"📊 Bar chart for {y} vs {x} shown above."
                handled = True

        # === 9. Behavioral / Trend Analysis ===
        if not handled and ("trend" in question or "improve" in question):
            id_col = text_cols[0]
            subject = next((col for col in num_cols if col.lower() in question), None)
            if id_col and subject:
                trend_df = data[[id_col, subject]].sort_values(by=[id_col])
                fig = px.line(trend_df, x=id_col, y=subject, title=f"{subject} Trend by Student")
                st.plotly_chart(fig)
                response = f"📈 Trend of {subject} marks is shown above."
                handled = True

        # === 10. Contextual & Role-Based ===
        if not handled and ("as a teacher" in question or "suggest" in question):
            low_score_cols = [col for col in num_cols if col.lower() in question]
            if low_score_cols:
                low = data[data[low_score_cols[0]] < 40]
                response = f"As a teacher, {len(low)} students may need support in {low_score_cols[0]}."
                st.dataframe(low)
                handled = True

        # === 11. Derived Field / Transformation ===
        if not handled and ("pass" in question or "fail" in question):
            target_col = next((col for col in num_cols if col.lower() in question), None)
            if target_col:
                data['status'] = np.where(data[target_col] >= 40, "Pass", "Fail")
                summary = data['status'].value_counts().to_dict()
                response = f"✅ Pass/Fail Summary for {target_col}: {summary}"
                st.dataframe(data[[target_col, 'status']])
                handled = True

        # === Fallback ===
        if not handled:
            response = "🤖 I'm not sure how to answer that yet. Try rephrasing or use specific subject/ID."

        st.markdown(f"📋 {response}")
        st.session_state.chat_log.append({"role": "assistant", "content": response})

    except Exception as e:
        error = f"❌ Error: {e}"
        st.error(error)
        st.session_state.chat_log.append({"role": "assistant", "content": error})

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