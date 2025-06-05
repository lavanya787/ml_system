import streamlit as st
import pandas as pd
import os
import sys
import spacy
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from utils.domain_detector import DomainDetector
from utils.data_processor import DataProcessor
from utils.logger import Logger
from typing import Dict, Any
import io
import logging
from datetime import datetime

# Initialize logger
logger = SystemLogger()

# Set up logging for Streamlit display
log_buffer = io.StringIO()
handler = logging.StreamHandler(log_buffer)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

st.set_page_config(page_title="Enhanced ML System", layout="wide")

# Initialize DomainDetector
@st.cache_resource
def get_detector():
    return DomainDetector(logger=logger)

detector = get_detector()

st.title("📊 Enhanced ML System")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select Feature", ["Data Analysis", "Model Training", "Predictions", "System Logs"])

# File uploader
uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv", key="file_uploader")
data = None
detection_result = None
models = st.session_state.get('models', {})

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        detection_result = detector.detect_domain(data)
        st.session_state['detection_result'] = detection_result
        st.session_state['data'] = data
        logger.log_info(f"Dataset uploaded: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        logger.log_error(f"Error loading dataset: {str(e)}")

# Data Analysis
if page == "Data Analysis" and data is not None:
    st.header("📈 Data Analysis")
    
    # Domain detection results
    st.subheader("Domain Detection")
    if detection_result['domain']:
        st.write(f"**Detected Domain**: {detection_result['domain'].replace('_', ' ').title()}")
        st.write(f"**Confidence**: {detection_result['confidence']:.2f}")
        st.write(f"**Detected Features**: {detection_result['features']}")
    else:
        st.warning("No domain detected with sufficient confidence. Using generic analysis.")
    
    # Dataset summary
    st.subheader("Dataset Summary")
    st.write("**Columns**: ", data.columns.tolist())
    st.write("**Data Types**:")
    st.dataframe(data.dtypes.to_frame().T)
    st.write("**Sample Data**:")
    st.dataframe(data.head())

    # Visualizations
    st.subheader("Exploratory Visualizations")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        st.warning("No numeric columns for visualization.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="x_axis")
        with col2:
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="y_axis")
        
        if x_col and y_col:
            fig = px.scatter(data, x=x_col, y=y_col, title=f"{y_col} vs. {x_col}")
            st.plotly_chart(fig)
        
        # Histogram
        hist_col = st.selectbox("Select column for histogram", numeric_cols, key="hist_col")
        fig = px.histogram(data, x=hist_col, title=f"Distribution of {hist_col}")
        st.plotly_chart(fig)

    # Domain-specific analysis
    if detection_result['domain'] and detection_result['config']:
        st.subheader("Domain-Specific Analysis")
        try:
            detection_result['config'].create_analysis(data, {})
        except Exception as e:
            st.error(f"Error in domain-specific analysis: {str(e)}")
            logger.log_error(f"Domain analysis error: {str(e)}")

# Model Training
elif page == "Model Training" and data is not None:
    st.header("🛠️ Model Training")
    
    if detection_result['domain']:
        st.write(f"Training for domain: {detection_result['domain'].replace('_', ' ').title()}")
    else:
        st.warning("No domain detected. Training on generic features.")

    target_col = st.selectbox("Select Target Column", data.columns, key="target_col")
    model_type = st.selectbox("Model Type", ["Regression", "Classification"], key="model_type")
    
    if st.button("Train Model"):
        try:
            features = [col for col in data.columns if col != target_col and data[col].dtype in ['int64', 'float64']]
            if not features:
                st.error("No valid numeric features for training.")
                logger.log_error("No valid numeric features for training")
            else:
                X = data[features]
                y = data[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if model_type == "Regression":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    models[target_col] = {'model': model, 'model_type': 'regression', 'r2': score}
                    st.success(f"Model trained! R² Score: {score:.2f}")
                    logger.log_info(f"Regression model trained for {target_col}, R²: {score:.2f}")
                
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    models[target_col] = {'model': model, 'model_type': 'classification', 'accuracy': score}
                    st.success(f"Model trained! Accuracy: {score:.2f}")
                    logger.log_info(f"Classification model trained for {target_col}, Accuracy: {score:.2f}")
                
                st.session_state['models'] = models
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            logger.log_error(f"Model training error: {str(e)}")

# Predictions
elif page == "Predictions" and data is not None:
    st.header("📉 Predictions")
    
    if not models:
        st.warning("No trained models available. Please train a model first.")
    else:
        st.subheader("Generate Predictions")
        target_col = st.selectbox("Select Target Column", list(models.keys()), key="pred_target")
        
        if st.button("Predict"):
            try:
                model_info = models[target_col]
                features = [col for col in data.columns if col != target_col and data[col].dtype in ['int64', 'float64']]
                X = data[features]
                
                predictions = model_info['model'].predict(X)
                data['Predictions'] = predictions
                
                st.write("**Prediction Results**:")
                st.dataframe(data[[target_col, 'Predictions']].head())
                
                fig = px.histogram(data, x='Predictions', title=f"Prediction Distribution for {target_col}")
                st.plotly_chart(fig)
                
                logger.log_info(f"Predictions generated for {target_col}")
                
                # Domain-specific prediction query
                if detection_result['domain'] and detection_result['config']:
                    query = f"Predict {target_col}"
                    result = detection_result['config'].handle_prediction_query(query, data, models)
                    st.write("**Domain-Specific Prediction Summary**:")
                    st.write(result.get('summary', 'No summary available'))
                    if 'visualization' in result:
                        st.plotly_chart(result['visualization'])
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                logger.log_error(f"Prediction error: {str(e)}")

# System Logs
elif page == "System Logs":
    st.header("🧾 System Logs")
    
    log_buffer.seek(0)
    logs = log_buffer.getvalue()
    
    if logs:
        log_lines = logs.strip().split('\n')
        log_level = st.selectbox("Filter by Level", ["All", "INFO", "WARNING", "ERROR"], key="log_filter")
        
        filtered_logs = log_lines
        if log_level != "All":
            filtered_logs = [line for line in log_lines if log_level in line]
        
        st.text_area("Logs", value="\n".join(filtered_logs), height=400)
    else:
        st.info("No logs available.")

# Display domain detection status
if data is not None and detection_result:
    st.sidebar.subheader("Domain Detection")
    st.sidebar.write(f"Domain: {detection_result['domain'] or 'None'}")
    st.sidebar.write(f"Confidence: {detection_result['confidence']:.2f}")
    
class MultiDomainMLSystem:
    def __init__(self):
        self.logger = Logger()
        self.detector = DomainDetector(logger=self.logger)
        self.processor = DataProcessor(logger=self.logger)
        # Initialize session state variables
        st.session_state.setdefault('dataset_uploaded', False)
        st.session_state.setdefault('detected_domain', None)
        st.session_state.setdefault('processed_data', None)
        st.session_state.setdefault('raw_data', None)

    def run(self):
        st.title("🧠 Multi-Domain ML System")
        st.markdown("### Automatically detects and analyzes datasets from multiple domains")

        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a section", [
            "Dataset Upload & Detection", "Data Analysis",
            "Model Training", "Intelligent Query",
            "Predictions", "System Logs"
        ])

        # Domain selection removed; always auto-detect
        if page == "Dataset Upload & Detection":
            self.upload_and_detect_page()
        elif page == "Data Analysis":
            self.data_analysis_page()
        elif page == "Model Training":
            self.model_training_page()
        elif page == "Intelligent Query":
            self.intelligent_query_page()
        elif page == "Predictions":
            self.predictions_page()
        elif page == "System Logs":
            self.system_logs_page()

    def upload_and_detect_page(self):
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls", "txt", "docx", "doc", "pdf"])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                self.logger.log_info(f"Dataset loaded successfully! Shape: {data.shape}")
                st.success(f"✅ Dataset loaded! Shape: {data.shape}")

                # Auto-detect domain
                detected = self.detector.detect_domain(data)
                detected_domain = detected['domain']
                confidence = detected['confidence']

                if detected_domain:
                    st.info(f"Detected domain: **{detected_domain}** (confidence: {confidence:.2f})")
                else:
                    st.warning("Could not confidently detect a domain.")

                # Apply spaCy NLP if there's a 'description' or 'text' column
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

                # Save in session state
                st.session_state['raw_data'] = data
                st.session_state['detected_domain'] = detected_domain
                st.session_state['processed_data'] = self.processor.process_domain_data(data, detected_domain) if detected_domain else data

                st.dataframe(st.session_state['processed_data'].head())

            except Exception as e:
                self.logger.log_error(f"❌ Dataset upload failed: {str(e)}")
                st.error(f"❌ Error: {e}")

    def data_analysis_page(self):
        st.header("📈 Data Analysis")
        st.info("Feature under development.")

    def model_training_page(self):
        st.header("🛠️ Model Training")
        st.info("Feature under development.")

    def intelligent_query_page(self):
        st.header("💡 Intelligent Query")

        if st.session_state.get('processed_data') is None:
            st.warning("Please upload a dataset first in the 'Dataset Upload & Detection' section.")
            return

        data = st.session_state['processed_data']
        st.markdown("Ask a question about your dataset (e.g., 'What is the average price?', 'Show top 5 rows')")

        question = st.text_input("🔍 Enter your question")

        if question:
            question_lower = question.lower()

            # Rule-based responses
            if "head" in question_lower or "top" in question_lower:
                num = 5
                for word in question_lower.split():
                    if word.isdigit():
                        num = int(word)
                        break
                st.write(f"Showing top {num} rows:")
                st.dataframe(data.head(num))

            elif "columns" in question_lower or "features" in question_lower:
                st.write("Available columns:")
                st.write(list(data.columns))

            elif "shape" in question_lower or "size" in question_lower:
                st.write(f"Shape of dataset: {data.shape[0]} rows × {data.shape[1]} columns")

            elif "describe" in question_lower or "summary" in question_lower:
                st.write("Dataset summary:")
                st.dataframe(data.describe())

            elif "null" in question_lower or "missing" in question_lower:
                st.write("Missing values per column:")
                st.dataframe(data.isnull().sum())

            elif "average" in question_lower or "mean" in question_lower:
                for col in data.select_dtypes(include='number').columns:
                    if col.lower() in question_lower:
                        st.write(f"Mean of `{col}`: {data[col].mean():.2f}")
                        break
                else:
                    st.write("Specify a numeric column name to calculate the average.")

            elif "max" in question_lower:
                for col in data.select_dtypes(include='number').columns:
                    if col.lower() in question_lower:
                        st.write(f"Max of `{col}`: {data[col].max():.2f}")
                        break
                else:
                    st.write("Specify a numeric column name to calculate the maximum.")

            elif "min" in question_lower:
                for col in data.select_dtypes(include='number').columns:
                    if col.lower() in question_lower:
                        st.write(f"Min of `{col}`: {data[col].min():.2f}")
                        break
                else:
                    st.write("Specify a numeric column name to calculate the minimum.")

            else:
                st.warning("❓ Sorry, I couldn't understand your question. Try asking about top rows, average, nulls, etc.")

    def predictions_page(self):
        st.header("📉 Predictions")
        st.info("Feature under development.")

    def system_logs_page(self):
        st.header("🧾 System Logs")
        st.info("Logs will appear here in future updates.")


if __name__ == "__main__":
    app = MultiDomainMLSystem()
    app.run()
