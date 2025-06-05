import streamlit as st
import pandas as pd
import os
import sys
import spacy
import io
import docx
import pdfplumber
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.domain_detector import DomainDetector
from utils.data_processor import DataProcessor
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
    
    def read_uploaded_file(self, uploaded_file):
        file_type = uploaded_file.name.split('.')[-1].lower()

        try:
            if file_type == "csv":
                return pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
            elif file_type in ["xlsx", "xls"]:
                return pd.read_excel(uploaded_file)
            elif file_type == "txt":
                content = uploaded_file.read().decode(errors="ignore")
                return pd.DataFrame({"text": content.splitlines()})
            elif file_type == "docx":
                doc = docx.Document(uploaded_file)
                fullText = "\n".join([para.text for para in doc.paragraphs])
                return pd.DataFrame({"text": fullText.splitlines()})
            elif file_type == "doc":
                st.error("❌ .doc format not supported. Please convert to .docx.")
                return None
            elif file_type == "pdf":
                with pdfplumber.open(uploaded_file) as pdf:
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
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls", "txt", "docx", "doc", "pdf"])

        if uploaded_file is not None:
            data = self.read_uploaded_file(uploaded_file)
            if data is None:
                self.logger.log_error("Unsupported or unreadable file.")
                return

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
