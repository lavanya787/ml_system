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
        self.query_handler = QueryHandler(logger=self.logger, llm_manager=self.llm_manager)

        # Initialize session state variables
        st.session_state.setdefault('dataset_uploaded', False)
        st.session_state.setdefault('detected_domain', None)
        st.session_state.setdefault('processed_data', None)
        st.session_state.setdefault('raw_data', None)
        st.session_state.setdefault('models', {})  # store models per domain or general

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
    
    def read_uploaded_file(self, uploaded_files):
        file_type = uploaded_files.name.split('.')[-1].lower()

        try:
            if file_type == "csv":
                return pd.read_csv(uploaded_files, encoding='utf-8', errors='ignore')
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
        "Upload one or more files",type=["csv", "xlsx", "xls", "txt", "docx", "doc", "pdf"],
         accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.markdown(f"#### 📁 Processing: `{uploaded_file.name}`")
                data = self.read_uploaded_file(uploaded_file)
                if data is None:
                    self.logger.log_error(f"Unsupported or unreadable file: {uploaded_file.name}")
                    continue
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
                    # Save each file's data separately in session state
            if 'user_uploads' not in st.session_state:
                st.session_state['user_uploads'] = {}
            st.session_state['user_uploads'][uploaded_file.name] = {
                'raw_data': data,
                'domain': detected_domain,
            'processed_data': self.processor.process_domain_data(data, detected_domain) if detected_domain else data
        }
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

        if st.session_state.get('processed_data') is None:
            st.warning("Please upload a dataset first in the 'Dataset Upload & Detection' section.")
            return
        
        if st.session_state.get('detected_domain') is None:
            st.warning("Domain not detected. Please upload an appropriate dataset.")
            return

        data = st.session_state['processed_data']
        raw_data = st.session_state['raw_data']
        domain = st.session_state['detected_domain']
        models = st.session_state.get('models', {})
        data = st.session_state['processed_data']
        st.markdown("Ask a question about your dataset (e.g., 'What is the average price?', 'Show top 5 rows')")
        question = st.text_input("🔍 Enter your question")

        if question:
            # First try the simple rule-based queries
            question_lower = question.lower()

            # Rule-based responses
            if any(k in question_lower for k in ["head", "top"]):
                num = 5
                for word in question_lower.split():
                    if word.isdigit():
                        num = int(word)
                        break
                st.write(f"Showing top {num} rows:")
                st.dataframe(data.head(num))

            elif any(k in question_lower for k in ["columns", "features"]):
                st.write("Available columns:")
                st.write(list(data.columns))

            elif any(k in question_lower for k in ["shape", "size"]):
                st.write(f"Shape of dataset: {data.shape[0]} rows × {data.shape[1]} columns")

            elif any(k in question_lower for k in ["describe", "summary"]):
                st.write("Dataset summary:")
                st.dataframe(data.describe())

            elif any(k in question_lower for k in ["null", "missing"]):
                st.write("Missing values per column:")
                st.dataframe(data.isnull().sum())
            
            elif any(k in question_lower for k in ["average", "mean"]):
                found = False
                for col in data.select_dtypes(include='number').columns:
                    if col.lower() in question_lower:
                        st.write(f"Mean of `{col}`: {data[col].mean():.2f}")
                        found = True
                        break
                if not found:
                    st.write("Specify a numeric column name to calculate the average.")

            elif "max" in question_lower:
                found = False
                for col in data.select_dtypes(include='number').columns:
                    if col.lower() in question_lower:
                        st.write(f"Max of `{col}`: {data[col].max():.2f}")
                        found = True
                        break
                if not found:
                    st.write("Specify a numeric column name to calculate the maximum.")

            elif "min" in question_lower:
                found = False
                for col in data.select_dtypes(include='number').columns:
                    if col.lower() in question_lower:
                        st.write(f"Min of `{col}`: {data[col].min():.2f}")
                        found = True
                        break
                if not found:
                    st.write("Specify a numeric column name to calculate the minimum.")

            else:
                # Use LLM + QueryHandler for more complex queries
                with st.spinner("Processing your query..."):
                    try:
                        # Assuming domain_config is an object with handler methods for the domain
                        # For this demo, let's create a dummy domain_config with methods we can call
                        domain_config = self.get_domain_config(domain)
                        response = self.query_handler.handle_query(
                            query=question,
                            domain_config=domain_config,
                            raw_data=raw_data,
                            processed_data=data,
                            models=models
                        )
                        if isinstance(response, dict) and 'error' in response:
                            st.error(response['error'])
                        else:
                            st.write("**Response:**")
                            if isinstance(response, pd.DataFrame):
                                st.dataframe(response)
                            else:
                                st.write(response)
                    except Exception as e:
                        st.error(f"Failed to process query: {e}")
                        self.logger.log_error(f"Query handling exception: {e}")
    def get_domain_config(self, domain_name):
        class DummyDomainConfig:
            def handle_prediction_query(self, query, raw_data, models):
                # Example: return model prediction output if model exists
                model = models.get(domain_name)
                if model:
                    # Dummy prediction logic
                    return f"Predicted result for query '{query}' using model of domain '{domain_name}'."
                else:
                    return f"No trained model found for domain '{domain_name}'."

            def handle_performance_query(self, query, processed_data):
                # Return summary stats as string
                return processed_data.describe()

            def handle_risk_query(self, query, raw_data, models):
                # Dummy risk analysis response
                return "Risk analysis feature not implemented yet."

            def handle_general_query(self, query, raw_data, processed_data):
                # Basic fallback: return first 5 rows containing keywords from query if possible
                keywords = [w.lower() for w in query.split()]
                filtered = processed_data[processed_data.apply(
                    lambda row: any(str(v).lower() in keywords for v in row.values), axis=1)]
                if not filtered.empty:
                    return filtered.head(5)
                return "Sorry, no relevant data found for your query."

        return DummyDomainConfig()

    def predictions_page(self):
        st.header("🔮 Predictions")

        domain = st.session_state.get('detected_domain')
        if domain is None:
            st.warning("Please upload and detect a domain first.")
            return

        models = st.session_state.get('models', {})
        model = models.get(domain)

        if model is None:
            st.warning(f"No trained model available for domain '{domain}'. Train a model first.")
            return

        st.write(f"Make predictions using the model for domain: **{domain}**")

        # Example: provide input UI depending on domain features (dummy example)
        # For demo, ask user to enter comma separated values for prediction features
        input_str = st.text_input("Enter feature values comma separated (e.g. 1.0, 2.5, 3.0):")

        if st.button("Predict") and input_str:
            try:
                features = [float(x.strip()) for x in input_str.split(',')]
                prediction = self.model_handler.predict(model, features)
                st.success(f"Prediction result: {prediction}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    def system_logs_page(self):
        st.header("📋 System Logs")
        logs = self.logger.read_logs()
        st.text_area("Logs", value=logs, height=400)


if __name__ == "__main__":
    app = MultiDomainMLSystem()
    app.run()
