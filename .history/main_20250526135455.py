import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.domain_detector import DomainDetector
from utils.data_processor import DataProcessor
from utils.model_handler import ModelHandler
from utils.query_handler import QueryHandler
from utils.visualizer import Visualizer
from utils.logger import Logger

# Available domains (for manual override)
DOMAINS = ['medical', 'healthcare', 'transport', 'devices', 'education']

# Streamlit page config
st.set_page_config(page_title="Multi-Domain ML System", page_icon="🧠", layout="wide")

class MultiDomainMLSystem:
    def __init__(self):
        self.logger = Logger()  # Initialize logger first
        self.detector = DomainDetector(logger=self.logger)
        self.processor = DataProcessor()
        self.model_manager = ModelHandler()
        self.query_handler = QueryHandler()
        self.visualizer = Visualizer()

        # Initialize session state
        st.session_state.setdefault('dataset_uploaded', False)
        st.session_state.setdefault('detected_domain', None)
        st.session_state.setdefault('processed_data', None)
        st.session_state.setdefault('trained_models', None)

    def run(self):
        st.title("🧠 Multi-Domain ML System")
        st.markdown("### Automatically detects and analyzes datasets from multiple domains")

        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a section", [
            "Dataset Upload & Detection",
            "Data Analysis",
            "Model Training",
            "Intelligent Query",
            "Predictions",
            "System Logs"
        ])

        st.sidebar.subheader("Domain Selection")
        selected_domain = st.sidebar.selectbox("Select domain (or Auto-Detect)", ['Auto-Detect'] + DOMAINS)

        # Routing
        if page == "Dataset Upload & Detection":
            self.upload_and_detect_page(selected_domain)
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

    def upload_and_detect_page(self, selected_domain: str):
        st.header("📤 Dataset Upload & Domain Detection")

        uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, or JSON)", type=['csv', 'xlsx', 'xls', 'json'])

        if uploaded_file:
            try:
                # Load dataset
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    return

                st.session_state.raw_data = df
                st.session_state.dataset_uploaded = True

                # Log upload
                self.logger.log_query(
                    query_type="dataset_upload",
                    query=f"Dataset uploaded: {uploaded_file.name}",
                    dataset_info=f"Shape: {df.shape}, Columns: {list(df.columns)}"
                )

                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

                # Domain Detection
                with st.spinner("🔍 Detecting domain..."):
                    if selected_domain == 'Auto-Detect':
                        detection_result = self.detector.detect_domain(df)
                        st.session_state.detected_domain = detection_result['domain']
                    else:
                        # Manual override
                        detection_result = {
                            'domain': selected_domain,
                            'confidence': 1.0,
                            'config': None,
                            'features': list(df.columns)
                        }
                        st.session_state.detected_domain = selected_domain

                st.success(f"🧠 Detected Domain: **{st.session_state.detected_domain}** "
                           f"(Confidence: {detection_result['confidence']:.1%})")

                # Process data
                with st.spinner(f"🔧 Processing {st.session_state.detected_domain} dataset..."):
                    processed_data = self.processor.process_domain_data(
                        df, detection_result, st.session_state.detected_domain)
                    st.session_state.processed_data = processed_data

                st.success("✅ Data processed and ready for analysis!")

                st.subheader("📊 Dataset Preview")
                st.dataframe(df.head(10))

            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
                self.logger.log_query(query_type="error", query="Dataset upload failed", response=str(e))

    def data_analysis_page(self):
        st.header("📈 Data Analysis")
        if not st.session_state.dataset_uploaded:
            st.warning("Please upload a dataset first.")
            return
        df = st.session_state.raw_data
        st.write("Summary Statistics:")
        st.dataframe(df.describe())

    def model_training_page(self):
        st.header("🤖 Model Training")
        st.info("This section is under development.")

    def intelligent_query_page(self):
        st.header("🧠 Intelligent Query")
        st.info("This section is under development.")

    def predictions_page(self):
        st.header("📍 Predictions")
        st.info("This section is under development.")

    def system_logs_page(self):
        st.header("📋 System Logs")
        self.logger.display_logs()

# Run app
if __name__ == "__main__":
    system = MultiDomainMLSystem()
    system.run()
