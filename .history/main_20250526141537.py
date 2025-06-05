import streamlit as st
import pandas as pd
import os
import sys
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.domain_detector import DomainDetector
from utils.data_processor import DataProcessor
from utils.model_handler import ModelHandler
from utils.query_handler import QueryHandler
from utils.visualizer import Visualizer
from utils.logger import Logger

# Available domains for manual override
DOMAINS = [
    'customer_support', 'entertainment', 'gaming', 'legal', 'marketing',
    'logistics', 'manufacturing', 'real_estate', 'agriculture', 'energy',
    'hospitality', 'automobile', 'telecommunications', 'government',
    'food_beverage', 'it_services', 'event_management', 'insurance',
    'retail', 'hr_resources', 'banking'
]

# Set page config
st.set_page_config(page_title="Multi-Domain ML System", page_icon="🧠", layout="wide")


class MultiDomainMLSystem:
    def __init__(self):
        self.logger = Logger()
        self.detector = DomainDetector(logger=self.logger)
        self.processor = DataProcessor(logger=self.logger)
        self.model_manager = ModelHandler()
        self.query_handler = QueryHandler(logger=self.logger, llm_manager=None)
        self.visualizer = Visualizer(logger=self.logger)

        # Initialize session state variables
        st.session_state.setdefault('dataset_uploaded', False)
        st.session_state.setdefault('detected_domain', None)
        st.session_state.setdefault('processed_data', None)
        st.session_state.setdefault('trained_models', None)
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

        st.sidebar.subheader("Domain Selection")
        selected_domain = st.sidebar.selectbox("Select domain (or Auto-Detect)", ['Auto-Detect'] + DOMAINS)

        # Handle page navigation
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

    def upload_and_detect_page(self, selected_domain):
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader("Upload your dataset (CSV format only)", type=["csv", "xls", "xlsx", "txt", "json", "pdf", ])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                self.logger.log_info(f"Dataset loaded successfully! Shape: {data.shape}")
                st.success(f"✅ Dataset loaded! Shape: {data.shape}")

                # Auto or manual domain selection
                if selected_domain == "Auto-Detect":
                    detection_result = self.detector.detect_domain(data)
                    detected_domain = detection_result.get("domain")
                    confidence = detection_result.get("confidence")

                    if detected_domain:
                        st.success(f"✅ Detected Domain: {detected_domain} (Confidence: {confidence:.2f})")
                    else:
                        st.warning("⚠️ No domain detected with high confidence.")
                        return
                else:
                    detected_domain = selected_domain
                    st.info(f"ℹ️ Manual Domain Selection: {detected_domain}")

                # Process domain-specific logic
                processed_data = self.processor.process_domain_data(data, detected_domain)
                st.session_state['processed_data'] = processed_data
                st.dataframe(processed_data.head())

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
        st.info("Feature under development.")

    def predictions_page(self):
        st.header("📉 Predictions")
        st.info("Feature under development.")

    def system_logs_page(self):
        st.header("🧾 System Logs")
        st.info("Logs will appear here in future updates.")


if __name__ == "__main__":
    app = MultiDomainMLSystem()
    app.run()
