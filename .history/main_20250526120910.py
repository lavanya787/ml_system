# main.py
import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
import importlib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.domain_detector import DomainDetector
from utils.data_processor import DataProcessor
from model_manager import ModelManager
from utils.query_handler import QueryHandler
from utils.visualizer import Visualizer
from utils.logger import SystemLogger

# Available domains
DOMAINS = ['medical', 'healthcare', 'transport', 'devices', 'education']

# Set page config
st.set_page_config(page_title="Multi-Domain ML System", page_icon="🧠", layout="wide")

class MultiDomainMLSystem:
    def __init__(self):
        self.detector = DomainDetector()
        self.processor = DataProcessor()
        self.model_manager = ModelManager()
        self.query_handler = QueryHandler()
        self.visualizer = Visualizer()
        self.logger = SystemLogger()
        
        # Initialize session state
        if 'dataset_uploaded' not in st.session_state:
            st.session_state.dataset_uploaded = False
        if 'detected_domain' not in st.session_state:
            st.session_state.detected_domain = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = None
    
    def run(self):
        st.title("🧠 Multi-Domain ML System")
        st.markdown("### Automatically detects and analyzes datasets from multiple domains")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a section", [
            "Dataset Upload & Detection",
            "Data Analysis",
            "Model Training",
            "Intelligent Query",
            "Predictions",
            "System Logs"
        ])
        
        # Domain selection
        st.sidebar.subheader("Domain Selection")
        selected_domain = st.sidebar.selectbox("Select domain (or Auto-Detect)", ['Auto-Detect'] + DOMAINS)
        
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
        
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.session_state.raw_data = df
                st.session_state.dataset_uploaded = True
                
                # Log dataset upload
                self.logger.log_query(
                    query_type="dataset_upload",
                    query=f"Dataset uploaded: {uploaded_file.name}",
                    dataset_info=f"Shape: {df.shape}, Columns: {list(df.columns)}"
                )
                
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                
                # Auto-detect domain or use selected
                with st.spinner("🔍 Analyzing dataset to detect domain..."):
                    if selected_domain == 'Auto-Detect':
                        detection_result = self.detector.detect_domain(df)
                        st.session_state.detected_domain = detection_result['domain']
                    else:
                        detection_result = self.detector.detect_domain(df, force_domain=selected_domain)
                        st.session_state.detected_domain = selected_domain
                
                st.success(f"🧠 Detected Domain: **{st.session_state.detected_domain}** (Confidence: {detection_result['confidence']:.1%})")
                
                # Process data
                with st.spinner(f"🔧 Processing {st.session_state.detected_domain} dataset..."):
                    processed_data = self.processor.process_domain_data(df, detection_result, st.session_state.detected_domain)
                    st.session_state.processed_data = processed_data
                
                st.success("✅ Data processed and ready for analysis!")
                
                # Display dataset overview
                st.subheader("📊 Dataset Overview")
                st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                self.logger.log_query(query_type="error", query="Dataset upload failed", response=str(e))
    
    # Other methods (data_analysis_page, model_training_page, etc.) remain similar but use detected_domain