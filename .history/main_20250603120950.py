import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import re
from typing import Optional
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import utilities individually to avoid circular imports
from utils.domain_detector import DomainDetector
from utils.data_processor import DataProcessor
from utils.logger import Logger
from utils.llm_manager import LLMManager
from utils.model_handler import ModelHandler
from utils.query_handler import QueryHandler


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
        
        # Initialize handlers only if imports were successful
        self.model_handler = ModelHandler(logger=self.logger) if ModelHandler else None
        self.query_handler = QueryHandler(logger=self.logger, llm_manager=self.llm_manager) if QueryHandler else None
        
        # Initialize essential session state variables only
        st.session_state.setdefault('dataset_uploaded', False)
        st.session_state.setdefault('detected_domain', None)
        st.session_state.setdefault('processed_data', None)
        st.session_state.setdefault('raw_data', None)
        st.session_state.setdefault('models', {})
        st.session_state.setdefault('user_id', '')
    
    def run(self):
        st.title("🧠 Multi-Domain ML System")
        st.markdown("### Automatically detects and analyzes datasets from multiple domains")
        
        # Check if critical components are available
        if not self.model_handler:
            st.error("❌ ModelHandler could not be loaded. Please check circular imports.")
        if not self.query_handler:
            st.error("❌ QueryHandler could not be loaded. Please check circular imports.")
        
        # Manual User ID Input
        self.handle_user_id_input()
        
        # Main navigation
        available_pages = ["Dataset Upload & Detection", "Data Analysis"]
        
        # Add advanced features only if handlers are available
        if self.query_handler:
            available_pages.append("Intelligent Query")
        if self.model_handler:
            available_pages.extend(["Model Training", "Predictions"])
        
        page = st.sidebar.selectbox("Choose a section", available_pages)

        if page == "Dataset Upload & Detection":
            self.upload_and_detect_page()
        elif page == "Model Training" and self.model_handler:
            self.model_training_page()
        elif page == "Intelligent Query" and self.query_handler:
            self.intelligent_query_page()
        elif page == "Data Analysis":
            self.data_analysis_page()
        elif page == "Predictions" and self.model_handler:
            self.predictions_page()
        elif page == "System Logs" and self.logger:
            self.logs_page()
    
    def handle_user_id_input(self):
        """Handle manual user ID input"""
        st.sidebar.title("User Configuration")
        user_id = st.sidebar.text_input("👤 Enter User ID:", value=st.session_state.get('user_id', ''))
        
        if user_id:
            st.session_state['user_id'] = user_id
            st.sidebar.success(f"✅ User ID set: {user_id}")
        else:
            st.sidebar.warning("⚠️ Please enter a User ID to continue")
    
    def read_uploaded_file(self, uploaded_file):
        """Simplified file reader for common formats"""
        file_type = uploaded_file.name.split('.')[-1].lower()

        try:
            if file_type == "csv":
                return pd.read_csv(uploaded_file, encoding='utf-8')
            elif file_type in ["xlsx", "xls"]:
                return pd.read_excel(uploaded_file)
            elif file_type == "txt":
                content = uploaded_file.read().decode(errors="ignore")
                return pd.DataFrame({"text": content.splitlines()})
            else:
                st.error("❌ Unsupported file type. Please use CSV, Excel, or TXT files.")
                return None
        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")
            return None
 
    def upload_and_detect_page(self):
        st.subheader("📤 Upload Dataset")
        
        if not st.session_state.get('user_id'):
            st.warning("⚠️ Please enter a User ID first")
            return
            
        uploaded_file = st.file_uploader(
            "Upload a dataset file",
            type=["csv", "xlsx", "xls", "txt"],
            accept_multiple_files=False
        )

        if uploaded_file:
            st.markdown(f"#### 📁 Processing: `{uploaded_file.name}`")
            data = self.read_uploaded_file(uploaded_file)

            if data is None:
                return

            st.success(f"✅ Dataset loaded! Shape: {data.shape}")

            # Auto-detect domain
            detected = self.detector.detect_domain(data)
            detected_domain = detected.get('domain', None)
            confidence = detected.get('confidence', 0)

            if detected_domain:
                st.info(f"Detected domain: **{detected_domain}** (confidence: {confidence:.2f})")
            else:
                st.warning("Could not confidently detect a domain.")

            # Process data
            processed_data = self.processor.process_domain_data(data, detected_domain) if detected_domain else data

            # Update session state
            st.session_state['raw_data'] = data
            st.session_state['detected_domain'] = detected_domain
            st.session_state['processed_data'] = processed_data
            st.session_state['dataset_uploaded'] = True

            # Show preview
            st.subheader("📊 Data Preview")
            st.dataframe(processed_data.head())

    def data_analysis_page(self):
        st.header("📈 Data Analysis")
        
        if not st.session_state.get('dataset_uploaded', False):
            st.warning("⚠️ Please upload a dataset first")
            return

        data = st.session_state['processed_data']
        detected_domain = st.session_state.get('detected_domain')
        
        if detected_domain:
            st.info(f"📊 Analysis for **{detected_domain.replace('_', ' ').title()}** domain")

        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Dataset Info")
            st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
            st.write("**Columns:**", list(data.columns))
        
        with col2:
            st.subheader("🔍 Missing Values")
            missing_df = data.isnull().sum()
            missing_df = missing_df[missing_df > 0]
            if not missing_df.empty:
                st.write(missing_df)
            else:
                st.write("No missing values! ✅")

        # Statistical summary for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.subheader("📊 Statistical Summary")
            st.dataframe(numeric_data.describe())

        # Simple visualization
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("📈 Distribution")
            selected_col = st.selectbox("Select column", numeric_cols)
            fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
    def model_training_page(self):
        st.header("🛠️ Model Training")
        
        if not self.model_handler:
            st.error("❌ Model training not available due to import issues.")
            return
        
        if not st.session_state.get('dataset_uploaded'):
            st.warning("⚠️ Please upload a dataset first")
            return

        domain = st.session_state.get('detected_domain')
        data = st.session_state['processed_data']

        if not domain:
            st.warning("⚠️ Domain not detected. Please upload an appropriate dataset.")
            return

        st.write(f"Training model for domain: **{domain}**")

        if st.button("🚀 Train Model"):
            try:
                with st.spinner("Training model..."):
                    model = self.model_handler.train_model(data, domain)
                    st.session_state['models'][domain] = model
                    st.success(f"✅ Model trained successfully for domain '{domain}'!")
            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")

    def predictions_page(self):
        st.header("🔮 Predictions")
        
        if not self.model_handler:
            st.error("❌ Predictions not available due to import issues.")
            return
        
        if not st.session_state.get('models'):
            st.warning("⚠️ No trained models available. Please train a model first.")
            return

        data = st.session_state['processed_data']
        models = st.session_state['models']
        
        target_col = st.selectbox("Select Model", list(models.keys()))
        
        if st.button("🎯 Generate Predictions"):
            try:
                with st.spinner("Generating predictions..."):
                    model_info = models[target_col]
                    features = model_info['features']
                    
                    X = data[features].dropna()
                    predictions = model_info['model'].predict(X)
                    
                    results_df = data.loc[X.index].copy()
                    results_df['Predictions'] = predictions
                    
                    st.success(f"✅ Generated {len(predictions)} predictions!")
                    st.dataframe(results_df[['Predictions'] + features[:3]].head(10))
                    
                    # Simple visualization
                    fig = px.histogram(x=predictions, title=f"Prediction Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")


if __name__ == "__main__":
    app = MultiDomainMLSystem()
    app.run()