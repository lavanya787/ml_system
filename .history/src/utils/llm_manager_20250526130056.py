import streamlit as st
import pandas as pd
from typing import Dict, Any
from utils.logger import Logger
from utils.data_processor import DataProcessor
from utils.domain_detector import DomainDetector
from utils.model_handler import ModelHandler
from utils.query_handler import QueryHandler
from utils.visualizer import Visualizer

def main():
    # Initialize logger
    logger = Logger(log_dir="logs")
    logger.log_info("Starting Multi-Domain ML System")
    
    # Initialize components
    llm_manager = LLMManager(logger)
    data_processor = DataProcessor(logger)
    domain_detector = DomainDetector(logger)
    model_handler = ModelHandler(logger)
    query_handler = QueryHandler(logger, llm_manager)
    visualizer = Visualizer(logger)
    
    st.title("Multi-Domain ML System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        # Load and process data
        df = data_processor.load_data(uploaded_file)
        if df is not None:
            st.write("Dataset Preview")
            st.dataframe(df.head())
            
            # Detect domain
            detection_result = domain_detector.detect_domain(df)
            if detection_result['domain']:
                st.write(f"Detected Domain: {detection_result['domain']} (Confidence: {detection_result['confidence']:.2f})")
                domain_config = detection_result['config']
                
                # Process data
                processed_result = data_processor.process_data(df, domain_config)
                if 'error' not in processed_result:
                    processed_df = processed_result['data']
                    target_col = processed_result['target']
                    
                    # Train model
                    if st.button("Train Model"):
                        models = model_handler.train_model(processed_df, target_col)
                        st.write("Model trained successfully!")
                    
                    # Load models
                    models = model_handler.load_models(detection_result['domain'])
                    
                    # Query input
                    query = st.text_input("Enter your query:")
                    if query:
                        result = query_handler.handle_query(
                            query, domain_config, df, processed_result, models
                        )
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            st.write("Query Result")
                            st.write(result['summary'])
                            if 'visualization' in result:
                                visualizer.display_visualization(result['visualization'])
                            if 'data' in result:
                                st.write("Prediction Data")
                                st.dataframe(result['data'])
                else:
                    st.error(processed_result['error'])
            else:
                st.error("Could not detect domain with sufficient confidence.")
    
    logger.log_info("Application run completed")

if __name__ == "__main__":
    main()