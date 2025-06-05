#
import pandas as pd
from typing import Dict, Any
import streamlit as st
from utils.llm_manager import LLMManager
from utils.logger import Logger

class DomainConfig:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.domain_keywords = {'generic': ['value', 'count', 'id', 'data', 'feature', 'col', 'variable']}
        self.llm_manager = LLMManager(logger=self.logger)
        self.logger.log_info("Initialized generic DomainConfig")

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.logger.log_info("Detecting generic domain...")
        columns = [col.lower() for col in df.columns]
        detected_features = {'generic': columns}
        total_matches = len(columns)
        total_keywords = len(self.domain_keywords['generic'])
        keyword_confidence = min(total_matches / total_keywords, 1.0) if total_keywords > 0 else 0.0
        pattern_score = 5  # Neutral score
        overall_confidence = (keyword_confidence * 0.7 + (pattern_score / 10) * 0.3)
        result = {
            'is_domain': overall_confidence >= 0.1,
            'confidence': overall_confidence,
            'detected_features': detected_features
        }
        self.logger.log_info(f"Generic domain confidence: {overall_confidence:.2f}")
        return result

    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        self.logger.log_info(f"Identified target: {numeric_cols[0] if len(numeric_cols) > 0 else None}")
        return numeric_cols[0] if len(numeric_cols) > 0 else None

    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        self.logger.log_info("No feature engineering for generic domain")
        return df.copy()

    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_warning("Generic domain does not support predictions")
        return {'error': 'Generic domain does not support predictions'}

    def handle_performance_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        summary = f"Generic analysis: {len(raw_data)} records with {len(raw_data.columns)} features"
        self.logger.log_info(summary)
        return {'summary': summary, 'query_type': 'performance'}

    def handle_risk_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_warning("Generic domain does not support risk analysis")
        return {'summary': 'Generic domain does not support risk analysis', 'query_type': 'risk'}

    def handle_general_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        summary = f"Generic analysis for query '{query}': {len(raw_data)} records"
        self.logger.log_info(summary)
        return {'summary': summary, 'query_type': 'general'}

    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        self.logger.log_info("Creating generic analysis")
        st.write(f"Generic domain analysis: {len(raw_data)} records with {len(raw_data.columns)} features.")