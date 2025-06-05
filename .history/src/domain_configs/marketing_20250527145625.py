import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from utils.llm_manager import LLMManager
from utils.logger import SystemLogger as Logger

class DomainConfig:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.domain_keywords = {
            'campaign': ['campaign', 'ad', 'id', 'promotion', 'marketing'],
            'metrics': ['clicks', 'impressions', 'conversion', 'ctr', 'roi'],
            'audience': ['user', 'segment', 'demographic', 'target'],
            'channel': ['social', 'email', 'web', 'mobile'],
            'engagement': ['likes', 'shares', 'comments', 'views']
        }
        self.llm_manager = LLMManager(logger=self.logger)
        self.logger.log_info("Initialized marketing DomainConfig")

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.logger.log_info("Detecting marketing domain...")
        columns = [col.lower() for col in df.columns]
        detected_features = {category: [] for category in self.domain_keywords.keys()}
        total_matches = 0

        for category, keywords in self.domain_keywords.items():
            for keyword in keywords:
                matching_columns = [col for col in columns if keyword in col]
                detected_features[category].extend(matching_columns)
                total_matches += len(matching_columns)

        pattern_score = self._check_data_patterns(df)
        total_keywords = sum(len(keywords) for keywords in self.domain_keywords.values())
        keyword_confidence = min(total_matches / total_keywords, 1.0)
        pattern_confidence = pattern_score / 10
        overall_confidence = (keyword_confidence * 0.7 + pattern_confidence * 0.3)

        result = {
            'is_domain': overall_confidence >= 0.3,
            'confidence': overall_confidence,
            'detected_features': detected_features
        }
        self.logger.log_info(f"Marketing domain confidence: {overall_confidence:.2f}")
        return result

    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'clicks' in col_lower and df[col].min() >= 0:
                score += 2
            if 'impressions' in col_lower and df[col].min() >= 0:
                score += 1
            if 'conversion' in col_lower and df[col].min() >= 0:
                score += 1
        return min(score, 10)

    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        target_keywords = ['conversion', 'clicks', 'roi']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                self.logger.log_info(f"Identified target variable: {col}")
                return col
        self.logger.log_warning("No target variable found for marketing domain")
        return None

    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        engineered_df = df.copy()
        if 'clicks' in df.columns and 'impressions' in df.columns:
            engineered_df['ctr'] = df['clicks'] / df['impressions'].replace(0, np.nan)
            self.logger.log_info("Engineered feature: ctr")
        if 'cost' in df.columns and 'conversion' in df.columns:
            engineered_df['cost_per_conversion'] = df['cost'] / df['conversion'].replace(0, np.nan)
            self.logger.log_info("Engineered feature: cost_per_conversion")
        return engineered_df

    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_info(f"Handling prediction query: {query}")
        if not models:
            self.logger.log_error("No trained models available for predictions")
            return {'error': 'No trained models available for predictions'}

        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()

        target_keywords = ['conversion', 'clicks', 'roi']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break

        if not target_col:
            self.logger.log_error("No suitable target variable found for prediction")
            return {'error': 'No suitable target variable found for prediction'}

        conditions = {}
        if