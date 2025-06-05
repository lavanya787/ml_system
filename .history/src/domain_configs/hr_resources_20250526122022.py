import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from llm_manager import LLMManager

class DomainConfig:
    def __init__(self):
        self.domain_keywords = {
            'employee': ['employee', 'staff', 'id'],
            'hr': ['recruitment', 'training'],
            'metrics': ['retention', 'performance', 'satisfaction']
        }
        self.llm_manager = LLMManager()
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to hr_resources domain"""
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
        keyword_confidence = min(total_matches / 10, 1.0)
        pattern_confidence = pattern_score / 10
        overall_confidence = (keyword_confidence * 0.7 + pattern_confidence * 0.3)
        
        return {
            'is_domain': overall_confidence >= 0.3,
            'confidence': overall_confidence,
            'detected_features': detected_features
        }
    
    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for hr_resources data patterns"""
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'satisfaction' in col_lower and df[col].min() >= 0 and df[col].max() <= 5:
                score += 2
        return min(score, 10)
    
    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable"""
        target_keywords = ['retention', 'performance']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                return col
        return None
    
    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer hr_resources-specific features"""
        engineered_df = df.copy()
        if 'tenure' in df.columns:
            engineered_df['tenure_years'] = df['tenure'] / 12
        return engineered_df
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM and T5"""