import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from utils.llm_manager import LLMManager
from utils.logger import Logger

class DomainConfig:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.llm_manager = LLMManager(logger=self.logger)
        self.domain_keywords = {
            'media': ['movie', 'show', 'episode'],
            'metrics': ['views', 'ratings'],
            'user': ['audience', 'viewer']
        }

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.logger.log_info("Detecting entertainment domain...")
        columns = [col.lower() for col in df.columns]
        detected_features = {category: [] for category in self.domain_keywords.keys()}
        total_matches = 0

        for category, keywords in self.domain_keywords.items():
            for keyword in keywords:
                matching_columns = [col for col in columns if keyword in col]
                detected_features[category].extend(matching_columns)
                total_matches += len(matching_columns)

        confidence = min(total_matches / 10, 1.0)
        result = {
            'is_domain': confidence >= 0.3,
            'confidence': confidence,
            'detected_features': detected_features
        }
        self.logger.log_info(f"Entertainment domain confidence: {confidence:.2f}")
        return result
    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for entertainment data patterns"""
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'rating' in col_lower and df[col].min() >= 0 and df[col].max() <= 10:
                score += 2
        return min(score, 10)
    
    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable"""
        target_keywords = ['rating', 'views']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                return col
        return None
    
    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer entertainment-specific features"""
        engineered_df = df.copy()
        if 'views' in df.columns:
            engineered_df['views_per_day'] = df['views'] / 30
        return engineered_df
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM and T5"""
        if not models:
            return {'error': 'No trained models available for predictions'}
        
        # Extract intent and entities
        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()
        
        # Identify target variable
        target_keywords = ['rating', 'views']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break
        
        if not target_col:
            return {'error': 'No suitable target variable found for prediction'}
        
        # Identify conditions
        conditions = {}
        if 'movie' in query_lower:
            conditions['type'] = 'Movie'
        
        # Filter data
        filtered_data = raw_data.copy()
        for col, value in conditions.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == value]
        
        if filtered_data.empty:
            return {'error': 'No data matches the specified conditions'}
        
        # Make predictions
        model_name = max(models.keys(), key=lambda k: models[k]['accuracy' if models[k]['model_type'] == 'classification' else 'r2'])
        model = models[model_name]['model']
        feature_cols = [col for col in filtered_data.columns if col != target_col and filtered_data[col].dtype in ['int64', 'float64']]
        X = filtered_data[feature_cols]
        
        if X.empty:
            return {'error': 'No valid features for prediction'}
        
        try:
            predictions = model.predict(X)
            if models[model_name]['model_type'] == 'classification':
                prediction_summary = pd.Series(predictions).value_counts().to_dict()
            else:
                prediction_summary = {'mean': float(np.mean(predictions)), 'std': float(np.std(predictions))}
        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}
        
        # Generate visualization
        fig = px.histogram(x=predictions, nbins=20, title=f'Prediction Distribution for {target_col}')
        
        # Generate response with T5
        context = f"Query: {query}\nTarget: {target_col}\nConditions: {conditions}\nPredictions: {prediction_summary}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({'Predictions': predictions}),
            'query_type': 'prediction'
        }
    
    def handle_performance_query(self, query: str, raw_data: pd.DataFrame,
                               processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance queries"""
        target_cols = [col for col in raw_data.columns if any(k in col.lower() for k in ['rating', 'views'])]
        if not target_cols:
            return {'error': 'No relevant columns found'}
        
        target_col = target_cols[0]
        fig = px.histogram(raw_data, x=target_col, title=f'Distribution of {target_col}')
        context = f"Performance analysis for {target_col}: Mean = {raw_data[target_col].mean():.2f}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'performance'
        }
    
    def handle_risk_query(self, query: str, raw_data: pd.DataFrame,
                         processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk queries"""
        risk_cols = [col for col in raw_data.columns if any(term in col.lower() for term in ['rating'])]
        if not risk_cols:
            return {'error': 'No risk-related columns found'}
        
        target_col = risk_cols[0]
        at_risk = raw_data[raw_data[target_col] < 3]
        fig = px.bar(x=['At Risk', 'Not At Risk'],
                          y=[len(at_risk), len(raw_data) - len(at_risk)],
                          title=f'{target_col.capitalize()} Distribution')
        
        context = f"{len(at_risk)} records at risk ({(len(at_risk)/len(raw_data)*100):.1f}%)"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'risk'
        }
    
    def handle_general_query(self, query: str, raw_data: pd.DataFrame,
                           processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries"""
        context = f"Entertainment analysis: {len(raw_data)} records with {len(raw_data.columns)} features."
        summary = self.llm_manager.generate_response(context)
        return {'summary': summary, 'query_type': 'general'}
    
    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        """Create entertainment-specific visualizations"""
        st.write(f"Entertainment-specific visualizations to be implemented.")