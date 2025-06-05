import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from utils.llm_manager import LLMManager

class DomainConfig:
    def __init__(self):
        self.domain_keywords = {
            'network': ['signal', 'bandwidth', 'network'],
            'customer': ['subscriber', 'user'],
            'metrics': ['churn', 'usage', 'latency']
        }
        self.llm_manager = LLMManager()
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to telecommunications domain"""
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
        """Check for telecommunications data patterns"""
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'latency' in col_lower and df[col].min() >= 0:
                score += 2
        return min(score, 10)
    
    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable"""
        target_keywords = ['churn', 'usage']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                return col
        return None
    
    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer telecommunications-specific features"""
        engineered_df = df.copy()
        if 'usage' in df.columns and 'time' in df.columns:
            engineered_df['usage_per_hour'] = df['usage'] / df['time']
        return engineered_df
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM and T5"""
        if not models:
            return {'error': 'No trained models available for predictions'}
        
        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()
        
        target_keywords = ['churn', 'usage']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break
        
        if not target_col:
            return {'error': 'No suitable target variable found for prediction'}
        
        conditions = {}
        if '5g' in query_lower:
            conditions['network_type'] = '5G'
        
        filtered_data = raw_data.copy()
        for col, value in conditions.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == value]
        
        if filtered_data.empty:
            return {'error': 'No data matches the specified conditions'}
        
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
        
        fig = px.histogram(x=predictions, nbins=20, title=f'Prediction Distribution for {target_col}')
        
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
        target_cols = [col for col in raw_data.columns if any(k in col.lower() for k in ['usage', 'latency'])]
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
        risk_cols = [col for col in raw_data.columns if any(term in col.lower() for term in ['churn'])]
        if not risk_cols:
            return {'error': 'No risk-related columns found'}
        
        target_col = risk_cols[0]
        at_risk = raw_data[raw_data[target_col] == 1]
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
        context = f"Telecommunications analysis: {len(raw_data)} records with {len(raw_data.columns)} features."
        summary = self.llm_manager.generate_response(context)
        return {'summary': summary, 'query_type': 'general'}
    
    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        """Create telecommunications-specific visualizations"""
        st.write(f"Telecommunications-specific visualizations to be implemented.")