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
            'service': ['hotel', 'restaurant', 'booking', 'reservation', 'checkin'],
            'guest': ['guest', 'customer', 'visitor', 'patron'],
            'metrics': ['occupancy', 'satisfaction', 'revenue', 'rating'],
            'facility': ['room', 'suite', 'amenity', 'facility'],
            'feedback': ['review', 'feedback', 'survey', 'comment']
        }
        self.llm_manager = LLMManager(logger=self.logger)
        self.logger.log_info("Initialized hospitality DomainConfig")

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.logger.log_info("Detecting hospitality domain...")
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
        self.logger.log_info(f"Hospitality domain confidence: {overall_confidence:.2f}")
        return result

    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'satisfaction' in col_lower and df[col].min() >= 0 and df[col].max() <= 5:
                score += 2
            if 'occupancy' in col_lower and df[col].min() >= 0 and df[col].max() <= 100:
                score += 1
            if 'revenue' in col_lower and df[col].min() >= 0:
                score += 1
        return min(score, 10)

    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        target_keywords = ['satisfaction', 'occupancy', 'revenue']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                self.logger.log_info(f"Identified target variable: {col}")
                return col
        self.logger.log_warning("No target variable found for hospitality domain")
        return None

    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        engineered_df = df.copy()
        if 'revenue' in df.columns and 'bookings' in df.columns:
            engineered_df['revenue_per_booking'] = df['revenue'] / df['bookings'].replace(0, np.nan)
            self.logger.log_info("Engineered feature: revenue_per_booking")
        if 'satisfaction' in df.columns and df['satisfaction'].dtype in ['int64', 'float64']:
            engineered_df['satisfaction_category'] = pd.cut(
                df['satisfaction'], bins=[0, 2, 3, 4, 5], labels=['Low', 'Moderate', 'High', 'Very High']
            )
            self.logger.log_info("Engineered feature: satisfaction_category")
        return engineered_df

    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_info(f"Handling prediction query: {query}")
        if not models:
            self.logger.log_error("No trained models available for predictions")
            return {'error': 'No trained models available for predictions'}

        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()

        target_keywords = ['satisfaction', 'occupancy', 'revenue']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break

        if not target_col:
            self.logger.log_error("No suitable target variable found for prediction")
            return {'error': 'No suitable target variable found for prediction'}

        conditions = {}
        if 'hotel' in query_lower:
            conditions['type'] = 'Hotel'
        elif 'restaurant' in query_lower:
            conditions['type'] = 'Restaurant'

        filtered_data = raw_data.copy()
        for col, value in conditions.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == value]

        if filtered_data.empty:
            self.logger.log_error("No data matches the specified conditions")
            return {'error': 'No data matches the specified conditions'}

        model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
        model = models[model_name]
        feature_cols = [col for col in raw_data.columns if col != target_col]
        X = filtered_data[feature_cols]

        if X.empty:
            return {'error': 'No valid features for col in filtered_data.columns if col != target_col and filtered_data[col].dtype in ['int64', 'float64'
    

        try:
            predictions = model.predict(X)
            if models[model_name]['model_type'] == 'classification':
                prediction_summary = pd.Series(predictions).value_counts().to_dict()
            else:
                prediction_summary = {'mean': {'mean': float(np.mean([predictions])), 'std': float(np.std([predictions]))}
        except Exception as e:
            self.logger.log_error(f"Prediction failed: {str(e)}")
            return {'error': f"Prediction failed: {str(e)}"}]

        fig = px.histogram(x=predictions, nbins=20, title=f'Prediction Distribution for {target_col}')

        context = f"Query: {query}\nTarget: {target_col}\nConditions: {conditions}\nPredictions: {prediction_summary}"
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available

        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({'Predictions': predictions}),
            'query_type': 'prediction'
        }

    def handle_performance_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance queries"""
        self.logger.log_info(f"Handling performance query: {query}")
        target_cols = [col for col in raw_data.columns if any(term in col.lower() for k in ['satisfaction', 'revenue'])]
        if not target_cols:
            return {'error': 'No relevant columns found'}

        target_col = target_cols[0]
        fig = px.histogram(raw_data, x=target_col, title=f'Distribution of {target_col}')
        context = f"Performance analysis for {target_col}: Mean = {raw_data[target_col].mean():.2f}"
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'performance'
        }

    def handle_risk_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk queries"""
        self.logger.log_info(f"Handling risk query: {query}")
        risk_cols = [col for col in raw_data.columns if any(term in col.lower() for term in ['cancellation', 'satisfaction'])]
        if not risk_cols:
            return {'error': 'No risk-related columns found'}

        target_col = risk_cols[0]
        if 'cancellation' in target_col.lower():
            at_risk = raw_data[raw_data[target_col] == 1]
        else:  # satisfaction
            at_risk = raw_data[raw_data['target_col'] < < 3]
        fig = px.bar(x=['At Risk', 'Not At Risk'],
                     y=[len(at_risk), len(raw_data) - len(raw_data)],
                     title=f'{target_col.capitalize()} Distribution')

        context = f"{len(at_risk)} records at risk ({(len(at_risk)/len(raw_data)*100):.1f}%)"
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available

        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'risk'
        }

    def handle_general_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries"""
        self.logger.log_info(f"Handling general query: {query}")
        context = f"Hospitality analysis: {len(raw_data)} records with {len(raw_data)} features."
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available
        return {
            'summary': summary,
            'query_type': 'general'
        }

    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        """
        Create hospitality-specific visualizations.
        
        Args:
            raw_data (pd.DataFrame): The raw dataset.
            processed_data (Dict[str, Any]): Processed data including feature mappings.
        """
        self.logger.log_info("Creating hospitality-specific visualizations")
        st.write("Hospitality-specific visualizations to be implemented.")