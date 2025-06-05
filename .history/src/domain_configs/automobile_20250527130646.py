import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from utils.llm_manager import LLMManager
from utils.logger import SystemLogger as Logger

class DomainConfig:
    def __init__(self, logger: Logger):
        """
        Initialize DomainConfig for automobile domain.

        Args:
            logger (Logger): Logger instance for logging information and errors.
        """
        self.logger = logger
        self.domain_keywords = {
            'vehicle': ['car', 'vehicle', 'model', 'truck', 'suv', 'sedan', 'coupe', 'van'],
            'metrics': ['sales', 'mileage', 'price', 'revenue', 'inventory', 'demand'],
            'service': ['maintenance', 'repair', 'service', 'warranty', 'inspection'],
            'specifications': ['engine', 'horsepower', 'fuel', 'transmission', 'torque', 'weight']
        }
        self.llm_manager = LLMManager(logger=self.logger)
        self.logger.log_info("Initialized automobile DomainConfig")

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to automobile domain."""
        self.logger.log_info("Detecting automobile domain...")
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
        self.logger.log_info(f"Automobile domain confidence: {overall_confidence:.2f}")
        return result

    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for automobile data patterns."""
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'mileage' in col_lower and df[col].min() >= 0:
                score += 2
            if 'price' in col_lower and df[col].min() >= 0:
                score += 1
        return min(score, 10)

    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable."""
        target_keywords = ['sales', 'price']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                self.logger.log_info(f"Identified target variable: {col}")
                return col
        self.logger.log_warning("No target variable found for automobile domain")
        return None

    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer automobile-specific features."""
        engineered_df = df.copy()
        if 'price' in df.columns and 'mileage' in df.columns:
            engineered_df['price_per_mile'] = df['price'] / df['mileage']
            self.logger.log_info("Engineered feature: price_per_mile")
        return engineered_df

    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM."""
        self.logger.log_info(f"Handling prediction query: {query}")
        if not models:
            self.logger.log_error("No trained models available for predictions")
            return {'error': 'No trained models available for predictions'}

        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()

        target_keywords = ['sales', 'price']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break

        if not target_col:
            self.logger.log_error("No suitable target variable found for prediction")
            return {'error': 'No suitable target variable found for prediction'}

        conditions = {}
        if 'electric' in query_lower:
            conditions['type'] = 'Electric'
        elif 'hybrid' in query_lower:
            conditions['type'] = 'Hybrid'

        filtered_data = raw_data.copy()
        for col, value in conditions.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == value]

        if filtered_data.empty:
            self.logger.log_error("No data matches the specified conditions")
            return {'error': 'No data matches the specified conditions'}

        model_name = max(models.keys(), key=lambda k: models[k]['accuracy' if models[k]['model_type'] == 'classification' else 'r2'])
        model = models[model_name]['model']
        feature_cols = [col for col in filtered_data.columns if col != target_col and filtered_data[col].dtype in ['int64', 'float64']]
        X = filtered_data[feature_cols]

        if X.empty:
            self.logger.log_error("No valid features for prediction")
            return {'error': 'No valid features for prediction'}

        try:
            predictions = model.predict(X)
            if models[model_name]['model_type'] == 'classification':
                prediction_summary = pd.Series(predictions).value_counts().to_dict()
            else:
                prediction_summary = {'mean': float(np.mean(predictions)), 'std': float(np.std(predictions))}
        except Exception as e:
            self.logger.log_error(f"Prediction failed: {str(e)}")
            return {'error': f"Prediction failed: {str(e)}"}

        fig = px.histogram(x=predictions, nbins=20, title=f'Prediction Distribution for {target_col}')

        context = f"Query: {query}\nTarget: {target_col}\nConditions: {conditions}\nPredictions: {prediction_summary}"
        summary = context  # Placeholder until generate_response is implemented
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available

        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({'Predictions': predictions}),
            'query_type': 'prediction'
        }

    def handle_performance_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance queries."""
        self.logger.log_info(f"Handling performance query: {query}")
        target_cols = [col for col in raw_data.columns if any(k in col.lower() for k in ['sales', 'price'])]
        if not target_cols:
            self.logger.log_error("No relevant columns found for performance query")
            return {'error': 'No relevant columns found'}

        target_col = target_cols[0]
        fig = px.histogram(raw_data, x=target_col, title=f'Distribution of {target_col}')
        context = f"Performance analysis for {target_col}: Mean = {raw_data[target_col].mean():.2f}"
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available

        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'performance'
        }

    def handle_risk_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk queries."""
        self.logger.log_info(f"Handling risk query: {query}")
        risk_cols = [col for col in raw_data.columns if any(term in col.lower() for term in ['recall', 'defect'])]
        if not risk_cols:
            self.logger.log_error("No risk-related columns found")
            return {'error': 'No risk-related columns found'}

        target_col = risk_cols[0]
        at_risk = raw_data[raw_data[target_col] == 1]
        fig = px.bar(x=['At Risk', 'Not At Risk'],
                     y=[len(at_risk), len(raw_data) - len(at_risk)],
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
        """Handle general queries."""
        self.logger.log_info(f"Handling general query: {query}")
        context = f"Automobile analysis: {len(raw_data)} records with {len(raw_data.columns)} features."
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available
        return {'summary': summary, 'query_type': 'general'}

    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        """Create automobile-specific visualizations."""
        self.logger.log_info("Creating automobile-specific visualizations")
        st.write("Automobile-specific visualizations to be implemented.")