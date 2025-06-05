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
        self.domain_keywords = {
            'crop': ['crop', 'plant', 'variety', 'seed', 'harvest'],
            'metrics': ['yield', 'production', 'growth', 'biomass', 'loss'],
            'soil': ['soil', 'nutrient', 'ph', 'moisture', 'fertility'],
            'weather': ['rain', 'temperature', 'humidity', 'wind'],
            'farm': ['farm', 'field', 'plot', 'acre', 'hectare'],
            'pest': ['pest', 'disease', 'insect', 'weed']
        }
        self.llm_manager = LLMManager(logger=self.logger)
        self.logger.log_info("Initialized agriculture DomainConfig")

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.logger.log_info("Detecting agriculture domain...")
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
        keyword_confidence = min(total_matches / total_keywords, 1.0) if total_keywords > 0 else 0.0
        pattern_confidence = pattern_score / 10
        overall_confidence = (keyword_confidence * 0.7 + pattern_confidence * 0.3)

        result = {
            'is_domain': overall_confidence >= 0.3,
            'confidence': overall_confidence,
            'detected_features': detected_features
        }
        self.logger.log_info(f"Agriculture domain confidence: {overall_confidence:.2f}")
        return result

    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'yield' in col_lower and df[col].min() >= 0:
                score += 3
            if 'moisture' in col_lower and df[col].min() >= 0:
                score += 2
            if 'temperature' in col_lower:
                score += 1
            if 'ph' in col_lower and 0 <= df[col].min() <= 14:
                score += 2
        return min(score, 10)

    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        target_keywords = ['yield', 'production', 'loss']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                self.logger.log_info(f"Identified target variable: {col}")
                return col
        self.logger.log_warning("No target variable found for agriculture domain")
        return None

    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        engineered_df = df.copy()
        if 'yield' in df.columns and 'area' in df.columns:
            engineered_df['yield_per_area'] = df['yield'] / df['area'].replace(0, np.nan)
            self.logger.log_info("Engineered feature: yield_per_area")
        if 'rain' in df.columns and 'temperature' in df.columns:
            engineered_df['weather_index'] = df['rain'] * df['temperature']
            self.logger.log_info("Engineered feature: weather_index")
        return engineered_df

    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_info(f"Handling prediction query: {query}")
        if not models:
            self.logger.log_error("No trained models available for predictions")
            return {'error': 'No trained models available for predictions'}

        query_lower = query.lower()
        target_keywords = ['yield', 'production', 'loss']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break

        if not target_col:
            self.logger.log_error("No suitable target variable found for prediction")
            return {'error': 'No suitable target variable found for prediction'}

        conditions = {}
        if 'wheat' in query_lower:
            conditions['crop_type'] = 'Wheat'
        elif 'corn' in query_lower:
            conditions['crop_type'] = 'Corn'

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
            prediction_summary = {'mean': float(np.mean(predictions)), 'std': float(np.std(predictions))}
        except Exception as e:
            self.logger.log_error(f"Prediction failed: {str(e)}")
            return {'error': f"Prediction failed: {str(e)}"}

        fig = px.histogram(x=predictions, nbins=20, title=f'Prediction Distribution for {target_col}')
        context = f"Query: {query}\nTarget: {target_col}\nConditions: {conditions}\nPredictions: {prediction_summary}"
        summary = context  # Placeholder until LLMManager is implemented
        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({'Predictions': predictions}),
            'query_type': 'prediction'
        }

    def handle_performance_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_info(f"Handling performance query: {query}")
        target_cols = [col for col in raw_data.columns if any(k in col.lower() for k in ['yield', 'production', 'loss'])]
        if not target_cols:
            self.logger.log_error("No relevant columns found")
            return {'error': 'No relevant columns found'}

        target_col = target_cols[0]
        fig = px.histogram(raw_data, x=target_col, title=f'Distribution of {target_col}')
        context = f"Performance analysis for {target_col}: Mean = {raw_data[target_col].mean():.2f}"
        summary = context
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'performance'
        }

    def handle_risk_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_info(f"Handling risk query: {query}")
        risk_cols = [col for col in raw_data.columns if any(term in col.lower() for term in ['loss', 'pest', 'disease'])]
        if not risk_cols:
            self.logger.log_error("No risk-related columns found")
            return {'error': 'No risk-related columns found'}

        target_col = risk_cols[0]
        at_risk = raw_data[raw_data[target_col] > raw_data[target_col].quantile(0.75)]
        fig = px.bar(x=['At Risk', 'Not At Risk'],
                     y=[len(at_risk), len(raw_data) - len(at_risk)],
                     title=f'{target_col.capitalize()} Distribution')
        context = f"{len(at_risk)} records at risk ({(len(at_risk)/len(raw_data)*100):.1f}%)"
        summary = context
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'risk'
        }

    def handle_general_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_info(f"Handling general query: {query}")
        context = f"Agriculture analysis: {len(raw_data)} records with {len(raw_data.columns)} features."
        summary = context
        return {'summary': summary, 'query_type': 'general'}

    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        self.logger.log_info("Creating agriculture-specific visualizations")
        if 'yield' in raw_data.columns and 'moisture' in raw_data.columns:
            fig = px.scatter(raw_data, x='moisture', y='yield', color='crop_type',
                             title='Yield vs. Soil Moisture by Crop Type')
            st.plotly_chart(fig)
        else:
            st.write("Insufficient data for agriculture visualizations.")