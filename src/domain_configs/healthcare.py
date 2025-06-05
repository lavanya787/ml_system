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
            'patient_identifiers': ['patient', 'id', 'record', 'mrn'],
            'clinical_measurements': ['blood_pressure', 'heart_rate', 'glucose', 'bmi', 'temperature'],
            'diagnoses': ['diagnosis', 'icd', 'disease', 'condition', 'syndrome'],
            'treatments': ['treatment', 'medication', 'procedure', 'surgery', 'therapy'],
            'outcomes': ['survival', 'readmission', 'recovery', 'mortality'],
            'demographics': ['age', 'gender', 'ethnicity', 'race'],
            'billing': ['cost', 'charge', 'billing', 'insurance'],
            'facility': ['hospital', 'clinic', 'ward', 'facility']
        }
        self.llm_manager = LLMManager(logger=self.logger)
        self.logger.log_info("Initialized healthcare DomainConfig")

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.logger.log_info("Detecting healthcare domain...")
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
        self.logger.log_info(f"Healthcare domain confidence: {overall_confidence:.2f}")
        return result

    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'heart_rate' in col_lower and df[col].min() >= 40 and df[col].max() <= 120:
                score += 2
            if 'age' in col_lower and df[col].min() >= 0 and df[col].max() <= 120:
                score += 1
            if 'bmi' in col_lower and df[col].min() >= 15 and df[col].max() <= 50:
                score += 1
        return min(score, 10)

    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        target_keywords = ['survival', 'readmission', 'recovery', 'mortality']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                self.logger.log_info(f"Identified target variable: {col}")
                return col
        self.logger.log_warning("No target variable found for healthcare domain")
        return None

    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        engineered_df = df.copy()
        if 'age' in df.columns:
            engineered_df['age_group'] = pd.cut(df['age'], 
                                               bins=[0, 18, 40, 60, 80, 120],
                                               labels=['Pediatric', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
            self.logger.log_info("Engineered feature: age_group")
        if 'heart_rate' in df.columns and 'age' in df.columns:
            engineered_df['heart_rate_age_ratio'] = df['heart_rate'] / df['age'].replace(0, np.nan)
            self.logger.log_info("Engineered feature: heart_rate_age_ratio")
        return engineered_df

    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_info(f"Handling prediction query: {query}")
        if not models:
            self.logger.log_error("No trained models available for predictions")
            return {'error': 'No trained models available for predictions'}

        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()

        target_keywords = ['readmission', 'survival', 'recovery', 'mortality']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break

        if not target_col:
            self.logger.log_error("No suitable target variable found for prediction")
            return {'error': 'No suitable target variable found for prediction'}

        conditions = {}
        if 'senior' in query_lower or 'elderly' in query_lower:
            conditions['age'] = 'Senior'
        elif 'adult' in query_lower:
            conditions['age'] = 'Adult'
        elif 'pediatric' in query_lower:
            conditions['age'] = 'Pediatric'

        filtered_data = raw_data.copy()
        for col, value in conditions.items():
            if col == 'age' and 'age_group' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['age_group'] == value]
            elif col in filtered_data.columns:
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
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available

        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({'Predictions': predictions}),
            'query_type': 'prediction'
        }

    def handle_performance_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_info(f"Handling performance query: {query}")
        outcome_cols = [col for col in raw_data.columns if any(k in col.lower() for k in ['readmission', 'survival', 'recovery'])]
        if not outcome_cols:
            self.logger.log_error("No outcome columns found")
            return {'error': 'No outcome columns found'}

        target_col = outcome_cols[0]
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
        self.logger.log_info(f"Handling risk query: {query}")
        outcome_cols = [col for col in raw_data.columns if any(k in col.lower() for k in ['readmission', 'mortality'])]
        if not outcome_cols:
            self.logger.log_error("No outcome columns found")
            return {'error': 'No outcome columns found'}

        target_col = outcome_cols[0]
        if 'readmission' in target_col.lower() or 'mortality' in target_col.lower():
            at_risk = raw_data[raw_data[target_col] == 1]
        fig = px.bar(x=['At Risk', 'Not At Risk'],
                     y=[len(at_risk), len(raw_data) - len(at_risk)],
                     title=f'{target_col.capitalize()} Distribution')

        context = f"{len(at_risk)} patients at risk ({(len(at_risk)/len(raw_data)*100):.1f}%)"
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available

        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'risk'
        }

    def handle_general_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_info(f"Handling general query: {query}")
        context = f"Healthcare analysis: {len(raw_data)} records with {len(raw_data.columns)} features."
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available
        return {'summary': summary, 'query_type': 'general'}

    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        self.logger.log_info("Creating healthcare-specific visualizations")
        st.write("Healthcare-specific visualizations to be implemented.")