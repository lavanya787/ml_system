import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from utils.llm_manager import LLMManager

class DomainConfig:
    def __init__(self):
        self.domain_keywords = {
            'patient_identifiers': ['patient', 'id', 'record', 'mrn'],
            'clinical_measurements': ['blood_pressure', 'heart_rate', 'glucose', 'bmi'],
            'diagnoses': ['diagnosis', 'icd', 'disease', 'condition'],
            'treatments': ['treatment', 'medication', 'procedure'],
            'outcomes': ['survival', 'readmission', 'recovery'],
            'demographics': ['age', 'gender', 'ethnicity']
        }
        self.llm_manager = LLMManager()
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to healthcare domain"""
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
        """Check for healthcare data patterns"""
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'heart_rate' in col_lower and df[col].min() >= 40 and df[col].max() <= 120:
                score += 2
            elif 'age' in col_lower and df[col].min() >= 0 and df[col].max() <= 120:
                score += 1
        return min(score, 10)
    
    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable"""
        target_keywords = ['survival', 'readmission', 'recovery']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                return col
        return None
    
    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer healthcare-specific features"""
        engineered_df = df.copy()
        if 'age' in df.columns:
            engineered_df['age_group'] = pd.cut(df['age'], 
                                               bins=[0, 18, 40, 60, 80, 120],
                                               labels=['Pediatric', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
        return engineered_df
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM and T5"""
        if not models:
            return {'error': 'No trained models available for predictions'}
        
        # Extract intent and entities using LLM
        query_result = self.llm_manager.process_query([query])[0]
        intent = query_result['intent']
        query_lower = query.lower()
        
        # Identify target variable
        target_keywords = ['readmission', 'survival', 'recovery']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break
        
        if not target_col:
            return {'error': 'No suitable target variable found for prediction'}
        
        # Identify conditions (e.g., "for seniors")
        conditions = {}
        if 'senior' in query_lower or 'elderly' in query_lower:
            conditions['age'] = (60, 120)
        elif 'adult' in query_lower:
            conditions['age'] = (18, 60)
        
        # Filter data based on conditions
        filtered_data = raw_data.copy()
        for col, (min_val, max_val) in conditions.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[(filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)]
        
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
        
        # Generate natural language response using T5
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
        outcome_cols = [col for col in raw_data.columns if 'readmission' in col.lower()]
        if not outcome_cols:
            return {'error': 'No outcome columns found'}
        
        target_col = outcome_cols[0]
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
        outcome_cols = [col for col in raw_data.columns if 'readmission' in col.lower()]
        if not outcome_cols:
            return {'error': 'No outcome columns found'}
        
        target_col = outcome_cols[0]
        at_risk = raw_data[raw_data[target_col] == 1]
        fig = px.bar(x=['At Risk', 'Not At Risk'],
                    y=[len(at_risk), len(raw_data) - len(at_risk)],
                    title='Patient Risk Distribution')
        
        context = f"{len(at_risk)} patients at risk ({(len(at_risk)/len(raw_data)*100):.1f}%)"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'risk'
        }
    
    def handle_general_query(self, query: str, raw_data: pd.DataFrame,
                           processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries"""
        context = f"Healthcare analysis: {len(raw_data)} records with {len(raw_data.columns)} features."
        summary = self.llm_manager.generate_response(context)
        return {'summary': summary, 'query_type': 'general'}
    
    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        """Create healthcare-specific analysis"""
        st.write("Healthcare-specific visualizations to be implemented.")