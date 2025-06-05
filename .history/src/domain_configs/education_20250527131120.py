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
        Initialize DomainConfig for education domain.

        Args:
            logger (Logger): Logger instance for logging information and errors.
        """
        self.logger = logger
        self.domain_keywords = {
            'student_identifiers': ['student', 'pupil', 'learner', 'id', 'enrollment', 'registration'],
            'academic_performance': ['grade', 'score', 'mark', 'gpa', 'result', 'achievement', 'rank'],
            'subjects_courses': ['math', 'science', 'english', 'course', 'history', 'language', 'biology', 'physics'],
            'demographics': ['age', 'gender', 'class', 'grade_level', 'ethnicity', 'parent'],
            'attendance_behavior': ['attendance', 'absent', 'behavior', 'participation', 'tardy', 'discipline'],
            'assessment': ['exam', 'test', 'quiz', 'assignment', 'project'],
            'institution': ['school', 'college', 'university', 'teacher', 'faculty', 'extracurricular']
        }
        self.llm_manager = LLMManager(logger=self.logger)
        self.logger.log_info("Initialized education domain")

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to education domain."""
        self.logger.log_info("Detecting education domain...")
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
        overall_confidence = (keyword_confidence * 0.6 + pattern_confidence * 0.4)

        result = {
            'is_domain': overall_confidence >= 0.3,
            'confidence': overall_confidence,
            'detected_features': detected_features
        }
        self.logger.log_info(f"Education domain confidence: {overall_confidence:.2f}")
        return result

    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for education data patterns."""
        score = 0
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'grade' in col_lower or 'score' in col_lower and df[col].max() <= 100 and df[col].min() >= 0:
                score += 2
            elif 'age' in col_lower and df[col].max() <= 30 and df[col].min() >= 4:
                score += 1
            elif 'attendance' in col_lower and df[col].max() <= 100 and df[col].min() >= 0:
                score += 1
        return min(score, 10)

    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable."""
        target_keywords = ['grade', 'score', 'gpa']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                self.logger.log_info(f"Identified target variable: {col}")
                return col
        self.logger.log_warning("No target variable found for education domain")
        return None

    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer education-specific features."""
        engineered_df = df.copy()
        grade_cols = [col for col in df.columns if 'grade' in col.lower() or 'score' in col.lower()]
        for col in grade_cols:
            if df[col].dtype in ['int64', 'float64']:
                engineered_df[f"{col}_category"] = pd.cut(
                    df[col], bins=[0, 60, 70, 80, 90, 100], labels=['F', 'D', 'C', 'B', 'A']
                )
                self.logger.log_info(f"Engineered feature: {col}_category")
        if 'attendance' in df.columns and df['attendance'].dtype in ['int64', 'float64']:
            engineered_df['attendance_rate'] = df['attendance'] / 100.0
            self.logger.log_info("Engineered feature: attendance_rate")
        return engineered_df

    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM."""
        self.logger.log_info(f"Handling prediction query: {query}")
        if not models:
            self.logger.log_error("No trained models available for predictions")
            return {'error': {'training': 'No models available'}}

        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()

        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in ['grade', 'score', 'gpa']):
                target_col = col
                break

        if not target_col:
            self.logger("No suitable target variable found for prediction")
            return {'error': {'target': 'No suitable target variable found'}}  

        conditions = {}
        if 'math' in query_lower:
            conditions['subject'] = 'Mathematics'
        elif 'science' in query_lower:
            conditions['subject'] = 'Science'
        elif 'english' in query_lower:
            conditions['subject'] = 'English'

        filtered_data = raw_data.copy()
        for col, value in conditions.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col].str.contains(value, case=False, na=False)]

        if filtered_data.empty:
            self.logger.log_error("No data matches the specified conditions")
            return {'error': {'data': 'No matching data'}}  

        model_name = max(models.keys(), key=lambda k: models[k].get('rating', models[k].get('score', 0)))
        model = models[model_name]['model']
        feature_cols = [col for col in filtered_data.columns if col != target_col and filtered_data[col].dtype in ['int64', 'float64']]
        X = filtered_data[feature_cols]

        if X.empty:
            self.logger.log_error("No valid features for prediction")
            return {'error': {'features': 'No valid features'}}  

        try:
            predictions = model.predict(X)
            prediction_summary = {'mean': float(np.mean(predictions)), 'std': float(np.std(predictions))}
        except Exception as e:
            self.logger.log_error(f"Prediction failed: {str(e)}")
            return {'error': {'prediction': str(e)}}  

        fig = px.histogram(x=predictions, nbins=50, title=f'Predicted {target_col} Distribution')  

        context = {
            'query': query,
            'target': target_col,
            'conditions': conditions,
            'predictions': prediction_summary
        }
        summary = str(context)  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available

        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({'predictions': predictions}),
            'query_type': 'predictive'
        }

    def handle_performance_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance queries."""
        self.logger.log_info(f"Handling performance query: {query}")
        grade_cols = [col for col in raw_data.columns if 'grade' in col.lower() or 'score' in col.lower()]
        if not grade_cols:
            self.logger.log_error("No suitable columns found")
            return {'error': 'No grade/score columns'}

        target_col = grade_cols[0]
        fig = px.histogram(raw_data, x=target_col, title=f'{target_col.capitalize()} Distribution')

        context = f"Performance analysis for {target_col}: mean = {raw_data[target_col].mean():.2f}, std = {raw_data[target_col].std():.2f}" 
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_summary(context)  # Use when available

        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({target_col: raw_data[target_col]}),
            'query_type': 'performance'
        }

    def handle_risk_query(self, query: str, raw_data: pd.DataFrame, processed_data: Dict[str, pd.DataFrame -> Dict[str, Any]:
        """Handle risk queries."""
        self.logger.log_info(f"Handling risk query: {query}")
        risk_cols = [col for col in raw_data.columns if any(term in col.lower() for term in ['grade', 'score', 'attendance'])]
        if not risk_cols:
            self.logger.log_error("No suitable columns found")
            return {'error': 'No risk-related columns'}

        target_col = risk_cols[0]
        at_risk = raw_data[raw_data[target_col] < 50] if 'grade' in target_col.lower() or 'score' in target_col.lower() else raw_data[raw_data[target_col] < 50]
        fig = px.bar(x=['At Risk', 'Standard'],
                   y=[len(at_risk), len(raw_data) - len(at_risk)],
                   title=f'{target_col} Risk Distribution')

        context = f"{len(at_risk)} students at risk ({(len(at_risk)/len(raw_data)*100):.1f}%)"
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_summary(context)  # Use when available

        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({'risk_status': ['At Risk']*len(at_risk) + ['Standard']*(len(raw_data) - len(at_risk))}),
            'query_type': 'risk'
        }

    def handle_general_query(self, query: str, raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> Dict:
        """Handle general queries."""
        self.logger.log_info(f"Handling general query: {query}")
        context = f"Education dataset: {len(raw_data)} records, {len(raw_data.columns)} features"
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_summary(context)  # Use when available

        return {
            'summary': summary,
            'query_type': 'general'
        }

    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        """Create education-specific visualizations."""
        self.logger.log_info("Creating education visualizations...")
        st.write("Education visualizations to be implemented.")