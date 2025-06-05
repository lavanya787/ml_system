import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from utils.llm_manager import LLMManager
from utils.logger import Logger

class DomainConfig:
    def __init__(self, logger: Logger):
        """
        Initialize DomainConfig for e-commerce domain.

        Args:
            logger (Logger): Logger instance for logging information and errors.
        """
        self.logger = logger
        self.domain_keywords = {
            'sales': ['sale', 'revenue', 'price', 'amount', 'cost', 'income', 'profit', 'margin'],
            'customer': ['customer', 'user', 'client', 'id', 'account', 'profile', 'member'],
            'product': ['product', 'item', 'sku', 'category', 'brand', 'stock', 'inventory'],
            'transaction': ['order', 'purchase', 'transaction', 'cart', 'checkout', 'return', 'refund'],
            'behavior': ['click', 'view', 'rating', 'review', 'search', 'session', 'engagement'],
            'promotion': ['discount', 'coupon', 'offer', 'promo', 'deal'],
            'logistics': ['shipping', 'delivery', 'warehouse', 'fulfillment', 'tracking']
        }
        self.llm_manager = LLMManager(logger=self.logger)
        self.logger.log_info("Initialized e-commerce DomainConfig")

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to e-commerce domain."""
        self.logger.log_info("Detecting e-commerce domain...")
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
        self.logger.log_info(f"E-commerce domain confidence: {overall_confidence:.2f}")
        return result

    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for e-commerce data patterns."""
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'price' in col_lower or 'revenue' in col_lower or 'profit' in col_lower:
                score += 2
            if 'quantity' in col_lower and df[col].min() >= 0:
                score += 1
        return min(score, 10)

    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable."""
        target_keywords = ['revenue', 'sales', 'purchase', 'churn']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                self.logger.log_info(f"Identified target variable: {col}")
                return col
        self.logger.log_warning("No target variable found for e-commerce domain")
        return None

    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer e-commerce-specific features."""
        engineered_df = df.copy()
        if 'price' in df.columns and 'quantity' in df.columns:
            engineered_df['total_revenue'] = df['price'] * df['quantity']
            self.logger.log_info("Engineered feature: total_revenue")
        if 'rating' in df.columns and 'review' in df.columns:
            engineered_df['customer_satisfaction'] = df['rating'] * df['review'].notna().astype(int)
            self.logger.log_info("Engineered feature: customer_satisfaction")
        return engineered_df

    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM."""
        self.logger.log_info(f"Handling prediction query: {query}")
        if not models:
            self.logger.log_error("No trained models available for predictions")
            return {'error': 'No trained models available for predictions'}

        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()

        target_keywords = ['revenue', 'sales', 'churn']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break

        if not target_col:
            self.logger.log_error("No suitable target variable found for prediction")
            return {'error': 'No suitable target variable found for prediction'}

        conditions = {}
        if 'electronics' in query_lower:
            conditions['category'] = 'Electronics'
        elif 'clothing' in query_lower:
            conditions['category'] = 'Clothing'
        elif 'home' in query_lower:
            conditions['category'] = 'Home'

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
        sales_cols = [col for col in raw_data.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
        if not sales_cols:
            self.logger.log_error("No sales/revenue columns found")
            return {'error': 'No sales/revenue columns found'}

        target_col = sales_cols[0]
        fig = px.histogram(raw_data, x=target_col, title=f'Distribution of {target_col}')
        context = f"Sales analysis for {target_col}: Mean = {raw_data[target_col].mean():.2f}"
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
        churn_cols = [col for col in raw_data.columns if 'churn' in col.lower()]
        if not churn_cols:
            self.logger.log_error("No churn columns found")
            return {'error': 'No churn columns found'}

        target_col = churn_cols[0]
        at_risk = raw_data[raw_data[target_col] == 1]
        fig = px.bar(x=['Churn', 'No Churn'],
                     y=[len(at_risk), len(raw_data) - len(at_risk)],
                     title='Customer Churn Distribution')

        context = f"{len(at_risk)} customers at risk of churn ({(len(at_risk)/len(raw_data)*100):.1f}%)"
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
        context = f"E-commerce analysis: {len(raw_data)} records with {len(raw_data.columns)} features."
        summary = context  # Placeholder
        # summary = self.llm_manager.generate_response(context)  # Uncomment when available
        return {'summary': summary, 'query_type': 'general'}

    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        """Create e-commerce-specific analysis."""
        self.logger.log_info("Creating e-commerce-specific visualizations")
        st.write("E-commerce-specific visualizations to be implemented.")