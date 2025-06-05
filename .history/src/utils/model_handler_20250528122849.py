import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
from utils.logger import Logger
import joblib
import os

class ModelHandler:
    def __init__(self, logger= None):
        self.logger = logging.getLogger("ModelHandler")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
    
    def train_education_models(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train specialized models for education data"""
        df = processed_data['processed_features']
        target_col = processed_data['target_variable']
        
        if not target_col or target_col not in df.columns:
            st.error("No suitable target variable found for model training.")
            return {}
        
        # Prepare features and target
        X, y = self._prepare_features_target(df, target_col)
        
        if X.empty or len(y) == 0:
            st.error("Insufficient data for model training.")
            return {}
        
        # Determine if it's regression or classification
        is_classification = self._is_classification_task(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models_to_train = self._get_models(is_classification)
        results = {}
        
        for name, model in models_to_train.items():
            try:
                # Train model
                if 'Linear' in name or 'Logistic' in name:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                if is_classification:
                    accuracy = accuracy_score(y_test, y_pred)
                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'predictions': y_pred,
                        'actual': y_test,
                        'model_type': 'classification'
                    }
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    results[name] = {
                        'model': model,
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'predictions': y_pred,
                        'actual': y_test,
                        'model_type': 'regression'
                    }
                
                # Save model
                model_path = f'models/{name.lower().replace(" ", "_")}_model.pkl'
                joblib.dump(model, model_path)
                
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
        
        self.model_results = results
        return results
    
    def _prepare_features_target(self, df: pd.DataFrame, target_col: str) -> tuple:
        """Prepare features and target for modeling"""
        # Remove non-numeric columns and target column
        feature_cols = []
        for col in df.columns:
            if col != target_col and df[col].dtype in ['int64', 'float64']:
                if not df[col].isna().all() and df[col].std() != 0:
                    feature_cols.append(col)
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if the task is classification or regression"""
        # Check if target has limited unique values or is categorical
        unique_values = y.nunique()
        if unique_values <= 10 and y.dtype == 'object':
            return True
        if unique_values <= 5 and y.dtype in ['int64', 'float64']:
            return True
        return False
    
    def _get_models(self, is_classification: bool) -> Dict[str, Any]:
        """Get appropriate models based on task type"""
        if is_classification:
            return {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }
        else:
            return {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            }
    
    def display_model_results(self, results: Dict[str, Any]):
        """Display model training results"""
        if not results:
            st.error("No model results to display.")
            return
        
        # Determine task type
        task_type = list(results.values())[0]['model_type']
        
        # Create results dataframe
        results_data = []
        for name, result in results.items():
            if task_type == 'classification':
                results_data.append({
                    'Model': name,
                    'Accuracy': result['accuracy']
                })
            else:
                results_data.append({
                    'Model': name,
                    'RMSE': result['rmse'],
                    'R² Score': result['r2']
                })
        
        results_df = pd.DataFrame(results_data)
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if task_type == 'classification':
                fig1 = px.bar(results_df, x='Model', y='Accuracy', title='Model Accuracy Comparison')
            else:
                fig1 = px.bar(results_df, x='Model', y='R² Score', title='Model R² Score Comparison')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Best model predictions vs actual
            if task_type == 'classification':
                best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            else:
                best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
            
            best_result = results[best_model_name]
            
            if task_type == 'regression':
                fig2 = px.scatter(
                    x=best_result['actual'], 
                    y=best_result['predictions'],
                    title=f'{best_model_name} - Actual vs Predicted',
                    labels={'x': 'Actual', 'y': 'Predicted'}
                )
                fig2.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Perfect Prediction'))
                st.plotly_chart(fig2, use_container_width=True)
    
    def create_prediction_interface(self, models: Dict[str, Any], processed_data: Dict[str, Any], detected_features: Dict[str, Any]):
        """Create interface for making predictions"""
        if not models:
            st.error("No trained models available.")
            return
        
        # Get best model
        task_type = list(models.values())[0]['model_type']
        if task_type == 'classification':
            best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
        else:
            best_model_name = max(models.keys(), key=lambda k: models[k]['r2'])
        
        best_model = models[best_model_name]['model']
        
        st.subheader(f"Make Predictions using {best_model_name}")
        
        # Create input form based on detected features
        input_data = self._create_input_form(processed_data, detected_features)
        
        if st.button("Make Prediction", type="primary"):
            try:
                prediction = self._make_prediction(best_model, input_data, best_model_name)
                self._display_prediction_results(prediction, task_type)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    def _create_input_form(self, processed_data: Dict[str, Any], detected_features: Dict[str, Any]) -> Dict[str, Any]:
        """Create dynamic input form based on dataset features"""
        input_data = {}
        df = processed_data['processed_features']
        
        st.write("Enter values for prediction:")
        
        col1, col2 = st.columns(2)
        
        # Get numeric features for input
        numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        numeric_features = [col for col in numeric_features if not col.endswith('_encoded')]
        
        for i, feature in enumerate(numeric_features[:10]):  # Limit to 10 features
            with col1 if i % 2 == 0 else col2:
                if 'age' in feature.lower():
                    input_data[feature] = st.slider(f"{feature.title()}", 5, 100, 18)
                elif 'grade' in feature.lower() or 'score' in feature.lower():
                    input_data[feature] = st.slider(f"{feature.title()}", 0, 100, 75)
                elif 'attendance' in feature.lower():
                    input_data[feature] = st.slider(f"{feature.title()}", 0, 100, 85)
                elif 'hour' in feature.lower():
                    input_data[feature] = st.slider(f"{feature.title()}", 0, 168, 20)
                else:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    input_data[feature] = st.slider(f"{feature.title()}", min_val, max_val, mean_val)
        
        return input_data
    
    def _make_prediction(self, model, input_data: Dict[str, Any], model_name: str):
        """Make prediction using the trained model"""
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        if 'Linear' in model_name or 'Logistic' in model_name:
            input_scaled = self.scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
        else:
            prediction = model.predict(input_df)[0]
        
        return prediction
    
    def _display_prediction_results(self, prediction, task_type: str):
        """Display prediction result in Streamlit"""
        st.success("Prediction completed successfully!")

        if task_type == 'classification':
            st.markdown(f"### 🎯 Predicted Class: `{prediction}`")
        else:
            st.markdown(f"### 📈 Predicted Value: `{prediction:.2f}`")

        # Add some visual flair
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=prediction if isinstance(prediction, (int, float)) else 0,
            delta={'reference': 50 if task_type == 'regression' else 0},
            title={'text': "Prediction Output"},
            gauge={
                'axis': {'range': [None, 100 if task_type == 'regression' else 1]},
                'bar': {'color': "green"},
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
