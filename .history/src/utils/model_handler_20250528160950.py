import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Union
from io import BytesIO, StringIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


class ModelHandler:
    def __init__(self, logger=None):
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger("ModelHandler")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = logger

        # Create models directory if not exist
        os.makedirs('models', exist_ok=True)

        # Initialize scaler
        self.scaler = StandardScaler()

        # To store model results
        self.model_results = {}

    def load_data_from_file(self, uploaded_file: Union[BytesIO, StringIO]) -> pd.DataFrame:
        """Detect file type and load into pandas DataFrame"""
        try:
            filename = getattr(uploaded_file, 'name', None)
            if filename is None:
                # If no name, try to read as CSV by default
                df = pd.read_csv(uploaded_file)
            elif filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
            elif filename.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, Excel or JSON files.")
                return pd.DataFrame()
            self.logger.info(f"Loaded data from file: {filename}")
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            self.logger.error(f"Error loading file: {str(e)}")
            return pd.DataFrame()

    def train_model(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Generic model training on any DataFrame with specified target column.
        This replaces train_education_models to be more flexible.
        """
        if df is None or df.empty:
            st.error("Empty dataset. Please upload a valid file.")
            return {}

        if target_col not in df.columns:
            st.error(f"Target column '{target_col}' not found in data.")
            return {}

        # Prepare features and target
        X, y = self._prepare_features_target(df, target_col)

        if X.empty or len(y) == 0:
            st.error("Insufficient data for model training after preprocessing.")
            return {}

        # Determine task type
        is_classification = self._is_classification_task(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Get models
        models_to_train = self._get_models(is_classification)
        results = {}

        for name, model in models_to_train.items():
            try:
                if 'Linear' in name or 'Logistic' in name:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

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
                self.logger.info(f"Saved model '{name}' at {model_path}")

            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
                self.logger.error(f"Training error for {name}: {str(e)}")

        self.model_results = results
        return results
    
    def _prepare_features_target(self, df: pd.DataFrame, target_col: str) -> tuple:
        """Prepare features and target for modeling"""
        feature_cols = []
        for col in df.columns:
            if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
                # Remove columns that are all NaN or zero std
                if not df[col].isna().all() and df[col].std() != 0:
                    feature_cols.append(col)

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Remove rows with missing data in features or target
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        return X, y

    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if the task is classification or regression"""
        unique_values = y.nunique()
        if unique_values <= 10 and pd.api.types.is_object_dtype(y):
            return True
        if unique_values <= 5 and pd.api.types.is_numeric_dtype(y):
            return True
        return False

    def _get_model(self, is_classification: bool) -> Dict[str, Any]:
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
    def load_model(self, model_name: str) -> Any:
        """Load a trained model from disk"""
        model_path = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"Model '{model_name}' not found.")
            self.logger.error(f"Model '{model_name}' not found at {model_path}")
            return None
        
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Loaded model '{model_name}' from {model_path}")
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.logger.error(f"Error loading model '{model_name}': {str(e)}")
            return None

    def display_model_results(self, results: Dict[str, Any]):
        """Display model training results in Streamlit"""
        if not results:
            st.error("No model results to display.")
            return
        
        task_type = list(results.values())[0]['model_type']
        
        results_data = []
        for name, result in results.items():
            if task_type == 'classification':
                results_data.append({'Model': name, 'Accuracy': result['accuracy']})
            else:
                results_data.append({'Model': name, 'RMSE': result['rmse'], 'R² Score': result['r2']})
        
        results_df = pd.DataFrame(results_data)
        
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if task_type == 'classification':
                fig1 = px.bar(results_df, x='Model', y='Accuracy', title='Model Accuracy Comparison')
            else:
                fig1 = px.bar(results_df, x='Model', y='R² Score', title='Model R² Score Comparison')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
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
                # Add perfect prediction line
                min_val = min(best_result['actual'].min(), best_result['predictions'].min())
                max_val = max(best_result['actual'].max(), best_result['predictions'].max())
                fig2.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', name='Perfect Prediction'
                ))
                st.plotly_chart(fig2, use_container_width=True)

    def create_prediction_interface(self, model: Dict[str, Any], processed_data: Dict[str, Any], detected_features: Dict[str, Any]):
        """Create Streamlit interface for making predictions"""
        if not model:
            st.error("No trained models available.")
            return
        
        task_type = list(model.values())[0]['model_type']
        if task_type == 'classification':
            best_model_name = max(model.keys(), key=lambda k: models[k]['accuracy'])
        else:
            best_model_name = max(model.keys(), key=lambda k: model[k]['r2'])
        
        best_model = model[best_model_name]['model']
        
        st.subheader(f"Make Predictions using {best_model_name}")
        
        # Create input form
        input_data = self._create_input_form(processed_data, detected_features)
        
        if st.button("Make Prediction", type="primary"):
            try:
                prediction = self._make_prediction(best_model, input_data, best_model_name)
                self._display_prediction_results(prediction, task_type)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    def _create_input_form(self, processed_data: Dict[str, Any], detected_features: Dict[str, Any]) -> Dict[str, Any]:
        """Create dynamic input form for user input in Streamlit"""
        input_data = {}
        df = processed_data['processed_features']
        
        st.write("Enter values for prediction:")
        
        col1, col2 = st.columns(2)
        
        # Use numeric features only
        numeric_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        numeric_features = [col for col in numeric_features if not col.endswith('_encoded')]
        
        for i, feature in enumerate(numeric_features[:10]):  # limit to 10 features
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
        """Make prediction with the given model and input data"""
        input_df = pd.DataFrame([input_data])
        
        if 'Linear' in model_name or 'Logistic' in model_name:
            input_df = self.scaler.transform(input_df)
        
        prediction = model.predict(input_df)
        return prediction

    def _display_prediction_results(self, prediction, task_type: str):
        """Display prediction results in Streamlit"""
        st.success("Prediction Result:")
        if task_type == 'classification':
            st.write(f"**Predicted Class:** {prediction[0]}")
        else:
            st.write(f"**Predicted Value:** {prediction[0]:.2f}")
    def save_model(self, model: Any, model_name: str):
        model_path = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
        obj_to_save = {
            'model': model,
            'scaler': self.scaler
        }
        joblib.dump(obj_to_save, model_path)
        self.logger.info(f"Saved model and scaler at {model_path}")

    def load_model(self, model_name: str) -> Any:
        model_path = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"Model '{model_name}' not found.")
            self.logger.error(f"Model '{model_name}' not found at {model_path}")
            return None

        try:
            obj = joblib.load(model_path)
            self.scaler = obj.get('scaler', StandardScaler())  # fallback
            self.logger.info(f"Loaded model and scaler from {model_path}")
            return obj['model']
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.logger.error(f"Error loading model '{model_name}': {str(e)}")
            return None
