import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Union, List
from io import BytesIO, StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import streamlit as st
import plotly.express as px
from datetime import datetime
import json

class ModelHandler:
    def __init__(self):
self.logger = self._setup_logger()
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_results = {}
        self.qa_log = []
        self.trained_models = {}
        self.data_context = {}

    def _setup_logger(self):
        logger = logging.getLogger("ModelHandler")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler('logs/model_handler.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def train_model(self, df: pd.DataFrame, target_col: str, domain: str = None) -> Dict[str, Any]:
        """Train model for given dataframe and target column (backward compatibility)"""
        if domain is None:
            domain = "general"
        
        # Store data context for Q&A
        self.data_context = {
            'dataframe': df,
            'domain': domain,
            'target_column': target_col,
            'columns': list(df.columns),
            'shape': df.shape,
            'summary': df.describe().to_dict()
        }
        
        # Check if model exists for this domain
        if self._domain_model_exists(domain):
            self.logger.info(f"Retraining existing model for {domain} domain")
            return self._retrain_domain_model(df, target_col, domain)
        else:
            self.logger.info(f"Creating new model for {domain} domain")
            return self._train_new_domain_model(df, target_col, domain)

    def load_and_train(self, uploaded_file: Union[BytesIO, StringIO], domain: str, target_col: str = None) -> Dict[str, Any]:
        """Main method to load data, detect target, train models and save"""
        try:
            # Load data
            df = self._load_data(uploaded_file)
            if df.empty:
                return {}

            # Auto-detect target column if not provided
            if not target_col:
                target_col = self._detect_target_column(df, domain)
            
            return self.train_model(df, target_col, domain)

        except Exception as e:
            self.logger.error(f"Error in load_and_train: {str(e)}")
            return {}

    def _load_data(self, uploaded_file) -> pd.DataFrame:
        """Load data from various file formats"""
        try:
            filename = getattr(uploaded_file, 'name', 'unknown')
            if filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif filename.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)  # Default to CSV
            
            self.logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading file: {str(e)}")
            return pd.DataFrame()

    def _detect_target_column(self, df: pd.DataFrame, domain: str) -> str:
        """Auto-detect target column based on domain and column names"""
        target_keywords = {
            'education': ['grade', 'score', 'result', 'performance', 'pass', 'fail'],
            'healthcare': ['diagnosis', 'outcome', 'status', 'result', 'risk'],
            'finance': ['price', 'amount', 'profit', 'loss', 'default', 'fraud'],
            'sales': ['sales', 'revenue', 'profit', 'quantity', 'conversion'],
            'hr': ['salary', 'performance', 'rating', 'promotion', 'retention']
        }
        
        keywords = target_keywords.get(domain.lower(), ['target', 'label', 'y'])
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in keywords):
                return col
        
        # Fallback: last column
        return df.columns[-1]

    def _domain_model_exists(self, domain: str) -> bool:
        """Check if model already exists for the domain"""
        model_path = f'models/{domain}_model.pkl'
        return os.path.exists(model_path)

    def _retrain_domain_model(self, df: pd.DataFrame, target_col: str, domain: str) -> Dict[str, Any]:
        """Retrain existing domain model with new data"""
        try:
            # Load existing model package
            self._load_model_package(domain)
            existing_package = self.trained_models[domain]
            
            # Use same features as existing model if possible
            existing_features = existing_package['features']
            available_features = [col for col in existing_features if col in df.columns]
            
            if len(available_features) < len(existing_features) * 0.7:  # If less than 70% features match
                self.logger.warning(f"Feature mismatch for {domain}. Creating new model.")
                return self._train_new_domain_model(df, target_col, domain)
            
            # Retrain with existing architecture
            X, y = self._prepare_data_for_retraining(df, target_col, existing_package)
            if X.empty:
                return {}

            model_name = existing_package['model_name']
            is_classification = existing_package['is_classification']
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Get same model type
            models = self._get_models(is_classification)
            model = models[model_name]
            
            # Train
            if 'Linear' in model_name or 'Logistic' in model_name:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate metrics
            if is_classification:
                score = accuracy_score(y_test, y_pred)
                result_type = 'classification'
            else:
                score = r2_score(y_test, y_pred)
                result_type = 'regression'

            results = {model_name: {'model': model, 'score': score, 'type': result_type}}
            
            # Save retrained model
            self._save_model_package(model, domain, model_name, X.columns.tolist(), 
                                   target_col, is_classification)
            
            self.model_results = results
            self.logger.info(f"Successfully retrained {model_name} for {domain} domain")
            return results
            
        except Exception as e:
            self.logger.error(f"Error retraining model for {domain}: {str(e)}")
            return self._train_new_domain_model(df, target_col, domain)

    def _train_new_domain_model(self, df: pd.DataFrame, target_col: str, domain: str) -> Dict[str, Any]:
        """Train new model for domain"""
        results = self._train_and_save_models(df, target_col, domain)
        self.logger.info(f"Successfully created new model for {domain} domain")
        return results

    def _prepare_data_for_retraining(self, df: pd.DataFrame, target_col: str, existing_package: Dict[str, Any]):
        """Prepare data for retraining using existing preprocessing"""
        # Use existing label encoders where possible
        existing_encoders = existing_package.get('label_encoders', {})
        features = existing_package['features']
        
        # Select available features
        available_features = [col for col in features if col in df.columns]
        X = df[available_features].copy()
        y = df[target_col].copy()

        # Apply existing label encoders
        for col in X.columns:
            if X[col].dtype == 'object' and col in existing_encoders:
                try:
                    # Handle new categories by adding them to existing encoder
                    encoder = existing_encoders[col]
                    unique_vals = X[col].unique()
                    new_vals = [val for val in unique_vals if val not in encoder.classes_]
                    if new_vals:
                        encoder.classes_ = np.append(encoder.classes_, new_vals)
                    X[col] = encoder.transform(X[col].astype(str))
                    self.label_encoders[col] = encoder
                except:
                    # If encoding fails, create new encoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
            elif X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        # Handle target encoding
        if y.dtype == 'object':
            if target_col in existing_encoders:
                try:
                    encoder = existing_encoders[target_col]
                    unique_vals = y.unique()
                    new_vals = [val for val in unique_vals if val not in encoder.classes_]
                    if new_vals:
                        encoder.classes_ = np.append(encoder.classes_, new_vals)
                    y = encoder.transform(y.astype(str))
                    self.label_encoders[target_col] = encoder
                except:
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                    self.label_encoders[target_col] = le
            else:
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                self.label_encoders[target_col] = le

        # Remove rows with missing values
        mask = ~(X.isna().any(axis=1) | pd.isna(y))
        return X[mask], y[mask]
    def _train_and_save_models(self, df: pd.DataFrame, target_col: str, domain: str) -> Dict[str, Any]:
        """Train multiple models and save the best one"""
        X, y = self._prepare_data(df, target_col)
        if X.empty:
            return {}

        is_classification = self._is_classification(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = self._get_models(is_classification)
        results = {}

        for name, model in models.items():
            try:
                # Train model
                X_tr = X_train_scaled if 'Linear' in name or 'Logistic' in name else X_train
                X_te = X_test_scaled if 'Linear' in name or 'Logistic' in name else X_test
                
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)

                # Calculate metrics
                if is_classification:
                    score = accuracy_score(y_test, y_pred)
                    results[name] = {'model': model, 'score': score, 'type': 'classification'}
                else:
                    score = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    results[name] = {'model': model, 'score': score, 'rmse': rmse, 'type': 'regression'}

            except Exception as e:
                self.logger.error(f"Training error for {name}: {str(e)}")

        # Save best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['score'])
            self._save_model_package(results[best_model_name]['model'], domain, best_model_name, 
                                   X.columns.tolist(), target_col, is_classification)
            
        self.model_results = results
        return results

    def get_domain_model_info(self, domain: str) -> Dict[str, Any]:
        """Get information about existing domain model"""
        if not self._domain_model_exists(domain):
            return {'exists': False}
        
        try:
            self._load_model_package(domain)
            package = self.trained_models[domain]
            return {
                'exists': True,
                'model_name': package['model_name'],
                'features': package['features'],
                'target_column': package['target_column'],
                'is_classification': package['is_classification'],
                'trained_at': package['trained_at']
            }
        except Exception as e:
            self.logger.error(f"Error getting domain model info: {str(e)}")
            return {'exists': False}

    def _prepare_data(self, df: pd.DataFrame, target_col: str):
        """Prepare features and target, handle categorical variables"""
        # Separate numeric and categorical columns
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        # Handle target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self.label_encoders[target_col] = le

        # Remove rows with missing values
        mask = ~(X.isna().any(axis=1) | pd.isna(y))
        return X[mask], y[mask]

    def _is_classification(self, y) -> bool:
        """Determine if task is classification"""
        return len(np.unique(y)) <= 10

    def _get_models(self, is_classification: bool) -> Dict[str, Any]:
        """Get appropriate models"""
        if is_classification:
            return {
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=500)
            }
        else:
            return {
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'Linear Regression': LinearRegression()
            }

    def _save_model_package(self, model, domain: str, model_name: str, features: List[str], 
                           target_col: str, is_classification: bool):
        """Save complete model package"""
        package = {
            'model': model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'features': features,
            'target_column': target_col,
            'domain': domain,
            'model_name': model_name,
            'is_classification': is_classification,
            'trained_at': datetime.now().isoformat()
        }
        
        model_path = f'models/{domain}_model.pkl'
        joblib.dump(package, model_path)
        self.trained_models[domain] = package
        self.logger.info(f"Saved {model_name} model for {domain} domain at {model_path}")

    def answer_question(self, question: str, domain: str = None) -> Dict[str, Any]:
        """Answer questions about the data/model"""
        timestamp = datetime.now().isoformat()
        
        try:
            # Load model if not in memory
            if domain and domain not in self.trained_models:
                self._load_model_package(domain)

            # Generate answer
            answer = self._generate_answer(question, domain)
            has_answer = answer != "I don't have enough information to answer this question."
            
            # Log Q&A
            qa_entry = {
                'timestamp': timestamp,
                'question': question,
                'answer': answer,
                'domain': domain,
                'has_answer': has_answer
            }
            
            self.qa_log.append(qa_entry)
            self._save_qa_log()
            
            self.logger.info(f"Q&A - Domain: {domain}, Has Answer: {has_answer}")
            return qa_entry

        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return {
                'timestamp': timestamp,
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'domain': domain,
                'has_answer': False
            }

    def _generate_answer(self, question: str, domain: str) -> str:
        """Generate answer based on data context and model"""
        question_lower = question.lower()
        
        if not self.data_context:
            return "No data loaded. Please train a model first."

        df = self.data_context['dataframe']
        
        # Basic statistics questions
        if 'how many' in question_lower and ('row' in question_lower or 'record' in question_lower):
            return f"The dataset contains {df.shape[0]} rows."
        
        if 'column' in question_lower and 'how many' in question_lower:
            return f"The dataset has {df.shape[1]} columns: {', '.join(df.columns)}."
        
        if 'average' in question_lower or 'mean' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col_name = self._find_relevant_column(question, numeric_cols)
                if col_name:
                    return f"The average {col_name} is {df[col_name].mean():.2f}."
        
        if 'maximum' in question_lower or 'highest' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col_name = self._find_relevant_column(question, numeric_cols)
                if col_name:
                    return f"The maximum {col_name} is {df[col_name].max():.2f}."
        
        if 'minimum' in question_lower or 'lowest' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col_name = self._find_relevant_column(question, numeric_cols)
                if col_name:
                    return f"The minimum {col_name} is {df[col_name].min():.2f}."
        
        # Model performance questions
        if 'accuracy' in question_lower or 'performance' in question_lower:
            if self.model_results:
                best_model = max(self.model_results.keys(), key=lambda k: self.model_results[k]['score'])
                score = self.model_results[best_model]['score']
                if self.model_results[best_model]['type'] == 'classification':
                    return f"The best model ({best_model}) achieved {score:.2%} accuracy."
                else:
                    return f"The best model ({best_model}) achieved an R² score of {score:.3f}."
        
        return "I don't have enough information to answer this question."

    def _find_relevant_column(self, question: str, columns) -> str:
        """Find most relevant column based on question keywords"""
        question_words = question.lower().split()
        for col in columns:
            if any(word in col.lower() for word in question_words):
                return col
        return columns[0] if len(columns) > 0 else None

    def _load_model_package(self, domain: str):
        """Load saved model package"""
        model_path = f'models/{domain}_model.pkl'
        if os.path.exists(model_path):
            package = joblib.load(model_path)
            self.trained_models[domain] = package
            self.scaler = package['scaler']
            self.label_encoders = package['label_encoders']

    def _save_qa_log(self):
        """Save Q&A log to file"""
        log_path = 'logs/qa_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.qa_log, f, indent=2)

    def predict(self, input_data: Dict[str, Any], domain: str) -> Any:
        """Make prediction with trained model"""
        if domain not in self.trained_models:
            self._load_model_package(domain)
        
        package = self.trained_models[domain]
        model = package['model']
        features = package['features']
        
        # Prepare input
        input_df = pd.DataFrame([input_data])
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[features]
        
        # Apply label encoding
        for col, encoder in package['label_encoders'].items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Scale if needed
        if 'Linear' in package['model_name'] or 'Logistic' in package['model_name']:
            input_df = package['scaler'].transform(input_df)
        
        prediction = model.predict(input_df)
        
        # Decode if classification
        if package['is_classification'] and package['target_column'] in package['label_encoders']:
            prediction = package['label_encoders'][package['target_column']].inverse_transform(prediction)
        
        return prediction[0]

    def display_results(self):
        """Display model results in Streamlit"""
        if not self.model_results:
            st.error("No model results to display.")
            return
        
        results_data = []
        for name, result in self.model_results.items():
            if result['type'] == 'classification':
                results_data.append({'Model': name, 'Accuracy': f"{result['score']:.2%}"})
            else:
                results_data.append({'Model': name, 'R² Score': f"{result['score']:.3f}"})
        
        st.subheader("Model Performance")
        st.dataframe(pd.DataFrame(results_data))
        
        best_model = max(self.model_results.keys(), key=lambda k: self.model_results[k]['score'])
        st.success(f"Best Model: {best_model}")