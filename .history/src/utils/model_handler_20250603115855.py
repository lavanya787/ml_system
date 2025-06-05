import pandas as pd
import numpy as np
import logging
import os
import hashlib
from typing import Dict, Any, Union, List, Optional
from io import BytesIO, StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.domain_detector import DomainDetector
from utils.data_processor import DataProcessor
from utils.logger import Logger
from utils.query_handler import QueryHandler
import joblib
import streamlit as st
import plotly.express as px
from datetime import datetime
import json

class ModelHandler:
    def __init__(self, logger: Logger = None):
        self.logger = self._setup_logger() if logger is None else logger
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('metadata', exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_results = {}
        self.qa_log = []
        self.trained_models = {}
        self.data_context = {}
        self.file_metadata_path = 'metadata/trained_files.json'
        self.file_metadata = self._load_file_metadata()

    def _setup_logger(self):
        """Setup Python's built-in logger as fallback"""
        logger = logging.getLogger("ModelHandler")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler('logs/model_handler.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _log_info(self, message: str):
        """Universal info logging method"""
        if hasattr(self.logger, 'log_info'):
            self.logger.log_info(message)
        elif hasattr(self.logger, 'info'):
            self.logger.info(message)
        else:
            print(f"INFO: {message}")

    def _log_error(self, message: str, exc_info: bool = False):
        """Universal error logging method"""
        if hasattr(self.logger, 'log_error'):
            self.logger.log_error(message)
        elif hasattr(self.logger, 'error'):
            self.logger.error(message, exc_info=exc_info)
        else:
            print(f"ERROR: {message}")

    def _log_warning(self, message: str):
        """Universal warning logging method"""
        if hasattr(self.logger, 'log_warning'):
            self.logger.log_warning(message)
        elif hasattr(self.logger, 'warning'):
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")

    def _load_file_metadata(self) -> Dict:
        """Load file metadata for duplicate detection"""
        try:
            if os.path.exists(self.file_metadata_path):
                with open(self.file_metadata_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self._log_error(f"Error loading file metadata: {str(e)}")
            return {}

    def _save_file_metadata(self):
        """Save file metadata to disk"""
        try:
            with open(self.file_metadata_path, 'w') as f:
                json.dump(self.file_metadata, f, indent=2, default=str)
        except Exception as e:
            self._log_error(f"Error saving file metadata: {str(e)}")

    def _generate_file_hash(self, uploaded_file) -> str:
        """Generate unique hash for uploaded file"""
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            
            # Generate hash
            hash_sha256 = hashlib.sha256()
            hash_sha256.update(file_content)
            file_hash = hash_sha256.hexdigest()
            
            # Reset file pointer for later use
            uploaded_file.seek(0)
            return file_hash
        except Exception as e:
            self._log_error(f"Error generating file hash: {str(e)}")
            return str(hash(str(uploaded_file)))

    def _register_trained_file(self, file_hash: str, filename: str, domain: str, 
                              target_col: str, model_info: Dict):
        """Register a newly trained file in metadata"""
        try:
            self.file_metadata[file_hash] = {
                'filename': filename,
                'domain': domain,
                'target_column': target_col,
                'training_date': datetime.now().isoformat(),
                'model_path': f'models/{domain}_model.pkl',
                'model_name': model_info.get('best_model', 'Unknown'),
                'performance': model_info.get('best_score', 0),
                'model_type': model_info.get('model_type', 'Unknown')
            }
            self._save_file_metadata()
            self._log_info(f"Registered trained file: {filename} for domain: {domain}")
        except Exception as e:
            self._log_error(f"Error registering trained file: {str(e)}")

    def _check_existing_model_for_file(self, file_hash: str) -> Optional[Dict]:
        """Check if model already exists for this file"""
        return self.file_metadata.get(file_hash)

    def _display_existing_model_info(self, model_info: Dict):
        """Display existing model information to user"""
        st.warning("🔍 **Duplicate File Detected!**")
        st.info("A model has already been trained with this exact file.")
        
        # Create an expander to show model details
        with st.expander("📊 **View Existing Model Details**", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**File Information:**")
                st.write(f"• **Filename:** {model_info['filename']}")
                st.write(f"• **Domain:** {model_info['domain']}")
                st.write(f"• **Target Column:** {model_info['target_column']}")
                
            with col2:
                st.write("**Model Information:**")
                st.write(f"• **Model Type:** {model_info['model_name']}")
                st.write(f"• **Performance:** {model_info['performance']:.4f}")
                st.write(f"• **Training Date:** {model_info['training_date'][:10]}")
                
        st.write("**What would you like to do?**")

    def _get_user_choice_for_duplicate(self) -> str:
        """Get user's choice when duplicate file is detected"""
        choice = st.radio(
            "Choose an option:",
            options=[
                "Use existing model",
                "Retrain with new data (overwrite existing)",
                "Cancel training"
            ],
            index=0,
            key="duplicate_choice"
        )
        return choice

    def load_and_train(self, uploaded_file: Union[BytesIO, StringIO], domain: str, target_col: str = None) -> Dict[str, Any]:
        """Main method to load data, detect duplicates, and train models"""
        try:
            filename = getattr(uploaded_file, 'name', 'unknown_file')
            self._log_info(f"Processing file: {filename}")
            
            # Generate file hash for duplicate detection
            file_hash = self._generate_file_hash(uploaded_file)
            self._log_info(f"Generated file hash: {file_hash[:8]}...")
            
            # Check for existing model
            existing_model = self._check_existing_model_for_file(file_hash)
            
            if existing_model:
                self._log_info(f"Found existing model for file: {filename}")
                
                # Display existing model info
                self._display_existing_model_info(existing_model)
                
                # Get user choice
                user_choice = self._get_user_choice_for_duplicate()
                
                if user_choice == "Use existing model":
                    st.success("✅ Using existing trained model!")
                    
                    # Load existing model
                    existing_domain = existing_model['domain']
                    self._load_model_package(existing_domain)
                    
                    # Load the data for context (without retraining)
                    df = self._load_data(uploaded_file)
                    self.data_context = {
                        'dataframe': df.copy(),
                        'target_column': existing_model['target_column'],
                        'domain': existing_domain
                    }
                    
                    return {
                        'status': 'existing_model_loaded',
                        'message': f"Loaded existing model for {existing_model['domain']} domain",
                        'model_info': existing_model,
                        'using_existing': True
                    }
                
                elif user_choice == "Retrain with new data (overwrite existing)":
                    st.info("🔄 Proceeding with retraining...")
                    return self._proceed_with_training(uploaded_file, domain, target_col, 
                                                     file_hash, filename, is_retrain=True)
                
                else:  # Cancel training
                    st.info("❌ Training cancelled by user.")
                    return {
                        'status': 'cancelled',
                        'message': 'Training cancelled by user choice'
                    }
            
            else:
                # No existing model found, proceed with training
                self._log_info(f"No existing model found for file: {filename}")
                st.info("🚀 No existing model found. Starting fresh training...")
                return self._proceed_with_training(uploaded_file, domain, target_col, 
                                                 file_hash, filename, is_retrain=False)

        except Exception as e:
            self._log_error(f"Error in load_and_train: {str(e)}")
            return {'error': str(e)}

    def _proceed_with_training(self, uploaded_file, domain: str, target_col: str, 
                             file_hash: str, filename: str, is_retrain: bool = False) -> Dict[str, Any]:
        """Proceed with actual training"""
        try:
            # Load data
            df = self._load_data(uploaded_file)
            if df.empty:
                return {'error': 'Failed to load data or empty dataset'}

            # Auto-detect target column if not provided
            if not target_col:
                target_col = self._detect_target_column(df, domain)
                self._log_info(f"Auto-detected target column: {target_col}")
            
            # Train the model
            results = self.train_model(df, target_col, domain)
            
            if 'error' not in results:
                # Register the trained file
                best_model_name = max(results.keys(), key=lambda k: results[k]['score'])
                best_score = results[best_model_name]['score']
                model_type = results[best_model_name]['type']
                
                model_info = {
                    'best_model': best_model_name,
                    'best_score': best_score,
                    'model_type': model_type
                }
                
                self._register_trained_file(file_hash, filename, domain, target_col, model_info)
                
                success_msg = "🎉 Model retrained successfully!" if is_retrain else "🎉 Model trained successfully!"
                st.success(success_msg)
                
                results['status'] = 'training_completed'
                results['file_registered'] = True
                
            return results

        except Exception as e:
            self._log_error(f"Error in training process: {str(e)}")
            return {'error': str(e)}

    def get_trained_files_summary(self) -> List[Dict]:
        """Get summary of all trained files"""
        summary = []
        for file_hash, info in self.file_metadata.items():
            summary.append({
                'file_hash': file_hash[:8] + '...',
                'filename': info['filename'],
                'domain': info['domain'],
                'target_column': info['target_column'],
                'model_name': info['model_name'],
                'performance': info['performance'],
                'training_date': info['training_date'][:10],
                'model_path': info['model_path']
            })
        return summary

    def display_trained_files_summary(self):
        """Display summary of all trained files in Streamlit"""
        summary = self.get_trained_files_summary()
        
        if not summary:
            st.info("No trained models found.")
            return
        
        st.subheader("📋 Trained Models Summary")
        
        # Convert to DataFrame for better display
        df_summary = pd.DataFrame(summary)
        
        # Rename columns for better readability
        df_summary = df_summary.rename(columns={
            'filename': 'File Name',
            'domain': 'Domain',
            'target_column': 'Target Column',
            'model_name': 'Model Type',
            'performance': 'Performance',
            'training_date': 'Training Date'
        })
        
        # Display with better formatting
        st.dataframe(
            df_summary[['File Name', 'Domain', 'Target Column', 'Model Type', 'Performance', 'Training Date']],
            hide_index=True,
            use_container_width=True
        )
        
        # Show total count
        st.info(f"Total trained models: {len(summary)}")

    def delete_trained_file_record(self, file_hash: str) -> bool:
        """Delete a trained file record"""
        try:
            if file_hash in self.file_metadata:
                # Get model info before deletion
                model_info = self.file_metadata[file_hash]
                
                # Remove from metadata
                del self.file_metadata[file_hash]
                self._save_file_metadata()
                
                # Optionally remove model file
                model_path = model_info.get('model_path', '')
                if os.path.exists(model_path):
                    os.remove(model_path)
                    self._log_info(f"Removed model file: {model_path}")
                
                self._log_info(f"Deleted trained file record: {model_info['filename']}")
                return True
            return False
        except Exception as e:
            self._log_error(f"Error deleting trained file record: {str(e)}")
            return False

    def train_model(self, df: pd.DataFrame, target_col: str, domain: str = None) -> Dict[str, Any]:
        """Train model for given dataframe and target column"""
        try:
            if domain is None:
                domain = "general"
            
            self._log_info(f"Starting training for {domain} domain with target column: {target_col}")
            self._log_info(f"Dataset shape: {df.shape}")
            self._log_info(f"Available columns: {df.columns.tolist()}")
            
            # Validate target column exists
            if target_col not in df.columns:
                available_cols = df.columns.tolist()
                self._log_error(f"Target column '{target_col}' not found. Available columns: {available_cols}")
                return {
                    'error': f"Target column '{target_col}' not found in dataset",
                    'available_columns': available_cols
                }
            
            # Store data context for Q&A
            self.data_context = {
                'dataframe': df.copy(),
                'target_column': target_col,
                'domain': domain
            }
            
            # Check if model exists for this domain
            if self._domain_model_exists(domain):
                self._log_info(f"Existing model found for {domain} domain. Retraining...")
                return self._retrain_domain_model(df, target_col, domain)
            else:
                self._log_info(f"No existing model for {domain} domain. Training new model...")
                return self._train_new_domain_model(df, target_col, domain)
                
        except Exception as e:
            self._log_error(f"Error in train_model: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _load_data(self, uploaded_file) -> pd.DataFrame:
        """Load data from various file formats"""
        try:
            filename = getattr(uploaded_file, 'name', 'unknown')
            self._log_info(f"Loading file: {filename}")
            
            if filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif filename.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)  # Default to CSV
            
            self._log_info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {filename}")
            return df
        except Exception as e:
            self._log_error(f"Error loading file: {str(e)}")
            return pd.DataFrame()

    def _detect_target_column(self, df: pd.DataFrame, domain: str) -> str:
        """Auto-detect target column based on domain and column names"""
        target_keywords = {
            'education': ['grade', 'score', 'result', 'performance', 'pass', 'fail', 'marks', 'percentage', 'total'],
            'healthcare': ['diagnosis', 'outcome', 'status', 'result', 'risk'],
            'finance': ['price', 'amount', 'profit', 'loss', 'default', 'fraud'],
            'sales': ['sales', 'revenue', 'profit', 'quantity', 'conversion'],
            'hr': ['salary', 'performance', 'rating', 'promotion', 'retention'],
            'generic': ['target', 'label', 'y', 'output', 'result']
        }
        
        keywords = target_keywords.get(domain.lower(), target_keywords['generic'])
        
        # Check for exact matches first
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in keywords):
                self._log_info(f"Found target column '{col}' based on keyword matching")
                return col
        
        # For education domain, specifically look for numeric columns that could be targets
        if domain.lower() == 'education':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['total', 'percentage', 'score', 'marks']):
                    self._log_info(f"Found education target column '{col}' from numeric columns")
                    return col
        
        # Fallback: last numeric column or last column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target = numeric_cols[-1]
            self._log_info(f"Using last numeric column as target: {target}")
            return target
        
        # Ultimate fallback: last column
        target = df.columns[-1]
        self._log_info(f"Using last column as target: {target}")
        return target

    def _domain_model_exists(self, domain: str) -> bool:
        """Check if model already exists for the domain"""
        model_path = f'models/{domain}_model.pkl'
        exists = os.path.exists(model_path)
        self._log_info(f"Model exists for {domain}: {exists}")
        return exists

    def _retrain_domain_model(self, df: pd.DataFrame, target_col: str, domain: str) -> Dict[str, Any]:
        """Retrain existing domain model with new data"""
        try:
            self._log_info(f"Retraining model for domain: {domain}")
            
            # Load existing model package
            self._load_model_package(domain)
            existing_package = self.trained_models[domain]
            
            # Use same features as existing model if possible
            existing_features = existing_package['features']
            available_features = [col for col in existing_features if col in df.columns and col != target_col]
            
            if len(available_features) < len(existing_features) * 0.7:  # If less than 70% features match
                self._log_warning(f"Feature mismatch for {domain}. Creating new model.")
                return self._train_new_domain_model(df, target_col, domain)
            
            # Retrain with existing architecture
            X, y = self._prepare_data_for_retraining(df, target_col, existing_package)
            if X.empty:
                return {'error': 'No valid data after preprocessing'}

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
            self._log_info(f"Successfully retrained {model_name} for {domain} domain with score: {score:.4f}")
            return results
            
        except Exception as e:
            self._log_error(f"Error retraining model for {domain}: {str(e)}", exc_info=True)
            return self._train_new_domain_model(df, target_col, domain)

    def _train_new_domain_model(self, df: pd.DataFrame, target_col: str, domain: str) -> Dict[str, Any]:
        """Train new model for domain"""
        try:
            self._log_info(f"Training new model for domain: {domain}, target: {target_col}")
            results = self._train_and_save_models(df, target_col, domain)
            if results:
                self._log_info(f"Successfully created new model for {domain} domain")
            return results
        except Exception as e:
            self._log_error(f"Error training new model: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _prepare_data_for_retraining(self, df: pd.DataFrame, target_col: str, existing_package: Dict[str, Any]):
        """Prepare data for retraining using existing preprocessing"""
        try:
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
        except Exception as e:
            self._log_error(f"Error preparing data for retraining: {str(e)}")
            return pd.DataFrame(), pd.Series()

    def _train_and_save_models(self, df: pd.DataFrame, target_col: str, domain: str) -> Dict[str, Any]:
        """Train multiple models and save the best one"""
        try:
            X, y = self._prepare_data(df, target_col)
            if X.empty or y.empty:
                return {'error': 'No valid data after preprocessing'}

            is_classification = self._is_classification(y)
            self._log_info(f"Task type: {'Classification' if is_classification else 'Regression'}")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            models = self._get_models(is_classification)
            results = {}

            for name, model in models.items():
                try:
                    self._log_info(f"Training {name}...")
                    
                    # Train model
                    X_tr = X_train_scaled if 'Linear' in name or 'Logistic' in name else X_train
                    X_te = X_test_scaled if 'Linear' in name or 'Logistic' in name else X_test
                    
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_te)

                    # Calculate metrics
                    if is_classification:
                        score = accuracy_score(y_test, y_pred)
                        results[name] = {'model': model, 'score': score, 'type': 'classification'}
                        self._log_info(f"{name} accuracy: {score:.4f}")
                    else:
                        score = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        results[name] = {'model': model, 'score': score, 'rmse': rmse, 'type': 'regression'}
                        self._log_info(f"{name} R² score: {score:.4f}, RMSE: {rmse:.4f}")

                except Exception as e:
                    self._log_error(f"Training error for {name}: {str(e)}")

            # Save best model
            if results:
                best_model_name = max(results.keys(), key=lambda k: results[k]['score'])
                self._log_info(f"Best model: {best_model_name} with score: {results[best_model_name]['score']:.4f}")
                
                self._save_model_package(results[best_model_name]['model'], domain, best_model_name, 
                                       X.columns.tolist(), target_col, is_classification)
                
            self.model_results = results
            return results
        except Exception as e:
            self._log_error(f"Error in train_and_save_models: {str(e)}", exc_info=True)
            return {'error': str(e)}

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
            self._log_error(f"Error getting domain model info: {str(e)}")
            return {'exists': False}

    def _prepare_data(self, df: pd.DataFrame, target_col: str):
        """Prepare features and target, handle categorical variables"""
        try:
            self._log_info(f"Preparing data with target column: {target_col}")
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            self._log_info(f"Features: {len(feature_cols)} columns")
            self._log_info(f"Target: {target_col}")

            # Handle categorical variables in features
            for col in X.columns:
                if X[col].dtype == 'object

  
            
    def _prepare_data(self, df: pd.DataFrame, target_col: str):
        """Prepare features and target, handle categorical variables"""
        try:
            self._log_info(f"Preparing data with target column: {target_col}")
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            self._log_info(f"Features: {len(feature_cols)} columns")
            self._log_info(f"Target: {target_col}")

            # Handle categorical variables in features
            for col in X.columns:
                if X[col].dtype == 'object':
                    self._log_info(f"Encoding categorical feature: {col}")
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str).fillna('missing'))
                    self.label_encoders[col] = le

            # Handle target if categorical
            if y.dtype == 'object':
                self._log_info(f"Encoding categorical target: {target_col}")
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str).fillna('missing'))
                self.label_encoders[target_col] = le

            # Handle missing values
            X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
            if pd.isna(y).any():
                if y.dtype in [np.number]:
                    y = y.fillna(y.mean())
                else:
                    y = y.fillna(0)

            self._log_info(f"Final data shape - X: {X.shape}, y: {y.shape}")
            return X, y
            
        except Exception as e:
            self._log_error(f"Error preparing data: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.Series()

    def _is_classification(self, y) -> bool:
        """Determine if task is classification"""
        unique_values = len(np.unique(y))
        is_classification = unique_values <= 10
        self._log_info(f"Unique target values: {unique_values}, Classification: {is_classification}")
        return is_classification

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
        try:
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
            self._log_info(f"Saved {model_name} model for {domain} domain at {model_path}")
        except Exception as e:
            self._log_error(f"Error saving model package: {str(e)}")

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
            
            self._log_info(f"Q&A - Domain: {domain}, Has Answer: {has_answer}")
            return qa_entry

        except Exception as e:
            self._log_error(f"Error answering question: {str(e)}")
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

    def _find_relevant_column(self, question: str, columns) -> Optional[str]:
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
            self._log_info(f"Loaded model package for {domain}")

    def _save_qa_log(self):
        """Save Q&A log to file"""
        try:
            log_path = 'logs/qa_log.json'
            with open(log_path, 'w') as f:
                json.dump(self.qa_log, f, indent=2)
        except Exception as e:
            self._log_error(f"Error saving Q&A log: {str(e)}")

    def predict(self, input_data: Dict[str, Any], domain: str) -> Any:
        """Make prediction with trained model"""
        try:
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
        except Exception as e:
            self._log_error(f"Error making prediction: {str(e)}")
            return None

    def display_results(self):
        """Display model results in Streamlit"""
        if not self.model_results:
            st.error("No model results to display.")
            return
        
        if 'error' in self.model_results:
            st.error(f"Training Error: {self.model_results['error']}")
            return
        
        results_data = []
        for name, result in self.model_results.items():
            if result['type'] == 'classification':
                results_data.append({'Model': name, 'Accuracy': f"{result['score']:.2%}"})
            else:
                results_data.append({'Model': name, 'R² Score': f"{result['score']:.3f}"})
        
        st.subheader("Model Performance")
        st.dataframe(pd.DataFrame(results_data))
        
        if results_data:
            best_model = max(self.model_results.keys(), key=lambda k: self.model_results[k]['score'])
            st.success(f"Best Model: {best_model}")

        
