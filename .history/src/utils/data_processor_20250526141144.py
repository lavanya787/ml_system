import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from utils.logger import Logger

class DataProcessor:
    def __init__(self, logger):
        self.logger = logge
    def process_domain_data(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        # Perform preprocessing based on domain logic (stub example)
        self.logger.log_info(f"Processing data for domain: {domain}")
        return df  # Modify this as needed
    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from CSV or Excel file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                self.logger.log_error(f"Unsupported file format: {file_path}")
                return None
            self.logger.log_info(f"Loaded data from {file_path} with {len(df)} rows")
            return df
        except Exception as e:
            self.logger.log_error(f"Failed to load data from {file_path}: {str(e)}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and data types."""
        try:
            # Remove duplicate rows
            df = df.drop_duplicates()
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna('Unknown')
            
            # Convert date columns
            for col in df.columns:
                if 'date' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        self.logger.log_warning(f"Could not convert {col} to datetime")
            
            self.logger.log_info(f"Cleaned data: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.log_error(f"Data cleaning failed: {str(e)}")
            return df
    
    def process_data(self, df: pd.DataFrame, domain_config: Any) -> Dict[str, Any]:
        """Process data with domain-specific feature engineering."""
        try:
            # Clean data
            cleaned_df = self.clean_data(df)
            
            # Detect domain
            detection_result = domain_config.detect_domain(cleaned_df)
            if not detection_result['is_domain']:
                self.logger.log_warning(f"Domain detection confidence too low: {detection_result['confidence']}")
                return {'error': 'Domain not confidently detected', 'data': None}
            
            # Engineer features
            engineered_df = domain_config.engineer_features(cleaned_df, detection_result)
            
            # Identify target
            target_col = domain_config.identify_target(cleaned_df, detection_result)
            if not target_col:
                self.logger.log_warning("No target variable identified")
                return {'error': 'No target variable found', 'data': None}
            
            self.logger.log_info(f"Processed data for domain with target: {target_col}")
            return {
                'data': engineered_df,
                'target': target_col,
                'detection_result': detection_result
            }
        except Exception as e:
            self.logger.log_error(f"Data processing failed: {str(e)}")
            return {'error': str(e), 'data': None}