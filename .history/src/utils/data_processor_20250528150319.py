import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from utils.logger import Logger

class DataProcessor:
    def __init__(self, logger: Logger):
        self.logger = logger
        
    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from CSV, Excel, or JSON file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                self.logger.log_error(f"Unsupported file format: {file_path}")
                return None
            self.logger.log_info(f"Loaded data from {file_path} with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.log_error(f"Failed to load data from {file_path}: {str(e)}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and data types."""
        try:
            df = df.drop_duplicates()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna('Unknown')
            for col in df.columns:
                if 'date' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception as e:
                        self.logger.log_warning(f"Could not convert {col} to datetime: {str(e)}")
            self.logger.log_info(f"Cleaned data: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.log_error(f"Data cleaning failed: {str(e)}")
            return df

    def process_data(self, df: pd.DataFrame, domain_config: Any) -> Dict[str, Any]:
        """Process data with domain-specific feature engineering."""
        try:
            cleaned_df = self.clean_data(df)
            detection_result = domain_config.detect_domain(cleaned_df)
            if not detection_result.get('is_domain', False):
                self.logger.log_warning(f"Domain detection confidence too low: {detection_result.get('confidence', 'N/A')}")
                return {'error': 'Domain not confidently detected', 'data': None}
            engineered_df = domain_config.engineer_features(cleaned_df, detection_result)
            target_col = domain_config.identify_target(engineered_df, detection_result)
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
        def process_domain_data(self, df: pd.DataFrame, domain_config: Any) -> pd.DataFrame:
    result = self.process_data(df, domain_config)
    if 'data' in result and result['data'] is not None:
        return result['data']
    else:
        return df  # fallback to raw data if processing failed

    def save_data(self, df: pd.DataFrame, file_path: str) -> bool:  
        """Save DataFrame to CSV, Excel, or JSON file."""
        try:
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif file_path.endswith(('.xlsx', '.xls')):
                df.to_excel(file_path, index=False)
            elif file_path.endswith('.json'):
                df.to_json(file_path, orient='records', lines=True)
            else:
                self.logger.log_error(f"Unsupported file format for saving: {file_path}")
                return False
            self.logger.log_info(f"Saved data to {file_path} with {len(df)} rows and {len(df.columns)} columns")
            return True
        except Exception as e:
            self.logger.log_error(f"Failed to save data to {file_path}: {str(e)}")
            return False