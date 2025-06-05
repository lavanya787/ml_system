import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Any, Tuple

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def process_education_data(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process education dataset for ML"""
        processed_data = {
            'original_data': df.copy(),
            'processed_features': None,
            'target_variable': None,
            'feature_info': {},
            'preprocessing_steps': []
        }
        
        # Identify potential target variable
        target_col = self._identify_target_variable(df, detection_result)
        if target_col:
            processed_data['target_variable'] = target_col
            processed_data['preprocessing_steps'].append(f"Identified target variable: {target_col}")
        
        # Clean and preprocess features
        cleaned_df = self._clean_data(df)
        processed_data['preprocessing_steps'].append("Data cleaning completed")
        
        # Encode categorical variables
        encoded_df = self._encode_categorical_variables(cleaned_df)
        processed_data['preprocessing_steps'].append("Categorical encoding completed")
        
        # Feature engineering for education domain
        engineered_df = self._engineer_education_features(encoded_df, detection_result)
        processed_data['preprocessing_steps'].append("Education-specific feature engineering completed")
        
        processed_data['processed_features'] = engineered_df
        processed_data['feature_info'] = self._get_feature_info(engineered_df)
        
        return processed_data
    
    def _identify_target_variable(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify the most likely target variable for prediction"""
        target_keywords = ['grade', 'score', 'result', 'performance', 'gpa', 'cgpa', 'final']
        
        # Check for columns with target keywords
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in target_keywords):
                if df[col].dtype in ['int64', 'float64']:
                    return col
        
        # If no obvious target, look for numeric columns with reasonable range
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].max() <= 100 and df[col].min() >= 0:
                return col
        
        return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset"""
        cleaned_df = df.copy()
        
        # Handle missing values
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if len(cleaned_df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Remove duplicate rows
        cleaned_df.drop_duplicates(inplace=True)
        
        return cleaned_df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        encoded_df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            encoded_df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        return encoded_df
    
    def _engineer_education_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Create education-specific features"""
        engineered_df = df.copy()
        
        # Create performance categories if grade columns exist
        grade_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['grade', 'score', 'mark'])]
        
        for col in grade_cols:
            if df[col].dtype in ['int64', 'float64']:
                # Create performance categories
                engineered_df[f"{col}_category"] = pd.cut(
                    df[col], 
                    bins=[0, 60, 70, 80, 90, 100], 
                    labels=['F', 'D', 'C', 'B', 'A']
                )
                
                # Create pass/fail indicator
                engineered_df[f"{col}_pass_fail"] = (df[col] >= 60).astype(int)
        
        # Create attendance categories if attendance column exists
        attendance_cols = [col for col in df.columns if 'attendance' in col.lower()]
        for col in attendance_cols:
            if df[col].dtype in ['int64', 'float64']:
                engineered_df[f"{col}_category"] = pd.cut(
                    df[col],
                    bins=[0, 60, 75, 85, 95, 100],
                    labels=['Very Low', 'Low', 'Average', 'Good', 'Excellent']
                )
        
        return engineered_df
    
    def _get_feature_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get information about processed features"""
        return {
            'total_features': len(df.columns),
            'numeric_features': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(df.select_dtypes(include=['object', 'category']).columns),
            'feature_types': dict(df.dtypes)
        }