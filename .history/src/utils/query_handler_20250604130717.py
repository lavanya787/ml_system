# src/utils/query_handler.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import os
import uuid
import re


class QueryHandler:    
    INTENT_KEYWORDS = {
        'details': ['details', 'show', 'display', 'find', 'get', 'record', 'of'],
        'prediction': ['predict', 'forecast', 'future', 'estimate', 'projection'],
        'performance': ['performance', 'accuracy', 'metrics', 'evaluate', 'score'],
        'risk': ['risk', 'danger', 'threat', 'hazard', 'vulnerability'],
        'history': ['history', 'past', 'previous', 'list', 'search'],
        'summary': ['summary', 'summarize', 'overview', 'highlight', 'key points']
    }
    
    def __init__(self, logger=None):
        self.logger = logger or self._create_simple_logger()
        
    def _create_simple_logger(self):
        """Create simple logger if none provided."""
        class SimpleLogger:
            def log_info(self, msg): print(f"INFO: {msg}")
            def log_warning(self, msg): print(f"WARNING: {msg}")
            def log_error(self, msg): print(f"ERROR: {msg}")
        return SimpleLogger()
    
    def classify_intent(self, query: str) -> tuple:
        """Classify query intent using keywords."""
        query_lower = query.lower()
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                confidence = 0.8
                return intent, confidence
        
        return 'general', 0.5
    
    def detect_id_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the most likely ID column in the dataset."""
        text_cols = data.select_dtypes(include='object').columns.tolist()
        priority_keywords = ['id', 'name', 'identifier', 'code', 'regno', 'roll', 'student']
    
        for keyword in priority_keywords:
            for col in text_cols:
                if keyword in col.lower():
                    return col
        return text_cols[0] if text_cols else None
    
    def handle_query(self, user_id: str, query: str, domain: str = None, data: pd.DataFrame = None, raw_data: pd.DataFrame = None, processed_data: pd.DataFrame = None) -> dict:
        """Handle incoming queries."""
        # Use processed_data if available, otherwise fall back to raw_data, then data
        if processed_data is not None:
            data = processed_data
        elif raw_data is not None:
            data = raw_data
        elif data is not None:
            data = data
        else:
            data = None
            
        intent, confidence = self.classify_intent(query)
        query_id = str(uuid.uuid4())

        # Handle specific record lookup (e.g., "Show details of STU0001")
        if intent == 'details' and data is not None:
            return self._handle_details_query(user_id, query, data, query_id, intent, confidence)

        # Handle other intents with placeholder responses
        elif intent == 'prediction':
            return {
                'type': 'prediction',
                'message': 'Prediction functionality is not yet available.\nThis feature will analyze data patterns to make forecasts.',
                'success': False
            }

        elif intent == 'performance':
            return {
                'type': 'performance',
                'message': 'Performance analysis functionality is not yet available.\nThis feature will evaluate model metrics and accuracy.',
                'success': False
            }

        elif intent == 'risk':
            return {
                'type': 'risk',
                'message': 'Risk assessment functionality is not yet available.\nThis feature will identify potential risks and mitigation strategies.',
                'success': False
            }

        elif intent == 'history':
            return {
                'type': 'history',
                'message': 'Query history functionality is not yet available.\nThis feature will show your previous queries and results.',
                'success': False
            }

        elif intent == 'summary' and data is not None:
            try:
                summary_text = self._generate_summary(data)
                return {
                    'type': 'summary',
                    'message': summary_text,
                    'success': True
                }
            except Exception as e:
                self.logger.log_error(f"Summary generation failed: {e}")
                return {
                    'type': 'summary',
                    'message': f'Failed to generate summary: {str(e)}',
                    'success': False
                }

        # Fallback for unknown intent
        else:
            return {
                'type': 'unknown',
                'message': 'I can help you with:\n- Show details of a record (e.g., "Show details of STU0001")\n- Generate data summary\n- Other data queries',
                'success': True
            }

    def _handle_details_query(self, user_id: str, query: str, data: pd.DataFrame, query_id: str, intent: str, confidence: float) -> dict:
        """Handle queries asking for details of specific records."""
        try:
            # Extract identifier from query (e.g., "STU0001" from "Show details of STU0001")
            matches = re.findall(r'\b[A-Z]{2,}[0-9]+\b|\b[0-9]+[A-Z]{2,}\b|\b[A-Z0-9]{4,}\b', query.upper())
            
            if not matches:
                # Try to find any potential identifier
                words = query.split()
                potential_ids = [word for word in words if len(word) > 2 and word.replace(' ', '').isalnum()]
                matches = potential_ids[:1]
            
            if matches:
                search_id = matches[0].strip()
                id_column = self.detect_id_column(data)
                
                if id_column:
                    # Search for the record (case-insensitive)
                    matching_records = data[data[id_column].astype(str).str.upper() == search_id.upper()]
                    
                    if not matching_records.empty:
                        record = matching_records.iloc[0]
                        # Format record details as readable text
                        details_text = f"Details for {search_id}:\n"
                        details_text += "-" * 30 + "\n"
                        for column, value in record.items():
                            details_text += f"{column}: {value}\n"
                        
                        return {
                            'type': 'details',
                            'message': details_text.strip(),
                            'success': True
                        }
                    else:
                        return {
                            'type': 'details',
                            'message': f'No record found for {search_id}.\nPlease check the ID and try again.',
                            'success': False
                        }
                else:
                    return {
                        'type': 'details',
                        'message': 'Could not identify the ID column in the data.\nPlease ensure the data has a proper ID field.',
                        'success': False
                    }
            else:
                return {
                    'type': 'details',
                    'message': 'Could not find a valid ID in your query.\nPlease specify a record ID (e.g., "Show details of STU0001").',
                    'success': False
                }
            
        except Exception as e:
            self.logger.log_error(f"Error handling details query: {e}")
            return {
                'type': 'details',
                'message': f'An error occurred while processing your request:\n{str(e)}',
                'success': False
            }

    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate a basic summary of the dataset."""
        try:
            summary_lines = [
                f"📊 Dataset Overview",
                f"{'='*40}",
                f"Total Records: {df.shape[0]}",
                f"Total Columns: {df.shape[1]}",
                ""
            ]

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if numeric_cols:
                summary_lines.append("🔢 Numeric Columns:")
                for col in numeric_cols:
                    stats = df[col].describe()
                    summary_lines.append(f"  • {col}: Mean={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")
                summary_lines.append("")
                
            if categorical_cols:
                summary_lines.append("📝 Text Columns:")
                for col in categorical_cols[:3]:  # Show first 3 to avoid too long output
                    unique_count = df[col].nunique()
                    summary_lines.append(f"  • {col}: {unique_count} unique values")
                summary_lines.append("")

            # Check for missing values
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            if total_nulls > 0:
                summary_lines.append("⚠️  Missing Values:")
                for col, count in null_counts[null_counts > 0].items():
                    percentage = (count / len(df)) * 100
                    summary_lines.append(f"  • {col}: {count} ({percentage:.1f}%)")
            else:
                summary_lines.append("✅ No missing values found")

            return "\n".join(summary_lines)

        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def generate_dynamic_suggestions(self, query: str, data: pd.DataFrame) -> list:
        """Generate dynamic suggestions based on query and data."""
        if data is None or data.empty:
            return ["No data available for suggestions"]
            
        try:
            query_lower = query.lower()
            suggestions = []

            text_cols = data.select_dtypes(include='object').columns.tolist()
            numeric_cols = data.select_dtypes(include='number').columns.tolist()

            # Suggestions for top/highest/ranking queries
            if any(k in query_lower for k in ["top", "highest", "rank", "score", "best"]):
                for col in numeric_cols[:2]:
                    suggestions.append(f"Show top 5 records by {col}")

            # Suggestions for average/mean/total queries
            if any(k in query_lower for k in ["average", "mean", "total", "sum"]):
                for col in numeric_cols[:2]:
                    suggestions.append(f"Calculate average of {col}")
                if text_cols and numeric_cols:
                    suggestions.append(f"Average {numeric_cols[0]} by {text_cols[0]}")

            # Suggestions for detail/record queries
            if any(k in query_lower for k in ["who", "what", "value", "of", "record", "detail", "show"]):
                if text_cols:
                    # Get a sample value from the first text column
                    sample_values = data[text_cols[0]].dropna().unique()
                    if len(sample_values) > 0:
                        sample_val = str(sample_values[0])
                        suggestions.append(f"Show details of {sample_val}")

            # Default suggestions if none matched
            if not suggestions and text_cols and numeric_cols:
                suggestions += [
                    f"List all {text_cols[0]}s",
                    f"Show top 3 by {numeric_cols[0]}",
                    f"Calculate average {numeric_cols[0]}",
                    "Generate dataset summary"
                ]

            # Limit to 3 suggestions and ensure they're unique
            return list(dict.fromkeys(suggestions))[:3]

        except Exception as e:
            self.logger.log_error(f"Error generating suggestions: {e}")
            return ["Generate dataset summary", "Show record details", "Calculate statistics"]


# Example usage:
if __name__ == "__main__":
    # Test the handler
    handler = QueryHandler()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'student_id': ['STU0001', 'STU0002', 'STU0003'],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'score': [85, 92, 78]
    })
    
    # Test query
    result = handler.handle_query(
        user_id="test_user",
        query="Show details of STU0001",
        data=sample_data
    )
    
    # Test the suggestions feature
    suggestions = handler.generate_dynamic_suggestions("show details", sample_data)
    print("Suggestions:", suggestions)
    
    # Display only the message (text format for user)
    print("\nQuery Result:")
    print(result['message'])