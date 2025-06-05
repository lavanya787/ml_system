# src/utils/query_handler.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import os
import uuid
import re
import traceback


class QueryHandler:    
    INTENT_KEYWORDS = {
        'details': ['details', 'show', 'display', 'find', 'get', 'record', 'of', 'what', 'physics', 'chemistry', 'computer', 'mark'],
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
            def log_info(self, msg): 
                print(f"INFO: {msg}")
                return True
            def log_warning(self, msg): 
                print(f"WARNING: {msg}")
                return True
            def log_error(self, msg): 
                print(f"ERROR: {msg}")
                return True
        return SimpleLogger()
    
    def classify_intent(self, query: str) -> tuple:
        """Classify query intent using keywords."""
        print(f"DEBUG: Classifying intent for query: '{query}'")
        query_lower = query.lower()
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            matching_keywords = [kw for kw in keywords if kw in query_lower]
            if matching_keywords:
                confidence = 0.8
                print(f"DEBUG: Intent '{intent}' matched with keywords: {matching_keywords}")
                return intent, confidence
        
        print("DEBUG: No specific intent matched, returning 'general'")
        return 'general', 0.5
    
    def detect_id_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the most likely ID column in the dataset."""
        print(f"DEBUG: Detecting ID column from columns: {list(data.columns)}")
        
        if data is None or data.empty:
            print("DEBUG: Data is None or empty")
            return None
            
        text_cols = data.select_dtypes(include='object').columns.tolist()
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        all_cols = list(data.columns)
        
        print(f"DEBUG: Text columns: {text_cols}")
        print(f"DEBUG: Numeric columns: {numeric_cols}")
        print(f"DEBUG: All columns: {all_cols}")
        
        priority_keywords = ['id', 'name', 'identifier', 'code', 'regno', 'roll', 'student', 'stu']
    
        # Check all columns (not just text) for ID patterns
        for keyword in priority_keywords:
            for col in all_cols:
                if keyword in col.lower():
                    print(f"DEBUG: Found ID column '{col}' matching keyword '{keyword}'")
                    return col
        
        # If no keyword match, return the first column that looks like an ID
        for col in all_cols:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['id', 'key', 'index']):
                print(f"DEBUG: Found ID-like column: {col}")
                return col
        
        # Fallback to first text column, then first column
        if text_cols:
            print(f"DEBUG: Using first text column as ID: {text_cols[0]}")
            return text_cols[0]
        elif all_cols:
            print(f"DEBUG: Using first column as ID: {all_cols[0]}")
            return all_cols[0]
        
        print("DEBUG: No suitable ID column found")
        return None
    
    def handle_query(self, user_id: str, query: str, domain: str = None, data: pd.DataFrame = None, 
                    raw_data: pd.DataFrame = None, processed_data: pd.DataFrame = None, 
                    model_handler=None, **kwargs) -> dict:
        """Handle incoming queries with comprehensive debugging."""
        
        print(f"\n{'='*50}")
        print(f"DEBUG: Starting query processing")
        print(f"DEBUG: User ID: {user_id}")
        print(f"DEBUG: Query: '{query}'")
        print(f"DEBUG: Domain: {domain}")
        print(f"DEBUG: Data provided: {data is not None}")
        print(f"DEBUG: Raw data provided: {raw_data is not None}")
        print(f"DEBUG: Processed data provided: {processed_data is not None}")
        print(f"DEBUG: Model handler provided: {model_handler is not None}")
        print(f"DEBUG: Additional kwargs: {list(kwargs.keys())}")
        
        try:
            # Use processed_data if available, otherwise fall back to raw_data, then data
            if processed_data is not None:
                data = processed_data
                print("DEBUG: Using processed_data")
            elif raw_data is not None:
                data = raw_data
                print("DEBUG: Using raw_data")
            elif data is not None:
                data = data
                print("DEBUG: Using data")
            else:
                data = None
                print("DEBUG: No data available")
                
            if data is not None:
                print(f"DEBUG: Data shape: {data.shape}")
                print(f"DEBUG: Data columns: {list(data.columns)}")
                print(f"DEBUG: First few rows:")
                print(data.head().to_string())
            
            intent, confidence = self.classify_intent(query)
            query_id = str(uuid.uuid4())
            
            print(f"DEBUG: Classified intent: {intent} (confidence: {confidence})")
            print(f"DEBUG: Generated query ID: {query_id}")

            # Handle specific record lookup (e.g., "Show details of STU0001")
            if intent == 'details' and data is not None:
                print("DEBUG: Processing details query")
                result = self._handle_details_query(user_id, query, data, query_id, intent, confidence)
                print(f"DEBUG: Details query result: {result}")
                return result

            # Handle prediction with model_handler if available
            elif intent == 'prediction':
                print("DEBUG: Processing prediction query")
                if model_handler is not None:
                    try:
                        prediction_result = self._handle_prediction_with_model(query, data, model_handler)
                        print(f"DEBUG: Prediction result: {prediction_result}")
                        return prediction_result
                    except Exception as e:
                        error_msg = f"Prediction with model failed: {e}"
                        print(f"DEBUG: {error_msg}")
                        self.logger.log_error(error_msg)
                        return {
                            'type': 'prediction',
                            'message': f'Prediction failed: {str(e)}\nPlease check your data and model configuration.',
                            'success': False
                        }
                else:
                    result = {
                        'type': 'prediction',
                        'message': 'Prediction functionality requires a model handler.\nThis feature will analyze data patterns to make forecasts.',
                        'success': False
                    }
                    print(f"DEBUG: Prediction result: {result}")
                    return result

            elif intent == 'performance':
                print("DEBUG: Processing performance query")
                if model_handler is not None:
                    try:
                        performance_result = self._handle_performance_with_model(query, data, model_handler)
                        print(f"DEBUG: Performance result: {performance_result}")
                        return performance_result
                    except Exception as e:
                        error_msg = f"Performance analysis failed: {e}"
                        print(f"DEBUG: {error_msg}")
                        self.logger.log_error(error_msg)
                        return {
                            'type': 'performance',
                            'message': f'Performance analysis failed: {str(e)}',
                            'success': False
                        }
                else:
                    result = {
                        'type': 'performance',
                        'message': 'Performance analysis functionality requires a model handler.\nThis feature will evaluate model metrics and accuracy.',
                        'success': False
                    }
                    print(f"DEBUG: Performance result: {result}")
                    return result

            elif intent == 'risk':
                result = {
                    'type': 'risk',
                    'message': 'Risk assessment functionality is not yet available.\nThis feature will identify potential risks and mitigation strategies.',
                    'success': False
                }
                print(f"DEBUG: Risk result: {result}")
                return result

            elif intent == 'history':
                result = {
                    'type': 'history',
                    'message': 'Query history functionality is not yet available.\nThis feature will show your previous queries and results.',
                    'success': False
                }
                print(f"DEBUG: History result: {result}")
                return result

            elif intent == 'summary' and data is not None:
                print("DEBUG: Processing summary query")
                try:
                    summary_text = self._generate_summary(data)
                    result = {
                        'type': 'summary',
                        'message': summary_text,
                        'success': True
                    }
                    print(f"DEBUG: Summary result: {result}")
                    return result
                except Exception as e:
                    error_msg = f"Summary generation failed: {e}"
                    print(f"DEBUG: {error_msg}")
                    self.logger.log_error(error_msg)
                    return {
                        'type': 'summary',
                        'message': f'Failed to generate summary: {str(e)}',
                        'success': False
                    }

            # Fallback for unknown intent
            else:
                print("DEBUG: Processing fallback/unknown intent")
                available_features = [
                    "- Show details of a record (e.g., \"Show details of STU0001\")",
                    "- Generate data summary"
                ]
                
                if model_handler is not None:
                    available_features.extend([
                        "- Make predictions",
                        "- Analyze model performance"
                    ])
                
                available_features.append("- Other data queries")
                
                result = {
                    'type': 'unknown',
                    'message': 'I can help you with:\n' + '\n'.join(available_features),
                    'success': True
                }
                print(f"DEBUG: Fallback result: {result}")
                return result
                
        except Exception as e:
            error_msg = f"Unexpected error in handle_query: {e}"
            print(f"DEBUG: {error_msg}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            self.logger.log_error(error_msg)
            return {
                'type': 'error',
                'message': f'An unexpected error occurred: {str(e)}',
                'success': False
            }

    def _handle_details_query(self, user_id: str, query: str, data: pd.DataFrame, query_id: str, intent: str, confidence: float) -> dict:
        """Handle queries asking for details of specific records."""
        print(f"DEBUG: Handling details query: '{query}'")
        
        try:
            # Extract identifier from query with more flexible patterns
            patterns = [
                r'\b[A-Z]{2,}[0-9]+\b',  # STU0001, ABC123
                r'\b[0-9]+[A-Z]{2,}\b',  # 123ABC
                r'\b[A-Z0-9]{4,}\b',     # ABCD1234
                r'\bstu\d+\b',           # stu0004 (case insensitive)
                r'\b\d+\b'               # Just numbers
            ]
            
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, query.upper())
                if found:
                    matches.extend(found)
                    break
            
            # Also try case-insensitive search for student IDs
            if not matches:
                stu_pattern = re.findall(r'\bstu\d+\b', query.lower())
                if stu_pattern:
                    matches = [match.upper() for match in stu_pattern]
            
            print(f"DEBUG: Extracted potential IDs: {matches}")
            
            if not matches:
                # Try to find any potential identifier in the query
                words = query.split()
                potential_ids = [word for word in words if len(word) > 2 and any(c.isalnum() for c in word)]
                matches = potential_ids[:1]
                print(f"DEBUG: Fallback potential IDs: {matches}")
            
            if matches:
                search_id = matches[0].strip()
                print(f"DEBUG: Searching for ID: '{search_id}'")
                
                id_column = self.detect_id_column(data)
                print(f"DEBUG: Detected ID column: '{id_column}'")
                
                if id_column:
                    # Search for the record (case-insensitive)
                    print(f"DEBUG: Searching in column '{id_column}'")
                    print(f"DEBUG: Column values: {list(data[id_column].astype(str))}")
                    
                    # Try multiple matching strategies
                    matching_records = pd.DataFrame()
                    
                    # Strategy 1: Exact case-insensitive match
                    matching_records = data[data[id_column].astype(str).str.upper() == search_id.upper()]
                    print(f"DEBUG: Exact match found {len(matching_records)} records")
                    
                    # Strategy 2: Contains match if exact doesn't work
                    if matching_records.empty:
                        matching_records = data[data[id_column].astype(str).str.upper().str.contains(search_id.upper(), na=False)]
                        print(f"DEBUG: Contains match found {len(matching_records)} records")
                    
                    # Strategy 3: Try removing leading zeros or prefixes
                    if matching_records.empty:
                        # Extract just numbers from search_id
                        search_numbers = re.findall(r'\d+', search_id)
                        if search_numbers:
                            search_num = search_numbers[0]
                            matching_records = data[data[id_column].astype(str).str.contains(search_num, na=False)]
                            print(f"DEBUG: Number-based match found {len(matching_records)} records")
                    
                    if not matching_records.empty:
                        record = matching_records.iloc[0]
                        print(f"DEBUG: Found matching record")
                        
                        # Check if user asked for specific subjects
                        query_lower = query.lower()
                        requested_subjects = []
                        
                        if 'physics' in query_lower:
                            requested_subjects.append('physics')
                        if 'chemistry' in query_lower:
                            requested_subjects.append('chemistry')
                        if 'computer' in query_lower:
                            requested_subjects.extend(['computer', 'cs', 'computerscience'])
                        
                        # Format record details as readable text
                        if requested_subjects:
                            # Show only requested subjects
                            details_text = f"Subject marks for {search_id}:\n"
                            details_text += "-" * 30 + "\n"
                            
                            found_subjects = False
                            for column, value in record.items():
                                col_lower = column.lower()
                                if any(subj in col_lower for subj in requested_subjects):
                                    details_text += f"{column}: {value}\n"
                                    found_subjects = True
                            
                            if not found_subjects:
                                details_text += "Requested subjects not found in data.\n"
                                details_text += "Available columns:\n"
                                for column, value in record.items():
                                    details_text += f"{column}: {value}\n"
                        else:
                            # Show all details
                            details_text = f"Details for {search_id}:\n"
                            details_text += "-" * 30 + "\n"
                            for column, value in record.items():
                                details_text += f"{column}: {value}\n"
                        
                        result = {
                            'type': 'details',
                            'message': details_text.strip(),
                            'success': True
                        }
                        print(f"DEBUG: Returning successful result")
                        return result
                    else:
                        result = {
                            'type': 'details',
                            'message': f'No record found for {search_id}.\nAvailable IDs: {list(data[id_column].astype(str)[:5])}...',
                            'success': False
                        }
                        print(f"DEBUG: No matching records found")
                        return result
                else:
                    result = {
                        'type': 'details',
                        'message': 'Could not identify the ID column in the data.\nPlease ensure the data has a proper ID field.',
                        'success': False
                    }
                    print(f"DEBUG: No ID column detected")
                    return result
            else:
                result = {
                    'type': 'details',
                    'message': 'Could not find a valid ID in your query.\nPlease specify a record ID (e.g., "Show details of STU0001").',
                    'success': False
                }
                print(f"DEBUG: No ID extracted from query")
                return result
            
        except Exception as e:
            error_msg = f"Error handling details query: {e}"
            print(f"DEBUG: {error_msg}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            self.logger.log_error(error_msg)
            return {
                'type': 'details',
                'message': f'An error occurred while processing your request:\n{str(e)}',
                'success': False
            }

    def _handle_prediction_with_model(self, query: str, data: pd.DataFrame, model_handler) -> dict:
        """Handle prediction queries using the model handler."""
        print("DEBUG: Handling prediction with model")
        try:
            if hasattr(model_handler, 'predict'):
                if data is not None and not data.empty:
                    predictions = model_handler.predict(data)
                    return {
                        'type': 'prediction',
                        'message': f'Prediction completed successfully.\nResults: {predictions}',
                        'success': True,
                        'data': predictions
                    }
                else:
                    return {
                        'type': 'prediction',
                        'message': 'No data available for prediction.',
                        'success': False
                    }
            else:
                return {
                    'type': 'prediction',
                    'message': 'Model handler does not support predictions.',
                    'success': False
                }
        except Exception as e:
            self.logger.log_error(f"Prediction error: {e}")
            return {
                'type': 'prediction',
                'message': f'Prediction failed: {str(e)}',
                'success': False
            }

    def _handle_performance_with_model(self, query: str, data: pd.DataFrame, model_handler) -> dict:
        """Handle performance analysis queries using the model handler."""
        print("DEBUG: Handling performance with model")
        try:
            if hasattr(model_handler, 'get_metrics'):
                metrics = model_handler.get_metrics()
                metrics_text = "Model Performance Metrics:\n" + "-" * 30 + "\n"
                for metric, value in metrics.items():
                    metrics_text += f"{metric}: {value}\n"
                
                return {
                    'type': 'performance',
                    'message': metrics_text.strip(),
                    'success': True,
                    'data': metrics
                }
            elif hasattr(model_handler, 'evaluate'):
                if data is not None:
                    evaluation = model_handler.evaluate(data)
                    return {
                        'type': 'performance',
                        'message': f'Model evaluation completed.\nResults: {evaluation}',
                        'success': True,
                        'data': evaluation
                    }
                else:
                    return {
                        'type': 'performance',
                        'message': 'No data available for model evaluation.',
                        'success': False
                    }
            else:
                return {
                    'type': 'performance',
                    'message': 'Model handler does not support performance analysis.',
                    'success': False
                }
        except Exception as e:
            self.logger.log_error(f"Performance analysis error: {e}")
            return {
                'type': 'performance',
                'message': f'Performance analysis failed: {str(e)}',
                'success': False
            }

    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate a basic summary of the dataset."""
        print("DEBUG: Generating summary")
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
                for col in categorical_cols[:3]:
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
        print(f"DEBUG: Generating suggestions for query: '{query}'")
        
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
            final_suggestions = list(dict.fromkeys(suggestions))[:3]
            print(f"DEBUG: Generated suggestions: {final_suggestions}")
            return final_suggestions

        except Exception as e:
            error_msg = f"Error generating suggestions: {e}"
            print(f"DEBUG: {error_msg}")
            self.logger.log_error(error_msg)
            return ["Generate dataset summary", "Show record details", "Calculate statistics"]


# Test function to help debug
def test_query_handler():
    """Test function to verify QueryHandler works correctly."""
    print("Starting QueryHandler test...")
    
    handler = QueryHandler()
    
    # Create sample data that matches your use case
    sample_data = pd.DataFrame({
        'student_id': ['STU0001', 'STU0002', 'STU0003', 'STU0004'],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
        'physics': [85, 92, 78, 88],
        'chemistry': [88, 85, 82, 90],
        'computer': [90, 88, 85, 92]
    })
    
    print("Sample data:")
    print(sample_data.to_string())
    print()
    
    # Test queries
    test_queries = [
        "Show details of STU0001",
        "What is the physics, chemistry, computer mark of stu0004?",
        "show details of STU0004"
    ]
    
    for query in test_queries:
        print(f"Testing query: '{query}'")
        result = handler.handle_query(
            user_id="test_user",
            query=query,
            data=sample_data
        )
        print(f"Result: {result}")
        print(f"Message: {result.get('message', 'No message')}")
        print("-" * 80)


if __name__ == "__main__":
    test_query_handler()