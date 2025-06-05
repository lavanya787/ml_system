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
        'details': ['details', 'show', 'display', 'find', 'get', 'record', 'of', 'what', 'info', 'information'],
        'prediction': ['predict', 'forecast', 'future', 'estimate', 'projection'],
        'performance': ['performance', 'accuracy', 'metrics', 'evaluate', 'score'],
        'risk': ['risk', 'danger', 'threat', 'hazard', 'vulnerability'],
        'history': ['history', 'past', 'previous', 'list', 'search'],
        'summary': ['summary', 'summarize', 'overview', 'highlight', 'key points'],
        'statistics': ['average', 'mean', 'total', 'sum', 'count', 'max', 'min', 'statistics', 'stats'],
        'top': ['top', 'highest', 'best', 'maximum', 'rank', 'ranking'],
        'comparison': ['compare', 'comparison', 'versus', 'vs', 'difference', 'against']
    }
    
    # Professional formatting templates
    RESPONSE_TEMPLATES = {
        'header': "╭" + "─" * 58 + "╮",
        'footer': "╰" + "─" * 58 + "╯",
        'separator': "├" + "─" * 58 + "┤",
        'divider': "│" + " " * 58 + "│"
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
        """Classify query intent using dynamic keyword matching."""
        print(f"DEBUG: Classifying intent for query: '{query}'")
        query_lower = query.lower()
        
        # Dynamic subject detection
        detected_subjects = self._detect_subjects(query)
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            matching_keywords = [kw for kw in keywords if kw in query_lower]
            if matching_keywords:
                confidence = min(0.9, 0.6 + (len(matching_keywords) * 0.1))
                print(f"DEBUG: Intent '{intent}' matched with keywords: {matching_keywords}")
                return intent, confidence, detected_subjects
        
        print("DEBUG: No specific intent matched, returning 'general'")
        return 'general', 0.5, detected_subjects
    
    def _detect_subjects(self, query: str) -> List[str]:
        """Dynamically detect subject fields from query."""
        query_lower = query.lower()
        detected = []
        
        # Common academic subjects
        subject_patterns = {
            r'\b(math|mathematics|maths)\b': 'Mathematics',
            r'\b(phys|physics)\b': 'Physics',
            r'\b(chem|chemistry)\b': 'Chemistry',
            r'\b(bio|biology)\b': 'Biology',
            r'\b(comp|computer|cs|programming)\b': 'Computer Science',
            r'\b(eng|english|literature)\b': 'English',
            r'\b(hist|history)\b': 'History',
            r'\b(geo|geography)\b': 'Geography',
            r'\b(econ|economics)\b': 'Economics'
        }
        
        for pattern, subject in subject_patterns.items():
            if re.search(pattern, query_lower):
                detected.append(subject.lower())
        
        return detected
    
    def detect_id_column(self, data: pd.DataFrame) -> Optional[str]:
        """Intelligently detect the most likely ID column."""
        print(f"DEBUG: Detecting ID column from columns: {list(data.columns)}")
        
        if data is None or data.empty:
            print("DEBUG: Data is None or empty")
            return None
            
        all_cols = list(data.columns)
        
        # Priority-based ID detection
        id_patterns = [
            r'.*id.*',
            r'.*key.*',
            r'.*code.*',
            r'.*number.*',
            r'.*no.*',
            r'.*index.*'
        ]
        
        # Check for ID patterns
        for pattern in id_patterns:
            for col in all_cols:
                if re.match(pattern, col.lower()):
                    print(f"DEBUG: Found ID column '{col}' matching pattern '{pattern}'")
                    return col
        
        # Fallback to first column
        if all_cols:
            print(f"DEBUG: Using first column as ID: {all_cols[0]}")
            return all_cols[0]
        
        print("DEBUG: No suitable ID column found")
        return None
    
    def _format_professional_response(self, title: str, content: str, response_type: str = "info") -> str:
        """Create professionally formatted response."""
        
        # Icons for different response types
        icons = {
            'success': '✅',
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'details': '📋',
            'summary': '📊',
            'prediction': '🔮',
            'performance': '📈'
        }
        
        icon = icons.get(response_type, 'ℹ️')
        
        # Create formatted response
        lines = []
        lines.append(self.RESPONSE_TEMPLATES['header'])
        lines.append(f"│ {icon} {title:<54} │")
        lines.append(self.RESPONSE_TEMPLATES['separator'])
        
        # Split content into lines and format
        content_lines = content.split('\n')
        for line in content_lines:
            if line.strip():
                # Handle long lines
                if len(line) > 56:
                    words = line.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + word) <= 56:
                            current_line += word + " "
                        else:
                            if current_line:
                                lines.append(f"│ {current_line.strip():<56} │")
                            current_line = word + " "
                    if current_line:
                        lines.append(f"│ {current_line.strip():<56} │")
                else:
                    lines.append(f"│ {line:<56} │")
            else:
                lines.append(self.RESPONSE_TEMPLATES['divider'])
        
        lines.append(self.RESPONSE_TEMPLATES['footer'])
        
        return '\n'.join(lines)
    
    def handle_query(self, user_id: str, query: str, domain: str = None, data: pd.DataFrame = None, 
                    raw_data: pd.DataFrame = None, processed_data: pd.DataFrame = None, 
                    model_handler=None, **kwargs) -> dict:
        """Handle incoming queries with professional formatting."""
        
        print(f"\n{'='*50}")
        print(f"DEBUG: Starting query processing")
        print(f"DEBUG: User ID: {user_id}")
        print(f"DEBUG: Query: '{query}'")
        
        try:
            # Use the best available data source
            working_data = processed_data if processed_data is not None else (raw_data if raw_data is not None else data)
            
            if working_data is not None:
                print(f"DEBUG: Data shape: {working_data.shape}")
                print(f"DEBUG: Data columns: {list(working_data.columns)}")
            
            intent, confidence, detected_subjects = self.classify_intent(query)
            query_id = str(uuid.uuid4())
            
            print(f"DEBUG: Classified intent: {intent} (confidence: {confidence})")
            print(f"DEBUG: Detected subjects: {detected_subjects}")

            # Route to appropriate handler
            if intent == 'details' and working_data is not None:
                return self._handle_details_query(user_id, query, working_data, detected_subjects)
            
            elif intent == 'statistics' and working_data is not None:
                return self._handle_statistics_query(query, working_data, detected_subjects)
            
            elif intent == 'top' and working_data is not None:
                return self._handle_top_query(query, working_data)
            
            elif intent == 'comparison' and working_data is not None:
                return self._handle_comparison_query(query, working_data)

            elif intent == 'prediction':
                return self._handle_prediction_query(query, working_data, model_handler)

            elif intent == 'performance':
                return self._handle_performance_query(query, working_data, model_handler)

            elif intent == 'summary' and working_data is not None:
                return self._handle_summary_query(working_data)
            
            elif intent in ['risk', 'history']:
                return self._handle_unavailable_feature(intent)

            else:
                return self._handle_general_query(working_data, model_handler)
                
        except Exception as e:
            error_msg = f"Unexpected error in query processing: {str(e)}"
            print(f"DEBUG: {error_msg}")
            self.logger.log_error(error_msg)
            
            formatted_response = self._format_professional_response(
                "System Error",
                f"An unexpected error occurred while processing your request.\n\nError Details: {str(e)}\n\nPlease try rephrasing your query or contact support.",
                "error"
            )
            
            return {
                'type': 'error',
                'message': formatted_response,
                'success': False
            }

    def _handle_details_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str]) -> dict:
        """Handle flexible queries for any value based on row+column matching."""
        print(f"DEBUG: Handling flexible query for: {query}")

        try:
            id_column = self.detect_id_column(data)
            query_lower = query.lower()

            # Extract ID-like values
            possible_ids = re.findall(r'\b[a-z]*\d{3,}\b', query_lower)
            found_id = None

            for pid in possible_ids:
                match = data[data[id_column].astype(str).str.lower() == pid]
                if not match.empty:
                    found_id = pid
                    break

            if not found_id:
                return self._warning_response("Record Not Found", f"No matching ID found in query: '{query}'")

            record = data[data[id_column].astype(str).str.lower() == found_id].iloc[0]

            # Match query words to column names
            query_words = re.findall(r'\b\w+\b', query_lower)

            matched_columns = []

            for col in data.columns:
                for word in query_words:
                    if word in col.lower():
                        matched_columns.append(col)
                        break
                        
            matched_columns = list(dict.fromkeys(matched_columns))  # Remove duplicates

            # If no specific columns matched, return the full record
            if not matched_columns:
                matched_columns = list(data.columns)
    
            # Format the response as a sentence
            description_parts = []
            for col in matched_columns:
                val = record[col]
                col_readable = col.replace("_", " ").replace(" (%)", " percentage").capitalize()
                if isinstance(val, (int, float)) and "percent" in col.lower():
                    description_parts.append(f"{col_readable} is {val}%")
                else:
                    description_parts.append(f"{col_readable} is {val}")

            joined = ", and ".join([
                ", ".join(description_parts[:-1]),
                description_parts[-1]
            ]) if len(description_parts) > 1 else description_parts[0]

            sentence = f"The student with ID **{found_id.upper()}** has the following details: {joined}."

            full_message = self._format_professional_response("Student Information", sentence, "details")

            

        except Exception as e:
            return self._error_response("Query Failed", str(e))

    def _find_matching_records(self, data: pd.DataFrame, id_column: str, search_id: str) -> pd.DataFrame:
        """Find matching records using multiple strategies."""
        
        # Strategy 1: Exact case-insensitive match
        matching_records = data[data[id_column].astype(str).str.upper() == search_id.upper()]
        
        # Strategy 2: Contains match
        if matching_records.empty:
            matching_records = data[data[id_column].astype(str).str.upper().str.contains(search_id.upper(), na=False)]
        
        # Strategy 3: Numeric extraction match
        if matching_records.empty:
            search_numbers = re.findall(r'\d+', search_id)
            if search_numbers:
                search_num = search_numbers[0]
                matching_records = data[data[id_column].astype(str).str.contains(search_num, na=False)]
        
        return matching_records
    
    def _format_record_details(self, record: pd.Series, record_id: str, detected_subjects: List[str]) -> str:
        """Format record details in a professional manner."""
        
        details_lines = []
        
        if detected_subjects:
            # Show only requested subjects
            details_lines.append("REQUESTED INFORMATION:")
            details_lines.append("")
            
            found_subjects = False
            for column, value in record.items():
                col_lower = column.lower()
                if any(subj in col_lower for subj in detected_subjects):
                    details_lines.append(f"{column:<20}: {value}")
                    found_subjects = True
            
            if not found_subjects:
                details_lines.append("⚠️  Requested subjects not found in data")
                details_lines.append("")
                details_lines.append("AVAILABLE INFORMATION:")
                details_lines.append("")
        else:
            details_lines.append("COMPLETE RECORD INFORMATION:")
            details_lines.append("")
        
        # Show all or remaining fields
        if not detected_subjects or not found_subjects:
            for column, value in record.items():
                if not detected_subjects or not any(subj in column.lower() for subj in detected_subjects):
                    formatted_value = str(value) if pd.notna(value) else "N/A"
                    details_lines.append(f"{column:<20}: {formatted_value}")
        
        return "\n".join(details_lines)
    
    def _handle_statistics_query(self, query: str, data: pd.DataFrame, detected_subjects: List[str]) -> dict:
        """Handle statistical queries."""
        try:
            numeric_cols = data.select_dtypes(include='number').columns.tolist()
            
            if not numeric_cols:
                formatted_response = self._format_professional_response(
                    "No Numeric Data",
                    "No numeric columns found for statistical analysis.",
                    "warning"
                )
                return {
                    'type': 'statistics',
                    'message': formatted_response,
                    'success': False
                }
            
            # Filter columns based on detected subjects if any
            target_cols = numeric_cols
            if detected_subjects:
                target_cols = [col for col in numeric_cols 
                             if any(subj in col.lower() for subj in detected_subjects)]
                if not target_cols:
                    target_cols = numeric_cols
            
            stats_content = self._generate_statistics_content(data, target_cols, query)
            
            formatted_response = self._format_professional_response(
                "Statistical Analysis",
                stats_content,
                "info"
            )
            
            return {
                'type': 'statistics',
                'message': formatted_response,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Error in statistics query: {str(e)}"
            self.logger.log_error(error_msg)
            
            formatted_response = self._format_professional_response(
                "Statistics Error",
                f"Failed to generate statistics: {str(e)}",
                "error"
            )
            
            return {
                'type': 'statistics',
                'message': formatted_response,
                'success': False
            }
    
    def _generate_statistics_content(self, data: pd.DataFrame, columns: List[str], query: str) -> str:
        """Generate statistical content based on query type."""
        query_lower = query.lower()
        content_lines = []
        
        for col in columns[:3]:  # Limit to first 3 columns
            stats = data[col].describe()
            content_lines.append(f"📊 {col.upper()}:")
            
            if 'average' in query_lower or 'mean' in query_lower:
                content_lines.append(f"   Average: {stats['mean']:.2f}")
            elif 'total' in query_lower or 'sum' in query_lower:
                content_lines.append(f"   Total: {data[col].sum():.2f}")
            elif 'max' in query_lower or 'highest' in query_lower:
                content_lines.append(f"   Maximum: {stats['max']:.2f}")
            elif 'min' in query_lower or 'lowest' in query_lower:
                content_lines.append(f"   Minimum: {stats['min']:.2f}")
            else:
                # Full statistics
                content_lines.extend([
                    f"   Count: {stats['count']:.0f}",
                    f"   Mean: {stats['mean']:.2f}",
                    f"   Std Dev: {stats['std']:.2f}",
                    f"   Min: {stats['min']:.2f}",
                    f"   Max: {stats['max']:.2f}"
                ])
            content_lines.append("")
        
        return "\n".join(content_lines)
    
    def _handle_top_query(self, query: str, data: pd.DataFrame) -> dict:
        """Handle top/ranking queries."""
        try:
            # Extract number from query
            numbers = re.findall(r'\d+', query)
            top_n = int(numbers[0]) if numbers else 5
            
            numeric_cols = data.select_dtypes(include='number').columns.tolist()
            if not numeric_cols:
                formatted_response = self._format_professional_response(
                    "No Numeric Data",
                    "No numeric columns available for ranking.",
                    "warning"
                )
                return {
                    'type': 'top',
                    'message': formatted_response,
                    'success': False
                }
            
            # Use first numeric column or detect from query
            sort_column = numeric_cols[0]
            for col in numeric_cols:
                if col.lower() in query.lower():
                    sort_column = col
                    break
            
            top_records = data.nlargest(top_n, sort_column)
            id_column = self.detect_id_column(data)
            
            content_lines = [f"TOP {top_n} RECORDS BY {sort_column.upper()}:", ""]
            
            for i, (idx, record) in enumerate(top_records.iterrows(), 1):
                id_val = record[id_column] if id_column else f"Record {idx}"
                score = record[sort_column]
                content_lines.append(f"{i:2d}. {id_val:<12} | {sort_column}: {score}")
            
            formatted_response = self._format_professional_response(
                f"Top {top_n} Rankings",
                "\n".join(content_lines),
                "info"
            )
            
            return {
                'type': 'top',
                'message': formatted_response,
                'success': True,
                'data': top_records.to_dict('records')
            }
            
        except Exception as e:
            error_msg = f"Error in top query: {str(e)}"
            self.logger.log_error(error_msg)
            
            formatted_response = self._format_professional_response(
                "Ranking Error",
                f"Failed to generate rankings: {str(e)}",
                "error"
            )
            
            return {
                'type': 'top',
                'message': formatted_response,
                'success': False
            }
    
    def _handle_comparison_query(self, query: str, data: pd.DataFrame) -> dict:
        """Handle comparison queries."""
        formatted_response = self._format_professional_response(
            "Feature Coming Soon",
            "Comparison functionality is currently under development.\n\nThis feature will allow you to compare records,\nsubjects, and performance metrics.",
            "info"
        )
        
        return {
            'type': 'comparison',
            'message': formatted_response,
            'success': False
        }
    
    def _handle_prediction_query(self, query: str, data: pd.DataFrame, model_handler) -> dict:
        """Handle prediction queries."""
        if model_handler is None:
            formatted_response = self._format_professional_response(
                "Model Required",
                "Prediction functionality requires a machine learning model.\n\nPlease configure a model handler to use this feature.",
                "warning"
            )
            return {
                'type': 'prediction',
                'message': formatted_response,
                'success': False
            }
        
        try:
            if hasattr(model_handler, 'predict') and data is not None:
                predictions = model_handler.predict(data)
                
                formatted_response = self._format_professional_response(
                    "Prediction Results",
                    f"Prediction completed successfully.\n\nResults: {predictions}",
                    "prediction"
                )
                
                return {
                    'type': 'prediction',
                    'message': formatted_response,
                    'success': True,
                    'data': predictions
                }
            else:
                formatted_response = self._format_professional_response(
                    "Prediction Unavailable",
                    "Model handler does not support predictions or no data available.",
                    "warning"
                )
                return {
                    'type': 'prediction',
                    'message': formatted_response,
                    'success': False
                }
                
        except Exception as e:
            formatted_response = self._format_professional_response(
                "Prediction Error",
                f"Prediction failed: {str(e)}",
                "error"
            )
            return {
                'type': 'prediction',
                'message': formatted_response,
                'success': False
            }
    
    def _handle_performance_query(self, query: str, data: pd.DataFrame, model_handler) -> dict:
        """Handle performance analysis queries."""
        if model_handler is None:
            formatted_response = self._format_professional_response(
                "Model Required",
                "Performance analysis requires a machine learning model.\n\nPlease configure a model handler to use this feature.",
                "warning"
            )
            return {
                'type': 'performance',
                'message': formatted_response,
                'success': False
            }
        
        try:
            if hasattr(model_handler, 'get_metrics'):
                metrics = model_handler.get_metrics()
                
                metrics_lines = ["MODEL PERFORMANCE METRICS:", ""]
                for metric, value in metrics.items():
                    metrics_lines.append(f"{metric:<20}: {value}")
                
                formatted_response = self._format_professional_response(
                    "Performance Analysis",
                    "\n".join(metrics_lines),
                    "performance"
                )
                
                return {
                    'type': 'performance',
                    'message': formatted_response,
                    'success': True,
                    'data': metrics
                }
            else:
                formatted_response = self._format_professional_response(
                    "Performance Unavailable", 
                    "Model handler does not support performance analysis.",
                    "warning"
                )
                return {
                    'type': 'performance',
                    'message': formatted_response,
                    'success': False
                }
                
        except Exception as e:
            formatted_response = self._format_professional_response(
                "Performance Error",
                f"Performance analysis failed: {str(e)}",
                "error"
            )
            return {
                'type': 'performance',
                'message': formatted_response,
                'success': False
            }
    
    def _handle_summary_query(self, data: pd.DataFrame) -> dict:
        """Handle dataset summary queries."""
        try:
            summary_content = self._generate_advanced_summary(data)
            
            formatted_response = self._format_professional_response(
                "Dataset Summary",
                summary_content,
                "summary"
            )
            
            return {
                'type': 'summary',
                'message': formatted_response,
                'success': True
            }
            
        except Exception as e:
            formatted_response = self._format_professional_response(
                "Summary Error",
                f"Failed to generate dataset summary: {str(e)}",
                "error"
            )
            return {
                'type': 'summary',
                'message': formatted_response,
                'success': False
            }
    
    def _generate_advanced_summary(self, df: pd.DataFrame) -> str:
        """Generate comprehensive dataset summary."""
        
        summary_lines = [
            f"DATASET OVERVIEW:",
            f"Total Records: {df.shape[0]:,}",
            f"Total Columns: {df.shape[1]}",
            ""
        ]

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if numeric_cols:
            summary_lines.append("📊 NUMERIC ANALYSIS:")
            for col in numeric_cols[:3]:
                stats = df[col].describe()
                summary_lines.extend([
                    f"  {col}:",
                    f"    Mean: {stats['mean']:.2f}",
                    f"    Range: {stats['min']:.2f} - {stats['max']:.2f}",
                    ""
                ])
                
        if categorical_cols:
            summary_lines.append("📝 CATEGORICAL DATA:")
            for col in categorical_cols[:3]:
                unique_count = df[col].nunique()
                most_common = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                summary_lines.extend([
                    f"  {col}:",
                    f"    Unique Values: {unique_count}",
                    f"    Most Common: {most_common}",
                    ""
                ])

        # Data quality assessment
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            summary_lines.append("⚠️  DATA QUALITY:")
            for col, count in null_counts[null_counts > 0].items():
                percentage = (count / len(df)) * 100
                summary_lines.append(f"  {col}: {count} missing ({percentage:.1f}%)")
        else:
            summary_lines.append("✅ DATA QUALITY: Complete (No missing values)")

        return "\n".join(summary_lines)
    
    def _handle_unavailable_feature(self, feature_type: str) -> dict:
        """Handle requests for unavailable features."""
        feature_descriptions = {
            'risk': 'Risk assessment and analysis capabilities',
            'history': 'Query history and previous results tracking'
        }
        
        description = feature_descriptions.get(feature_type, 'This feature')
        
        formatted_response = self._format_professional_response(
            "Feature Coming Soon",
            f"{description} will be available in a future update.\n\nStay tuned for enhanced functionality!",
            "info"
        )
        
        return {
            'type': feature_type,
            'message': formatted_response,
            'success': False
        }
    
    def _handle_general_query(self, data: pd.DataFrame, model_handler) -> dict:
        """Handle general queries and provide guidance."""
        
        available_features = [
            "🔍 Record Details: 'Show details of [ID]'",
            "📊 Statistics: 'Calculate average of [subject]'",
            "🏆 Rankings: 'Show top 5 by [subject]'",
            "📋 Summary: 'Generate dataset summary'"
        ]
        
        if model_handler is not None:
            available_features.extend([
                "🔮 Predictions: 'Predict future performance'",
                "📈 Performance: 'Show model metrics'"
            ])
        
        if data is not None:
            data_info = f"\n\nCurrent Dataset: {data.shape[0]} records, {data.shape[1]} columns"
        else:
            data_info = "\n\nNo dataset currently loaded."
        
        content = "AVAILABLE COMMANDS:\n\n" + "\n".join(available_features) + data_info
        
        formatted_response = self._format_professional_response(
            "Query Assistant",
            content,
            "info"
        )
        
        return {
            'type': 'general',
            'message': formatted_response,
            'success': True
        }

    def generate_dynamic_suggestions(self, query: str, data: pd.DataFrame) -> List[str]:
        """Generate intelligent suggestions based on query context and data."""
        print(f"DEBUG: Generating suggestions for query: '{query}'")
        
        if data is None or data.empty:
            return [
                "💡 Load a dataset to get started",
                "📖 Check available features",
                "❓ Ask for help with commands"
            ]
            
        try:
            query_lower = query.lower()
            suggestions = []

            # Get data characteristics
            numeric_cols = data.select_dtypes(include='number').columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            id_column = self.detect_id_column(data)

            # Context-aware suggestions based on query intent
            if any(k in query_lower for k in ["top", "highest", "rank", "best"]):
                for col in numeric_cols[:2]:
                    suggestions.append(f"🏆 Show top 10 records by {col}")
                if len(numeric_cols) > 1:
                    suggestions.append(f"📊 Compare top performers across subjects")

            elif any(k in query_lower for k in ["average", "mean", "statistics"]):
                for col in numeric_cols[:2]:
                    suggestions.append(f"📈 Calculate detailed statistics for {col}")
                if categorical_cols:
                    suggestions.append(f"📋 Show average by {categorical_cols[0]}")

            elif any(k in query_lower for k in ["detail", "show", "record", "info"]):
                if id_column and not data[id_column].empty:
                    sample_ids = data[id_column].dropna().astype(str).head(3).tolist()
                    for sample_id in sample_ids[:2]:
                        suggestions.append(f"🔍 Show details of {sample_id}")

            elif any(k in query_lower for k in ["summary", "overview"]):
                suggestions.extend([
                    "📊 Generate comprehensive dataset summary",
                    "📈 Show column-wise statistics",
                    "🔍 Analyze data quality"
                ])

            # Default intelligent suggestions
            if not suggestions:
                suggestions = [
                    f"📋 Generate dataset overview",
                    f"🔍 Show details of a specific record",
                    f"📊 Calculate statistics"
                ]
                
                if numeric_cols:
                    suggestions.append(f"🏆 Show top performers in {numeric_cols[0]}")
                
                if id_column and not data[id_column].empty:
                    sample_id = str(data[id_column].iloc[0])
                    suggestions.append(f"👤 Show details of {sample_id}")

            # Ensure unique suggestions and limit to 4
            unique_suggestions = []
            seen = set()
            for suggestion in suggestions:
                if suggestion not in seen:
                    unique_suggestions.append(suggestion)
                    seen.add(suggestion)
                if len(unique_suggestions) >= 4:
                    break

            print(f"DEBUG: Generated {len(unique_suggestions)} suggestions")
            return unique_suggestions

        except Exception as e:
            error_msg = f"Error generating suggestions: {e}"
            print(f"DEBUG: {error_msg}")
            self.logger.log_error(error_msg)
            return [
                "💡 Try asking for dataset summary",
                "🔍 Show record details",
                "📊 Calculate statistics",
                "❓ Get help with commands"
            ]

    def get_interactive_help(self) -> str:
        """Provide interactive help with examples."""
        
        help_content = """
INTERACTIVE QUERY GUIDE:

🔍 RECORD DETAILS:
   • "Show details of [ID]"
   • "Get information for student 12345"  
   • "Display record STU0001"

📊 STATISTICS & ANALYSIS:
   • "Calculate average marks"
   • "Show statistics for mathematics"
   • "What's the mean score in physics?"

🏆 RANKINGS & TOP PERFORMERS:
   • "Show top 10 students"
   • "List highest scorers in chemistry"
   • "Rank by total marks"

📋 DATASET OVERVIEW:
   • "Generate summary"
   • "Show dataset overview"
   • "Analyze data quality"

💡 TIPS:
   • Be specific with record IDs
   • Mention subject names for targeted analysis
   • Use natural language - I understand context!

🎯 EXAMPLE QUERIES:
   • "What are the physics marks of STU0001?"
   • "Show top 5 students by chemistry scores"
   • "Calculate average mathematics marks"
   • "Generate complete dataset summary"
        """
        
        return self._format_professional_response(
            "Interactive Query Assistant",
            help_content.strip(),
            "info"
        )

    def validate_query_format(self, query: str) -> Dict[str, Any]:
        """Validate and suggest improvements for query format."""
        
        validation_result = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'confidence': 1.0
        }
        
        query_lower = query.lower().strip()
        
        # Check for empty or too short queries
        if len(query_lower) < 3:
            validation_result['is_valid'] = False
            validation_result['issues'].append("Query too short")
            validation_result['suggestions'].append("Please provide more specific details")
            validation_result['confidence'] = 0.1
            return validation_result
        
        # Check for vague queries
        vague_queries = ['help', 'what', 'how', 'why', 'show', 'get', 'find']
        if query_lower in vague_queries:
            validation_result['confidence'] = 0.3
            validation_result['suggestions'].append("Try being more specific about what you want to see")
        
        # Check for potential ID queries without clear ID
        if any(word in query_lower for word in ['detail', 'show', 'get', 'find']):
            if not re.search(r'\b[A-Z0-9]{3,}\b', query, re.IGNORECASE):
                validation_result['suggestions'].append("Include a specific ID (e.g., STU0001)")
                validation_result['confidence'] = min(validation_result['confidence'], 0.7)
        
        # Positive indicators
        if re.search(r'\b[A-Z]{2,}[0-9]+\b', query, re.IGNORECASE):
            validation_result['confidence'] = min(validation_result['confidence'] + 0.2, 1.0)
        
        if any(word in query_lower for word in ['average', 'top', 'summary', 'statistics']):
            validation_result['confidence'] = min(validation_result['confidence'] + 0.1, 1.0)
        
        return validation_result
