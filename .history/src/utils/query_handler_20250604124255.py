# src/utils/query_handler.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import hashlib
import os
import uuid
from utils.logger import Logger
from utils.llm_manager import LLMManager
import re
class QueryLogger:
    """Simple logging of queries to files."""
    
    def __init__(self, log_folder: str = "logs"):
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        self.queries_file = os.path.join(self.log_folder, "queries.jsonl")
    
    def log_query(self, user_id: str, query: str, response: Dict[str, Any], domain: str = None):
        """Log query and response."""
        timestamp = datetime.now()
        entry = {
            "user_id": user_id,
            "query": query,
            "response": response,
            "domain": domain,
            "timestamp": timestamp.isoformat(),
            "intent": response.get('intent', 'unknown')
        }
        
        try:
            with open(self.queries_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')
        except Exception as e:
            print(f"Error logging query: {e}")


class QueryHistory:
    """Simple query history management."""
    
    def __init__(self, storage_file: str = "query_history.json"):
        self.storage_file = storage_file
        self.history = self._load_history()
    
    def _load_history(self) -> Dict[str, List[Dict]]:
        """Load query history from file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _save_history(self):
        """Save query history to file."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def add_query(self, user_id: str, query: str, response: Dict[str, Any]):
        """Add query to history."""
        if user_id not in self.history:
            self.history[user_id] = []
        
        query_record = {
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history[user_id].append(query_record)
        self._save_history()
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent query history for user."""
        user_history = self.history.get(user_id, [])
        return sorted(user_history, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_query_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Alias for get_user_history for backward compatibility."""
        return self.get_user_history(user_id, limit)
    
    def search_history(self, user_id: str, search_term: str) -> List[Dict]:
        """Search user's query history."""
        user_history = self.history.get(user_id, [])
        search_term_lower = search_term.lower()
        
        results = []
        for record in user_history:
            if search_term_lower in record['query'].lower():
                results.append(record)
        
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)


class IntentClassifier:
    """Simple intent classification."""
    
    INTENT_KEYWORDS = {
        'prediction': ['predict', 'forecast', 'future', 'estimate', 'projection'],
        'performance': ['performance', 'accuracy', 'metrics', 'evaluate', 'score'],
        'risk': ['risk', 'danger', 'threat', 'hazard', 'vulnerability'],
        'history': ['history', 'past', 'previous', 'show', 'list', 'search'],
        'summary': ['summary', 'summarize', 'overview', 'highlight', 'key points'],
        'details': ['details', 'show', 'display', 'find', 'get', 'record']
    }
    
    def classify_intent(self, query: str) -> tuple:
        """Classify query intent using keywords."""
        query_lower = query.lower()
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                confidence = 0.7 if intent != 'general' else 0.5
                return intent, confidence
        
        return 'general', 0.5


class QueryHandler:    
    INTENT_KEYWORDS = {
        'prediction': ['predict', 'forecast', 'future', 'estimate', 'projection'],
        'performance': ['performance', 'accuracy', 'metrics', 'evaluate', 'score'],
        'risk': ['risk', 'danger', 'threat', 'hazard', 'vulnerability'],
        'history': ['history', 'past', 'previous', 'show', 'list', 'search'],
        'summary': ['summary', 'summarize', 'overview', 'highlight', 'key points'],
        'details': ['details', 'show', 'display', 'find', 'get', 'record']
    }
    
    def __init__(self, logger: Logger = None, llm_manager: Optional[LLMManager] = None):
        self.logger = logger or self._create_simple_logger()
        self.llm_manager = llm_manager
        self.query_history = QueryHistory()
        self.query_logger = QueryLogger()
        self.intent_classifier = IntentClassifier()
        
        # Initialize placeholder handlers
        self.prediction_handler = self._create_placeholder_handler('prediction')
        self.performance_analyzer = self._create_placeholder_handler('performance')
        self.risk_analyzer = self._create_placeholder_handler('risk')
        
    def _create_simple_logger(self):
        """Create simple logger if none provided."""
        class SimpleLogger:
            def log_info(self, msg): print(f"INFO: {msg}")
            def log_warning(self, msg): print(f"WARNING: {msg}")
            def log_error(self, msg): print(f"ERROR: {msg}")
        return SimpleLogger()
    
    def _create_placeholder_handler(self, handler_type: str):
        """Create placeholder handlers for missing components."""
        class PlaceholderHandler:
            def __init__(self, handler_type):
                self.type = handler_type
            
            def predict(self, query, user_id):
                return {'message': f'Prediction functionality not yet implemented', 'data': []}
            
            def analyze_performance(self):
                return {'message': f'Performance analysis not yet implemented', 'data': []}
            
            def assess_risk(self):
                return {'message': f'Risk assessment not yet implemented', 'data': []}
        
        return PlaceholderHandler(handler_type)
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query intent using LLM or keywords."""
        # Try LLM classification first
        if self.llm_manager:
            try:
                query_result = self.llm_manager.process_query([query])[0]
                intent = query_result.get('intent', 'general')
                confidence = query_result.get('confidence', 0.5)
                return {'intent': intent, 'confidence': confidence}
            except Exception as e:
                self.logger.log_warning(f"LLM classification failed: {str(e)}")
        
        # Fallback to keyword classification
        return self._classify_by_keywords(query)
    
    def _classify_by_keywords(self, query: str) -> Dict[str, Any]:
        """Simple keyword-based classification."""
        query_lower = query.lower()
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                confidence = 0.7 if intent != 'general' else 0.5
                return {'intent': intent, 'confidence': confidence}
        
        return {'intent': 'general', 'confidence': 0.5}
    
    def detect_id_column(self, data: pd.DataFrame) -> Optional[str]:
        text_cols = data.select_dtypes(include='object').columns.tolist()
        priority_keywords = ['id', 'name', 'identifier', 'code', 'regno', 'roll']
    
        for keyword in priority_keywords:
            for col in text_cols:
                if keyword in col.lower():
                    return col
        return text_cols[0] if text_cols else None

    def detect_numeric_columns(self, data: pd.DataFrame, query_lower: str) -> List[str]:
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        matched = [col for col in numeric_cols if col.lower() in query_lower]
        return matched if matched else numeric_cols
    
    def handle_query(self, user_id: str, query: str, domain: str, raw_data: pd.DataFrame = None, processed_data: pd.DataFrame = None) -> dict:
        """
        Handle incoming queries with support for both raw_data and processed_data parameters.
        """
        # Use processed_data if available, otherwise fall back to raw_data
        data = processed_data if processed_data is not None else raw_data
        
        intent, confidence = self.intent_classifier.classify_intent(query)
        query_id = str(uuid.uuid4())

        # Handle specific record lookup (e.g., "Show details of STU0001")
        if intent == 'details' and data is not None:
            return self._handle_details_query(user_id, query, domain, data, query_id, intent, confidence)

        # 1. Prediction Intent
        if intent == 'prediction':
            prediction_response = self.prediction_handler.predict(query, user_id)
            prediction_response.update({
                'type': 'prediction',
                'intent': intent,
                'confidence': confidence,
                'user_id': user_id,
                'query_id': query_id
            })
            self.query_logger.log_query(user_id, query, prediction_response, domain)
            self.query_history.add_query(user_id, query, prediction_response)
            return prediction_response

        # 2. Performance Analysis Intent
        elif intent == 'performance':
            performance_response = self.performance_analyzer.analyze_performance()
            performance_response.update({
                'type': 'performance',
                'intent': intent,
                'confidence': confidence,
                'user_id': user_id,
                'query_id': query_id
            })
            self.query_logger.log_query(user_id, query, performance_response, domain)
            self.query_history.add_query(user_id, query, performance_response)
            return performance_response

        # 3. Risk Analysis Intent
        elif intent == 'risk':
            risk_response = self.risk_analyzer.assess_risk()
            risk_response.update({
                'type': 'risk',
                'intent': intent,
                'confidence': confidence,
                'user_id': user_id,
                'query_id': query_id
            })
            self.query_logger.log_query(user_id, query, risk_response, domain)
            self.query_history.add_query(user_id, query, risk_response)
            return risk_response

        # 4. History or Search Intent
        elif intent == 'history':
            history = self.query_history.get_query_history(user_id)
            history_response = {
                'type': 'history',
                'intent': intent,
                'confidence': confidence,
                'user_id': user_id,
                'query_id': query_id,
                'message': 'Query history retrieved successfully.',
                'data': history
            }
            self.query_logger.log_query(user_id, query, history_response, domain)
            return history_response

        # 5. Summary Intent
        elif intent == 'summary':
            if data is not None:
                try:
                    summary_text = self._generate_summary(data, query=query)
                    summary_response = {
                        'type': 'summary',
                        'summary': summary_text,
                        'intent': intent,
                        'confidence': confidence,
                        'user_id': user_id,
                        'query_id': query_id,
                        'message': 'Generated summary of the dataset based on your query.'
                    }
                    self.query_logger.log_query(user_id, query, summary_response, domain)
                    self.query_history.add_query(user_id, query, summary_response)
                    return summary_response
                except Exception as e:
                    self.logger.log_error(f"Summary generation failed: {e}")
                    return {
                        'type': 'summary',
                        'summary': '',
                        'intent': intent,
                        'confidence': confidence,
                        'user_id': user_id,
                        'query_id': query_id,
                        'message': 'Failed to generate summary. Please try again.'
                    }
            else:
                return {
                    'type': 'summary',
                    'summary': '',
                    'intent': intent,
                    'confidence': confidence,
                    'user_id': user_id,
                    'query_id': query_id,
                    'message': 'No data available for summary generation.'
                }

        # Fallback for Unknown Intent
        else:
            fallback_response = {
                'type': 'unknown',
                'intent': intent,
                'confidence': confidence,
                'user_id': user_id,
                'query_id': query_id,
                'message': 'Sorry, I could not understand your query.'
            }
            self.query_logger.log_query(user_id, query, fallback_response, domain)
            self.query_history.add_query(user_id, query, fallback_response)
            return fallback_response

    def _handle_details_query(self, user_id: str, query: str, domain: str, data: pd.DataFrame, query_id: str, intent: str, confidence: float) -> dict:
        """Handle queries asking for details of specific records."""
        try:
            # Extract the identifier from the query (e.g., "STU0001" from "Show details of STU0001")
            import re
            # Look for alphanumeric identifiers that might be student IDs, codes, etc.
            matches = re.findall(r'\b[A-Z]{2,}[0-9]+\b|\b[0-9]+[A-Z]{2,}\b|\b[A-Z0-9]{5,}\b', query.upper())
            
            if not matches:
                # Try to find any word that might be an identifier
                words = query.split()
                potential_ids = [word for word in words if len(word) > 2 and (word.isalnum() or any(c.isdigit() for c in word))]
                matches = potential_ids[:1]  # Take the first potential match
            
            if matches:
                search_id = matches[0]
                
                # Find the best column to search in
                id_column = self.detect_id_column(data)
                
                if id_column:
                    # Search for the record
                    matching_records = data[data[id_column].astype(str).str.upper() == search_id.upper()]
                    
                    if not matching_records.empty:
                        record_dict = matching_records.iloc[0].to_dict()
                        details_response = {
                            'type': 'details',
                            'intent': intent,
                            'confidence': confidence,
                            'user_id': user_id,
                            'query_id': query_id,
                            'message': f'Details found for {search_id}',
                            'record': record_dict,
                            'search_id': search_id
                        }
                    else:
                        # Record not found
                        details_response = {
                            'type': 'details',
                            'intent': intent,
                            'confidence': confidence,
                            'user_id': user_id,
                            'query_id': query_id,
                            'message': f'No record found for {search_id}',
                            'record': None,
                            'search_id': search_id
                        }
                else:
                    details_response = {
                        'type': 'details',
                        'intent': intent,
                        'confidence': confidence,
                        'user_id': user_id,
                        'query_id': query_id,
                        'message': 'No suitable ID column found in the data',
                        'record': None,
                        'search_id': search_id
                    }
            else:
                details_response = {
                    'type': 'details',
                    'intent': intent,
                    'confidence': confidence,
                    'user_id': user_id,
                    'query_id': query_id,
                    'message': 'Could not extract ID from query',
                    'record': None,
                    'search_id': None
                }
            
            self.query_logger.log_query(user_id, query, details_response, domain)
            self.query_history.add_query(user_id, query, details_response)
            return details_response
            
        except Exception as e:
            self.logger.log_error(f"Error handling details query: {e}")
            error_response = {
                'type': 'details',
                'intent': intent,
                'confidence': confidence,
                'user_id': user_id,
                'query_id': query_id,
                'message': f'Error processing details query: {str(e)}',
                'record': None,
                'search_id': None
            }
            return error_response

    def generate_dynamic_suggestions(self, query: str, data: pd.DataFrame) -> list:
        query_lower = query.lower()
        suggestions = []

        text_cols = data.select_dtypes(include='object').columns.tolist()
        numeric_cols = data.select_dtypes(include='number').columns.tolist()

        if any(k in query_lower for k in ["top", "highest", "rank", "score"]):
            for col in numeric_cols[:2]:
                suggestions.append(f"Top 5 records by {col}")

        if any(k in query_lower for k in ["average", "mean", "total", "sum"]):
            for col in numeric_cols[:2]:
                suggestions.append(f"Average of {col}")
            if text_cols and numeric_cols:
                suggestions.append(f"Average of {numeric_cols[0]} by {text_cols[0]}")

        if any(k in query_lower for k in ["who", "what", "value", "of", "record", "detail"]):
            if text_cols:
                sample_val = str(data[text_cols[0]].dropna().unique()[0])
                suggestions.append(f"Show details of {sample_val}")

        if not suggestions and text_cols and numeric_cols:
            suggestions += [
                f"List of {text_cols[0]}s",
                f"Top 3 by {numeric_cols[0]}",
                f"Average of {numeric_cols[0]}"
            ]

        return suggestions[:3]
    
    def _generate_summary(self, df: pd.DataFrame, query: str = "") -> str:
        try:
            # Basic Data Overview
            summary_lines = [
                f"🔍 The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.",
                f"📊 Column types: {dict(df.dtypes.value_counts())}"
            ]

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

            if numeric_cols:
                summary_lines.append(f"\n🔢 **Numeric columns**: {', '.join(numeric_cols)}")
                numeric_desc = df[numeric_cols].describe().T.round(2)
                for col in numeric_desc.index:
                    stats = numeric_desc.loc[col]
                    summary_lines.append(
                        f"  - {col}: mean={stats['mean']}, std={stats['std']}, min={stats['min']}, max={stats['max']}"
                    )

            if categorical_cols:
                summary_lines.append(f"\n🔡 **Categorical columns**: {', '.join(categorical_cols)}")
                for col in categorical_cols:
                    top_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                    summary_lines.append(
                        f"  - {col}: top value = '{top_val}', unique = {df[col].nunique()}"
                    )

            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            if total_nulls > 0:
                summary_lines.append(f"\n⚠️ Missing values: {total_nulls} across {null_counts[null_counts > 0].shape[0]} columns")
                for col, count in null_counts[null_counts > 0].items():
                    summary_lines.append(f"  - {col}: {count} missing")

            summary = "\n".join(summary_lines)

            # Optional: Enhance with LLM (if available/configured)
            if hasattr(self, 'llm') and self.llm:
                prompt = (
                    f"You are a data analyst. Summarize the following dataset info in simple, friendly language.\n"
                    f"User query: {query}\n"
                    f"Data summary:\n{summary}"
                )
                llm_summary = self.llm.generate(prompt)  # Or OpenAI API if integrated
                return llm_summary.strip()

            return summary

        except Exception as e:
            self.logger.log_error(f"Error generating summary: {e}")
            return "An error occurred while generating the summary. Please check the dataset."