# src/utils/query_handler.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import hashlib
import os
from utils.logger import Logger
from utils.llm_manager import LLMManager


class KnowledgeBaseHandler:
    """Handles knowledge-based responses for different query types."""
    
    def __init__(self):
        self.knowledge_base = {
            'prediction': {
                'techniques': ['Time series forecasting (ARIMA, LSTM, Prophet)', 'Regression models', 'ML approaches (XGBoost, Neural Networks)'],
                'factors': ['Historical patterns', 'External variables', 'Data quality', 'Model validation'],
                'best_practices': ['Use ensemble methods', 'Validate with out-of-sample data', 'Consider uncertainty intervals']
            },
            'performance': {
                'metrics': {
                    'classification': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                    'regression': ['MAE', 'RMSE', 'R²', 'MAPE'],
                    'clustering': ['Silhouette Score', 'Davies-Bouldin Index']
                },
                'evaluation_methods': ['Cross-validation', 'Hold-out validation', 'Time series validation'],
                'improvement_strategies': ['Feature engineering', 'Hyperparameter tuning', 'Model ensemble']
            },
            'risk': {
                'types': ['Model risk (overfitting)', 'Data risk (quality, bias)', 'Operational risk (deployment)', 'Regulatory risk'],
                'assessment_methods': ['Statistical testing', 'Sensitivity analysis', 'Stress testing', 'Monte Carlo simulation'],
                'mitigation_strategies': ['Robust validation', 'Continuous monitoring', 'Model governance', 'Regular updates']
            }
        }
    
    def generate_response(self, intent: str, query: str) -> Dict[str, Any]:
        """Generate knowledge-based response based on intent."""
        if intent == 'prediction':
            return {
                'type': 'prediction',
                'recommendations': self.knowledge_base['prediction'],
                'message': 'Here are comprehensive prediction recommendations and best practices.'
            }
        elif intent == 'performance':
            return {
                'type': 'performance',
                'metrics': self.knowledge_base['performance']['metrics'],
                'evaluation_methods': self.knowledge_base['performance']['evaluation_methods'],
                'improvement_strategies': self.knowledge_base['performance']['improvement_strategies'],
                'message': 'Here\'s a comprehensive performance evaluation guide.'
            }
        elif intent == 'risk':
            return {
                'type': 'risk',
                'risk_types': self.knowledge_base['risk']['types'],
                'assessment_methods': self.knowledge_base['risk']['assessment_methods'],
                'mitigation_strategies': self.knowledge_base['risk']['mitigation_strategies'],
                'message': 'Comprehensive risk assessment guidance.'
            }
        else:
            return {
                'type': 'general',
                'message': 'I can help with machine learning, data science, and analytics questions.',
                'areas_of_expertise': ['Model prediction', 'Performance evaluation', 'Risk assessment', 'Data preprocessing']
            }


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
    
    def search_history(self, user_id: str, search_term: str) -> List[Dict]:
        """Search user's query history."""
        user_history = self.history.get(user_id, [])
        search_term_lower = search_term.lower()
        
        results = []
        for record in user_history:
            if search_term_lower in record['query'].lower():
                results.append(record)
        
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)


class QueryHandler:
    """Streamlined query handler with knowledge-based responses."""
    
    INTENT_KEYWORDS = {
        'prediction': ['predict', 'forecast', 'future', 'estimate', 'projection'],
        'performance': ['performance', 'accuracy', 'metrics', 'evaluate', 'score'],
        'risk': ['risk', 'danger', 'threat', 'hazard', 'vulnerability'],
        'history': ['history', 'past', 'previous', 'show', 'list', 'search']
    }
    
    def __init__(self, logger: Logger = None, llm_manager: Optional[LLMManager] = None):
        self.logger = logger or self._create_simple_logger()
        self.llm_manager = llm_manager
        self.query_history = QueryHistory()
        self.knowledge_handler = KnowledgeBaseHandler()
        self.query_logger = QueryLogger()
        
    def _create_simple_logger(self):
        """Create simple logger if none provided."""
        class SimpleLogger:
            def log_info(self, msg): print(f"INFO: {msg}")
            def log_warning(self, msg): print(f"WARNING: {msg}")
            def log_error(self, msg): print(f"ERROR: {msg}")
        return SimpleLogger()
    
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
    
    def _handle_history_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """Handle history-related queries."""
        query_lower = query.lower()
        
        if 'search' in query_lower:
            # Extract search term (simple approach)
            parts = query.split()
            search_term = ' '.join(parts[2:]) if len(parts) > 2 else ''
            if search_term:
                results = self.query_history.search_history(user_id, search_term)
                return {
                    'message': f'Found {len(results)} queries matching "{search_term}"',
                    'results': results[:5]  # Limit to 5 results
                }
        
        # Default: show recent history
        history = self.query_history.get_user_history(user_id)
        return {
            'message': f'Your recent {len(history)} queries:',
            'history': history
        }
    
    def handle_query(self, query: str, user_id: str, domain_config: Any = None, 
                raw_data: pd.DataFrame = None, processed_data: Dict[str, Any] = None, 
                models: Dict[str, Any] = None, domain: str = None, 
                model_handler: Any = None) -> Dict[str, Any]:

        try:
            classification = self.classify_query(query)
            intent = classification['intent']
            confidence = classification['confidence']

            query_id = hashlib.md5(f"{user_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]

            # ✅ Step 1: Try model-based answering only for specific cases
            model_keywords = ['predict', 'classification', 'forecast', 'train', 'model']
            if any(k in query.lower() for k in model_keywords):
                if model_handler and model_handler.get_model(domain):
                    try:
                        model_answer = model_handler.answer_question(question=query, domain=domain)
                        if model_answer.get("has_answer"):
                            model_answer.update({
                                'intent': 'model_specific',
                                'confidence': 1.0,
                                'user_id': user_id,
                                'query_id': query_id
                            })
                            self.query_logger.log_query(user_id, query, model_answer, domain)
                            self.query_history.add_query(user_id, query, model_answer)
                            return model_answer
                    except Exception as e:
                        self.logger.log_warning(f"Model handler failed: {str(e)}")
                else:
                    return {
                        "query": query,
                            "intent": "model_specific",
                        "has_answer": False,
                        "answer": "No model loaded. Please train a model first.",
                        "confidence": 1.0,
                        "user_id": user_id,
                    "query_id": query_id
                }

        # ✅ Step 2: Handle history
        if intent == 'history':
            result = self._handle_history_query(query, user_id)
            response = {
                'type': 'history',
                'result': result,
                'intent': intent,
                'confidence': confidence,
                'user_id': user_id,
                'query_id': query_id
            }
            return response

        # ✅ Step 3: Handle "list of students"
        query_lower = query.lower()
        if 'list of students' in query_lower or ('students' in query_lower and 'list' in query_lower):
            student_cols = [col for col in raw_data.columns if 'name' in col.lower() or 'id' in col.lower()]
            if student_cols:
                data_preview = raw_data[student_cols].drop_duplicates().head(50)
                return {
                    "query": query,
                    "intent": "data_lookup",
                    "has_answer": True,
                    "columns": student_cols,
                    "answer": data_preview.to_dict(orient="records"),
                    "confidence": 0.9,
                    "user_id": user_id,
                    "query_id": query_id
                }
            else:
                return {
                    "query": query,
                    "intent": "data_lookup",
                    "has_answer": False,
                    "answer": "Could not detect a student ID or name column.",
                    "confidence": 0.6,
                    "user_id": user_id,
                    "query_id": query_id
                }

        # ✅ Step 4: Knowledge-based fallback
        kb_response = self.knowledge_handler.generate_response(intent, query)
        response = {
            **kb_response,
            'intent': intent,
            'confidence': confidence,
            'user_id': user_id,
            'query_id': query_id,
            'query': query
        }
        self.query_logger.log_query(user_id, query, response, domain)
        if intent != 'history':
            self.query_history.add_query(user_id, query, response)
        return response

    except Exception as e:
        self.logger.log_error(f"Query handling failed: {str(e)}")
        error_response = {
            'type': 'error',
            'message': f'Sorry, I encountered an error: {str(e)}',
            'intent': 'error',
            'confidence': 0.0,
            'user_id': user_id,
            'query': query
        }
        self.query_logger.log_query(user_id, query, error_response, domain)
        return error_response
