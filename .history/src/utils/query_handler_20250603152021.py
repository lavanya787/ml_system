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
    
    