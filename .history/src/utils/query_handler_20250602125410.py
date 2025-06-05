# src/utils/query_handler.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import hashlib
from utils.logger import Logger
from utils.llm_manager import LLMManager
class QueryHistory:
    """Manages query history for users."""
    
    def __init__(self, storage_file: str = "query_history.json"):
        self.storage_file = storage_file
        self.history = self._load_history()
    
    def _load_history(self) -> Dict[str, List[Dict]]:
        """Load query history from storage file."""
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_history(self):
        """Save query history to storage file."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def add_query(self, user_id: str, query: str, response: Dict[str, Any], 
                  domain: str = None) -> str:
        """
        Add a query to history.
        
        Args:
            user_id: Unique identifier for the user
            query: The user's query
            response: The system's response
            domain: Detected domain (optional)
            
        Returns:
            Query ID for reference
        """
        timestamp = datetime.now()
        query_id = hashlib.md5(f"{user_id}_{timestamp.isoformat()}".encode()).hexdigest()[:8]
        
        # Generate a title from the query (first 50 chars or first sentence)
        title = self._generate_title(query)
        
        query_record = {
            'query_id': query_id,
            'title': title,
            'query': query,
            'response': response,
            'domain': domain,
            'timestamp': timestamp.isoformat(),
            'date': timestamp.strftime('%Y-%m-%d'),
            'time': timestamp.strftime('%H:%M:%S')
        }
        
        if user_id not in self.history:
            self.history[user_id] = []
        
        self.history[user_id].append(query_record)
        self._save_history()
        
        return query_id
    
    def _generate_title(self, query: str) -> str:
        """Generate a meaningful title from the query."""
        # Remove extra whitespace and limit length
        clean_query = ' '.join(query.split())
        
        # Try to find first sentence
        sentences = clean_query.split('.')
        if len(sentences) > 1 and len(sentences[0]) < 60:
            return sentences[0].strip() + '.'
        
        # Otherwise, truncate at word boundary
        if len(clean_query) <= 50:
            return clean_query
        
        words = clean_query[:50].split()
        return ' '.join(words[:-1]) + '...'
    
    def get_user_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get query history for a user."""
        user_history = self.history.get(user_id, [])
        
        # Sort by timestamp (newest first)
        sorted_history = sorted(user_history, 
                              key=lambda x: x['timestamp'], 
                              reverse=True)
        
        if limit:
            return sorted_history[:limit]
        return sorted_history
    
    def get_query_by_id(self, user_id: str, query_id: str) -> Optional[Dict]:
        """Get a specific query by ID."""
        user_history = self.history.get(user_id, [])
        for query_record in user_history:
            if query_record['query_id'] == query_id:
                return query_record
        return None
    
    def search_history(self, user_id: str, search_term: str) -> List[Dict]:
        """Search through user's query history."""
        user_history = self.history.get(user_id, [])
        results = []
        
        search_term_lower = search_term.lower()
        for query_record in user_history:
            if (search_term_lower in query_record['query'].lower() or 
                search_term_lower in query_record['title'].lower()):
                results.append(query_record)
        
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)
    
class QueryHandler:
    """Handles query classification and routing to appropriate domain handlers."""
    
    # Define supported intents and their corresponding handler methods
    INTENT_HANDLERS = {
        'prediction': 'handle_prediction_query',
        'performance': 'handle_performance_query',
        'risk': 'handle_risk_query',
        'general': 'handle_general_query',
        'history': 'handle_history_query'
    }
    
    def __init__(self, logger: Logger, llm_manager: Optional[LLMManager] = None):
        self.logger = logger
        self.llm_manager = llm_manager
                self.query_history = QueryHistory()

        
    def classify_query(self, query: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Classify query intent using LLM with retry logic.
        
        Returns:
            Dict containing 'intent' and 'confidence' keys
        """
        if not self.llm_manager:
            self.logger.log_warning("No LLM manager available, defaulting to general intent")
            return {'intent': 'general', 'confidence': 0.0}
            
        for attempt in range(max_retries + 1):
            try:
                query_result = self.llm_manager.process_query([query])[0]
                intent = query_result.get('intent', 'general')
                confidence = query_result.get('confidence', 0.5)
                
                # Validate intent is supported
                if intent not in self.INTENT_HANDLERS:
                    self.logger.log_warning(f"Unknown intent '{intent}', defaulting to general")
                    intent = 'general'
                    confidence = 0.0
                
                self.logger.log_info(f"Classified query intent: {intent} (confidence: {confidence})")
                return {'intent': intent, 'confidence': confidence}
                
            except Exception as e:
                self.logger.log_warning(f"Query classification attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries:
                    self.logger.log_error("All query classification attempts failed, defaulting to general")
                    return {'intent': 'general', 'confidence': 0.0}
                    
        return {'intent': 'general', 'confidence': 0.0}

    def _validate_domain_config(self, domain_config: Any, intent: str) -> bool:
        """Validate that domain_config has the required handler method."""
        handler_method = self.INTENT_HANDLERS.get(intent)
        if not handler_method:
            return False
            
        if not hasattr(domain_config, handler_method):
            self.logger.log_error(f"Domain config missing method: {handler_method}")
            return False
            
        return True

    def handle_query(self, query: str, domain_config: Any, raw_data: pd.DataFrame,
                     processed_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle query by routing to appropriate domain config method.
        
        Args:
            query: User query string
            domain_config: Domain-specific configuration object with handler methods
            raw_data: Raw dataset
            processed_data: Processed data dictionary
            models: Dictionary of trained models
            
        Returns:
            Dictionary containing query results or error information
        """
        try:
            # Classify the query
            classification_result = self.classify_query(query)
            intent = classification_result['intent']
            confidence = classification_result['confidence']
            
            # Log low confidence classifications
            if confidence < 0.7:
                self.logger.log_warning(f"Low confidence classification: {intent} ({confidence})")
            
            # Validate domain config has required method
            if not self._validate_domain_config(domain_config, intent):
                self.logger.log_error(f"Cannot handle intent '{intent}' - missing handler method")
                return {
                    'error': f"Handler not available for query type: {intent}",
                    'intent': intent,
                    'confidence': confidence
                }
            
            # Route to appropriate handler
            handler_method = self.INTENT_HANDLERS[intent]
            handler_func = getattr(domain_config, handler_method)
            
            # Call the appropriate handler based on intent
            if intent == 'prediction':
                result = handler_func(query, raw_data, models)
            elif intent in ['performance', 'risk']:
                result = handler_func(query, raw_data, processed_data)
            else:  # general
                result = handler_func(query, raw_data, processed_data)
            
            # Enhance result with metadata
            if isinstance(result, dict):
                result.update({
                    'intent': intent,
                    'confidence': confidence,
                    'query': query
                })
            
            self.logger.log_info(f"Successfully handled query with intent: {intent}")
            return result
            
        except AttributeError as e:
            error_msg = f"Domain config method error: {str(e)}"
            self.logger.log_error(error_msg)
            return {
                'error': error_msg,
                'query': query,
                'intent': intent if 'intent' in locals() else 'unknown'
            }
            
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            self.logger.log_error(error_msg)
            return {
                'error': error_msg,
                'query': query,
                'intent': intent if 'intent' in locals() else 'unknown'
            }
    
    def get_supported_intents(self) -> List[str]:
        """Return list of supported query intents."""
        return list(self.INTENT_HANDLERS.keys())
    
    def add_intent_handler(self, intent: str, handler_method: str) -> None:
        """
        Add support for a new intent type.
        
        Args:
            intent: Intent name
            handler_method: Method name on domain_config that handles this intent
        """
        self.INTENT_HANDLERS[intent] = handler_method
        self.logger.log_info(f"Added handler for intent: {intent} -> {handler_method}")