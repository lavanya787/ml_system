# src/utils/query_handler.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import hashlib
import os
import re
import uuid
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
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load query history: {e}")
            return {}
    
    def _save_history(self):
        """Save query history to storage file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_file) if os.path.dirname(self.storage_file) else '.', exist_ok=True)
            
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
        if len(words) > 1:
            return ' '.join(words[:-1]) + '...'
        return clean_query[:50] + '...' if len(clean_query) > 50 else clean_query
    
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
            query_text = query_record.get('query', '').lower()
            title_text = query_record.get('title', '').lower()
            
            if (search_term_lower in query_text or 
                search_term_lower in title_text):
                results.append(query_record)
        
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_query(self, user_id: str, query_id: str) -> bool:
        """Delete a specific query from history."""
        if user_id in self.history:
            original_length = len(self.history[user_id])
            self.history[user_id] = [
                q for q in self.history[user_id] 
                if q.get('query_id') != query_id
            ]
            if len(self.history[user_id]) < original_length:
                self._save_history()
                return True
        return False
    
    def clear_user_history(self, user_id: str) -> bool:
        """Clear all history for a user."""
        if user_id in self.history:
            self.history[user_id] = []
            self._save_history()
            return True
        return False
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's query history."""
        user_history = self.history.get(user_id, [])
        
        if not user_history:
            return {
                'total_queries': 0,
                'domains': {},
                'date_range': None
            }
        
        # Count queries by domain
        domain_counts = {}
        dates = []
        
        for query in user_history:
            domain = query.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            dates.append(query.get('date'))
        
        # Filter out None dates and sort
        valid_dates = [d for d in dates if d]
        date_range = None
        if valid_dates:
            valid_dates.sort()
            date_range = {
                'earliest': valid_dates[0],
                'latest': valid_dates[-1]
            }
        
        return {
            'total_queries': len(user_history),
            'domains': domain_counts,
            'date_range': date_range
        }


class MockDomainConfig:
    """Mock domain configuration for fallback handling."""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame = None, models: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle prediction queries with mock response."""
        return {
            'type': 'prediction',
            'message': f'Prediction query received: "{query}"',
            'status': 'mock_response',
            'details': 'This is a mock response. Please configure proper domain handlers.'
        }
    
    def handle_performance_query(self, query: str, raw_data: pd.DataFrame = None, processed_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle performance queries with mock response."""
        return {
            'type': 'performance',
            'message': f'Performance query received: "{query}"',
            'status': 'mock_response',
            'details': 'This is a mock response. Please configure proper domain handlers.'
        }
    
    def handle_risk_query(self, query: str, raw_data: pd.DataFrame = None, processed_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle risk queries with mock response."""
        return {
            'type': 'risk',
            'message': f'Risk query received: "{query}"',
            'status': 'mock_response',
            'details': 'This is a mock response. Please configure proper domain handlers.'
        }
    
    def handle_general_query(self, query: str, raw_data: pd.DataFrame = None, processed_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle general queries with mock response."""
        return {
            'type': 'general',
            'message': f'General query received: "{query}"',
            'status': 'mock_response',
            'details': 'This is a mock response. Please configure proper domain handlers.'
        }


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
    
    # Keywords for basic intent classification (fallback when LLM is unavailable)
    INTENT_KEYWORDS = {
        'prediction': ['predict', 'forecast', 'future', 'estimate', 'projection', 'next', 'upcoming'],
        'performance': ['performance', 'accuracy', 'metrics', 'evaluate', 'score', 'results'],
        'risk': ['risk', 'danger', 'threat', 'hazard', 'vulnerability', 'exposure'],
        'history': ['history', 'past', 'previous', 'show', 'list', 'search', 'clear']
    }
    
    def __init__(self, logger: Logger = None, llm_manager: Optional[LLMManager] = None):
        self.logger = logger or self._create_default_logger()
        self.llm_manager = llm_manager
        self.query_history = QueryHistory()
        self.mock_domain_config = MockDomainConfig(self.logger)
        
    def _create_default_logger(self) -> Logger:
        """Create a default logger if none provided."""
        try:
            return Logger()
        except:
            # Fallback simple logger
            class SimpleLogger:
                def log_info(self, msg): print(f"INFO: {msg}")
                def log_warning(self, msg): print(f"WARNING: {msg}")
                def log_error(self, msg): print(f"ERROR: {msg}")
            return SimpleLogger()
        
    def classify_query(self, query: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Classify query intent using LLM with fallback to keyword-based classification.
        
        Returns:
            Dict containing 'intent' and 'confidence' keys
        """
        # Check if it's a history-related query first
        if self._is_history_query(query):
            return {'intent': 'history', 'confidence': 1.0}
        
        # Try LLM classification if available
        if self.llm_manager:
            for attempt in range(max_retries + 1):
                try:
                    query_result = self.llm_manager.process_query([query])[0]
                    intent = query_result.get('intent', 'general')
                    confidence = query_result.get('confidence', 0.5)
                    
                    # Validate intent is supported
                    if intent not in self.INTENT_HANDLERS:
                        self.logger.log_warning(f"Unknown intent '{intent}', falling back to keyword classification")
                        return self._classify_by_keywords(query)
                    
                    self.logger.log_info(f"LLM classified query intent: {intent} (confidence: {confidence})")
                    return {'intent': intent, 'confidence': confidence}
                    
                except Exception as e:
                    self.logger.log_warning(f"LLM classification attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries:
                        self.logger.log_warning("All LLM classification attempts failed, falling back to keyword classification")
                        return self._classify_by_keywords(query)
        else:
            self.logger.log_info("No LLM manager available, using keyword classification")
        
        # Fallback to keyword-based classification
        return self._classify_by_keywords(query)
    
    def _classify_by_keywords(self, query: str) -> Dict[str, Any]:
        """Classify query intent based on keywords."""
        query_lower = query.lower()
        scores = {}
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[intent] = score
        
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent] * 0.3, 0.9)  # Scale confidence
            self.logger.log_info(f"Keyword classified query intent: {best_intent} (confidence: {confidence})")
            return {'intent': best_intent, 'confidence': confidence}
        else:
            self.logger.log_info("No keywords matched, defaulting to general intent")
            return {'intent': 'general', 'confidence': 0.5}
    
    def _is_history_query(self, query: str) -> bool:
        """Check if query is asking for history/past queries."""
        history_keywords = [
            'history', 'past queries', 'previous questions', 'show my queries',
            'what did i ask', 'my questions', 'search history', 'past questions',
            'show history', 'query history', 'recent queries', 'last queries'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in history_keywords)

    def _validate_domain_config(self, domain_config: Any, intent: str) -> bool:
        """Validate that domain_config has the required handler method."""
        if domain_config is None:
            return False
            
        handler_method = self.INTENT_HANDLERS.get(intent)
        if not handler_method:
            return False
            
        if not hasattr(domain_config, handler_method):
            return False
            
        return True
    
    def _generate_user_id(self) -> str:
        """Generate a default user ID if none provided."""
        return f"user_{str(uuid.uuid4())[:8]}"

    def handle_query(self, query: str, domain_config: Any = None, raw_data: pd.DataFrame = None,
                     processed_data: Dict[str, Any] = None, models: Dict[str, Any] = None, 
                     user_id: str = None, domain: str = None) -> Dict[str, Any]:
        """
        Handle query by routing to appropriate domain config method.
        
        Args:
            query: User query string
            domain_config: Domain-specific configuration object with handler methods
            raw_data: Raw dataset
            processed_data: Processed data dictionary
            models: Dictionary of trained models
            user_id: Unique identifier for the user (auto-generated if None)
            domain: Detected domain name
            
        Returns:
            Dictionary containing query results or error information
        """
        # Generate user_id if not provided
        if user_id is None:
            user_id = self._generate_user_id()
            self.logger.log_info(f"Generated user_id: {user_id}")
        
        # Initialize variables for error handling
        intent = 'unknown'
        confidence = 0.0
        
        try:
            # Classify the query
            classification_result = self.classify_query(query)
            intent = classification_result['intent']
            confidence = classification_result['confidence']
            
            # Log low confidence classifications
            if confidence < 0.7:
                self.logger.log_warning(f"Low confidence classification: {intent} ({confidence})")
            
            # Handle history queries directly
            if intent == 'history':
                result = self._handle_history_query(query, user_id)
                response = {
                    'result': result,
                    'intent': intent,
                    'confidence': confidence,
                    'query': query,
                    'user_id': user_id,
                    'is_history': True
                }
                # Save history queries to history as well
                query_id = self.query_history.add_query(user_id, query, response, domain)
                response['query_id'] = query_id
                return response
            
            # Use mock config if no domain config provided or validation fails
            effective_config = domain_config
            if not self._validate_domain_config(domain_config, intent):
                if domain_config is None:
                    self.logger.log_info(f"No domain config provided, using mock handler for intent: {intent}")
                else:
                    self.logger.log_warning(f"Domain config validation failed for intent: {intent}, using mock handler")
                effective_config = self.mock_domain_config
            
            # Route to appropriate handler
            handler_method = self.INTENT_HANDLERS[intent]
            handler_func = getattr(effective_config, handler_method)
            
            # Call the appropriate handler based on intent
            try:
                if intent == 'prediction':
                    result = handler_func(query, raw_data, models)
                elif intent in ['performance', 'risk']:
                    result = handler_func(query, raw_data, processed_data)
                else:  # general
                    result = handler_func(query, raw_data, processed_data)
            except TypeError as e:
                # Handle cases where handler doesn't accept expected parameters
                self.logger.log_warning(f"Handler parameter mismatch: {e}, trying with just query")
                result = handler_func(query)
            
            # Enhance result with metadata
            if isinstance(result, dict):
                result.update({
                    'intent': intent,
                    'confidence': confidence,
                    'query': query,
                    'user_id': user_id
                })
            else:
                # Wrap non-dict results
                result = {
                    'result': result,
                    'intent': intent,
                    'confidence': confidence,
                    'query': query,
                    'user_id': user_id
                }
            
            # Save to history
            query_id = self.query_history.add_query(user_id, query, result, domain)
            result['query_id'] = query_id
            
            self.logger.log_info(f"Successfully handled query with intent: {intent}")
            return result
            
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            self.logger.log_error(error_msg)
            result = {
                'error': error_msg,
                'query': query,
                'intent': intent,
                'user_id': user_id,
                'type': 'error',
                'message': f'Failed to process query: {query}',
                'details': str(e)
            }
            # Save error to history
            try:
                query_id = self.query_history.add_query(user_id, query, result, domain)
                result['query_id'] = query_id
            except Exception as history_error:
                self.logger.log_error(f"Failed to save error to history: {history_error}")
            
            return result
    
    def _handle_history_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """Handle queries asking for history information."""
        query_lower = query.lower()
        
        # Handle stats request
        if 'stats' in query_lower or 'statistics' in query_lower:
            stats = self.query_history.get_stats(user_id)
            return {
                'type': 'statistics',
                'stats': stats,
                'message': f'Found {stats["total_queries"]} total queries'
            }
        
        # Handle clear history request
        if 'clear' in query_lower and 'history' in query_lower:
            success = self.query_history.clear_user_history(user_id)
            return {
                'type': 'clear_history',
                'success': success,
                'message': 'History cleared successfully' if success else 'Failed to clear history'
            }
        
        # Search for specific terms
        if 'search' in query_lower:
            # Extract search term (basic approach)
            words = query.split()
            try:
                search_idx = next(i for i, word in enumerate(words) if 'search' in word.lower())
                if search_idx < len(words) - 1:
                    search_term = ' '.join(words[search_idx + 1:])
                    # Remove common words
                    search_term = search_term.replace('for', '').replace('in', '').strip()
                    if search_term:
                        results = self.query_history.search_history(user_id, search_term)
                        return {
                            'type': 'search_results',
                            'search_term': search_term,
                            'results': results,
                            'count': len(results),
                            'message': f'Found {len(results)} queries matching "{search_term}"'
                        }
            except StopIteration:
                pass
        
        # Get recent history (default)
        limit = 10
        if 'last' in query_lower or 'recent' in query_lower:
            # Try to extract number
            numbers = re.findall(r'\d+', query)
            if numbers:
                limit = min(int(numbers[0]), 50)  # Cap at 50
        elif 'all' in query_lower:
            limit = None
        
        history = self.query_history.get_user_history(user_id, limit)
        return {
            'type': 'history_list',
            'history': history,
            'count': len(history),
            'limit': limit,
            'message': f'Showing {"all" if limit is None else f"last {limit}"} queries ({len(history)} found)'
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
    
    def remove_intent_handler(self, intent: str) -> bool:
        """
        Remove support for an intent type.
        
        Args:
            intent: Intent name to remove
            
        Returns:
            True if removed, False if not found
        """
        if intent in self.INTENT_HANDLERS:
            del self.INTENT_HANDLERS[intent]
            self.logger.log_info(f"Removed handler for intent: {intent}")
            return True
        return False
    
    def get_user_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get query history for a user."""
        return self.query_history.get_user_history(user_id, limit)
    
    def search_user_history(self, user_id: str, search_term: str) -> List[Dict]:
        """Search through user's query history."""
        return self.query_history.search_history(user_id, search_term)
    
    def get_query_by_id(self, user_id: str, query_id: str) -> Optional[Dict]:
        """Get a specific query by ID."""
        return self.query_history.get_query_by_id(user_id, query_id)
    
    def delete_query(self, user_id: str, query_id: str) -> bool:
        """Delete a specific query from history."""
        return self.query_history.delete_query(user_id, query_id)
    
    def clear_user_history(self, user_id: str) -> bool:
        """Clear all history for a user."""
        return self.query_history.clear_user_history(user_id)
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's query history."""
        return self.query_history.get_stats(user_id)
