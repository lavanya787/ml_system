# src/utils/query_handler.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import hashlib
import os
import re
import uuid
import traceback
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
    
    def __init__(self, logger: Logger = None, llm_manager: Optional[LLMManager] = None, debug_mode: bool = False):
        self.logger = logger or self._create_default_logger()
        self.llm_manager = llm_manager
        self.query_history = QueryHistory()
        self.debug_mode = debug_mode
        
        # Log initialization status
        self.logger.log_info("QueryHandler initialized")
        self.logger.log_info(f"LLM Manager available: {self.llm_manager is not None}")
        self.logger.log_info(f"Debug mode: {self.debug_mode}")
        
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
        Classify query intent using LLM with retry logic and enhanced debugging.
        
        Returns:
            Dict containing 'intent' and 'confidence' keys
        """
        if self.debug_mode:
            self.logger.log_info(f"=== QUERY CLASSIFICATION START ===")
            self.logger.log_info(f"Query: '{query}'")
        
        # Check if it's a history-related query
        if self._is_history_query(query):
            if self.debug_mode:
                self.logger.log_info("Query identified as history query by keyword matching")
            return {'intent': 'history', 'confidence': 1.0}
        
        if not self.llm_manager:
            self.logger.log_warning("No LLM manager available, defaulting to general intent")
            return {'intent': 'general', 'confidence': 0.0}
            
        for attempt in range(max_retries + 1):
            try:
                if self.debug_mode:
                    self.logger.log_info(f"LLM classification attempt {attempt + 1}")
                
                # Check if LLM manager is properly configured
                if not hasattr(self.llm_manager, 'process_query'):
                    self.logger.log_error("LLM manager missing process_query method")
                    return {'intent': 'general', 'confidence': 0.0}
                
                if self.debug_mode:
                    self.logger.log_info("Calling LLM manager process_query...")
                
                query_result = self.llm_manager.process_query([query])[0]
                
                if self.debug_mode:
                    self.logger.log_info(f"LLM returned: {query_result}")
                
                intent = query_result.get('intent', 'general')
                confidence = query_result.get('confidence', 0.5)
                
                # Validate intent is supported
                if intent not in self.INTENT_HANDLERS:
                    self.logger.log_warning(f"Unknown intent '{intent}', defaulting to general")
                    intent = 'general'
                    confidence = 0.0
                
                if self.debug_mode:
                    self.logger.log_info(f"Final classification: intent={intent}, confidence={confidence}")
                    self.logger.log_info(f"=== QUERY CLASSIFICATION END ===")
                    
                return {'intent': intent, 'confidence': confidence}
                
            except Exception as e:
                error_msg = f"Query classification attempt {attempt + 1} failed: {str(e)}"
                self.logger.log_warning(error_msg)
                
                if self.debug_mode:
                    self.logger.log_error(f"Exception type: {type(e).__name__}")
                    self.logger.log_error(f"Full traceback: {traceback.format_exc()}")
                
                if attempt == max_retries:
                    self.logger.log_error("All query classification attempts failed, defaulting to general")
                    return {'intent': 'general', 'confidence': 0.0}
                    
        return {'intent': 'general', 'confidence': 0.0}
    
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
            self.logger.log_error("Domain config is None")
            return False
            
        handler_method = self.INTENT_HANDLERS.get(intent)
        if not handler_method:
            self.logger.log_error(f"No handler method defined for intent: {intent}")
            return False
            
        if not hasattr(domain_config, handler_method):
            self.logger.log_error(f"Domain config missing method: {handler_method}")
            if self.debug_mode:
                available_methods = [method for method in dir(domain_config) if not method.startswith('_')]
                self.logger.log_error(f"Available methods: {available_methods}")
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
        
        # Debug logging for input parameters
        if self.debug_mode:
            self.logger.log_info(f"=== QUERY HANDLER DEBUG START ===")
            self.logger.log_info(f"Query: '{query}'")
            self.logger.log_info(f"User ID: {user_id}")
            self.logger.log_info(f"Domain: {domain}")
            self.logger.log_info(f"Domain config available: {domain_config is not None}")
            self.logger.log_info(f"Raw data available: {raw_data is not None}")
            self.logger.log_info(f"Processed data available: {processed_data is not None}")
            self.logger.log_info(f"Models available: {models is not None}")
            self.logger.log_info(f"LLM Manager available: {self.llm_manager is not None}")
        
        # Initialize variables for error handling
        intent = 'unknown'
        confidence = 0.0
        
        try:
            # Classify the query
            if self.debug_mode:
                self.logger.log_info("Starting query classification...")
                
            classification_result = self.classify_query(query)
            intent = classification_result['intent']
            confidence = classification_result['confidence']
            
            self.logger.log_info(f"Query classified as: {intent} (confidence: {confidence})")
            
            # Log low confidence classifications
            if confidence < 0.7:
                self.logger.log_warning(f"Low confidence classification: {intent} ({confidence})")
            
            # Handle history queries directly
            if intent == 'history':
                if self.debug_mode:
                    self.logger.log_info("Handling as history query")
                    
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
                
                if self.debug_mode:
                    self.logger.log_info(f"History query handled successfully, query_id: {query_id}")
                    self.logger.log_info(f"=== QUERY HANDLER DEBUG END ===")
                    
                return response
            
            # For non-history queries, validate domain config
            if self.debug_mode:
                self.logger.log_info(f"Validating domain config for intent: {intent}")
                
            if not self._validate_domain_config(domain_config, intent):
                error_msg = f"Cannot handle intent '{intent}' - domain config validation failed"
                self.logger.log_error(error_msg)
                
                # Enhanced error info for debugging
                debug_info = {
                    'domain_config_available': domain_config is not None,
                    'expected_method': self.INTENT_HANDLERS.get(intent),
                    'intent': intent,
                    'confidence': confidence
                }
                
                if domain_config is not None:
                    debug_info['available_methods'] = [method for method in dir(domain_config) if not method.startswith('_')]
                else:
                    debug_info['available_methods'] = []
                
                result = {
                    'error': error_msg,
                    'intent': intent,
                    'confidence': confidence,
                    'query': query,
                    'user_id': user_id,
                    'debug_info': debug_info
                }
                
                # Still save to history even if there's an error
                query_id = self.query_history.add_query(user_id, query, result, domain)
                result['query_id'] = query_id
                
                if self.debug_mode:
                    self.logger.log_error(f"Domain config validation failed. Debug info: {debug_info}")
                    self.logger.log_info(f"=== QUERY HANDLER DEBUG END ===")
                    
                return result
            
            # Route to appropriate handler
            handler_method = self.INTENT_HANDLERS[intent]
            handler_func = getattr(domain_config, handler_method)
            
            if self.debug_mode:
                self.logger.log_info(f"Calling handler method: {handler_method}")
            
            # Call the appropriate handler based on intent
            if intent == 'prediction':
                if models is None:
                    raise ValueError("Models dictionary is required for prediction queries")
                if self.debug_mode:
                    self.logger.log_info("Calling prediction handler")
                result = handler_func(query, raw_data, models)
                
            elif intent in ['performance', 'risk']:
                if processed_data is None:
                    raise ValueError(f"Processed data is required for {intent} queries")
                if self.debug_mode:
                    self.logger.log_info(f"Calling {intent} handler")
                result = handler_func(query, raw_data, processed_data)
                
            else:  # general
                if self.debug_mode:
                    self.logger.log_info("Calling general handler")
                result = handler_func(query, raw_data, processed_data)
            
            if self.debug_mode:
                self.logger.log_info(f"Handler returned result type: {type(result)}")
                if isinstance(result, dict) and 'error' in result:
                    self.logger.log_warning(f"Handler returned error: {result.get('error')}")
            
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
            
            self.logger.log_info(f"Successfully handled query with intent: {intent}, query_id: {query_id}")
            
            if self.debug_mode:
                self.logger.log_info(f"=== QUERY HANDLER DEBUG END ===")
                
            return result
            
        except AttributeError as e:
            error_msg = f"Domain config method error: {str(e)}"
            self.logger.log_error(error_msg)
            
            if self.debug_mode:
                self.logger.log_error(f"Exception details: {type(e).__name__}: {e}")
                self.logger.log_error(f"Full traceback: {traceback.format_exc()}")
            
            result = {
                'error': error_msg,
                'query': query,
                'intent': intent,
                'user_id': user_id,
                'exception_type': type(e).__name__
            }
            # Save error to history
            query_id = self.query_history.add_query(user_id, query, result, domain)
            result['query_id'] = query_id
            return result
            
        except ValueError as e:
            error_msg = f"Invalid parameters: {str(e)}"
            self.logger.log_error(error_msg)
            
            result = {
                'error': error_msg,
                'query': query,
                'intent': intent,
                'user_id': user_id,
                'exception_type': type(e).__name__
            }
            # Save error to history
            query_id = self.query_history.add_query(user_id, query, result, domain)
            result['query_id'] = query_id
            return result
            
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            self.logger.log_error(error_msg)
            
            if self.debug_mode:
                self.logger.log_error(f"Exception details: {type(e).__name__}: {e}")
                self.logger.log_error(f"Full traceback: {traceback.format_exc()}")
            
            result = {
                'error': error_msg,
                'query': query,
                'intent': intent,
                'user_id': user_id,
                'exception_type': type(e).__name__
            }
            # Save error to history
            query_id = self.query_history.add_query(user_id, query, result, domain)
            result['query_id'] = query_id
            return result
    
    def _handle_history_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """Handle queries asking for history information."""
        query_lower = query.lower()
        
        # Handle stats request
        if 'stats' in query_lower or 'statistics' in query_lower:
            stats = self.query_history.get_stats(user_id)
            return {
                'type': 'statistics',
                'stats': stats
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
                            'count': len(results)
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
            'limit': limit
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
    
    def enable_debug_mode(self):
        """Enable detailed debug logging."""
        self.debug_mode = True
        self.logger.log_info("Debug mode enabled")
    
    def disable_debug_mode(self):
        """Disable detailed debug logging."""
        self.debug_mode = False
        self.logger.log_info("Debug mode disabled")
    
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
    
    def diagnose_system(self, domain_config=None, sample_query="test query") -> Dict[str, Any]:
        """
        Run a comprehensive system diagnostic.
        
        Args:
            domain_config: Domain configuration to test
            sample_query: Query to use for testing
            
        Returns:
            Dictionary with diagnostic results
        """
        diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'query_handler_status': {},
            'llm_manager_status': {},
            'domain_config_status': {},
            'classification_test': {},
            'recommendations': []
        }
        
        # Check QueryHandler status
        diagnosis['query_handler_status'] = {
            'logger_available': self.logger is not None,
            'llm_manager_available': self.llm_manager is not None,
            'query_history_initialized': self.query_history is not None,
            'debug_mode': self.debug_mode,
            'supported_intents': list(self.INTENT_HANDLERS.keys())
        }
        
        # Check LLM Manager
        if self.llm_manager:
            diagnosis['llm_manager_status'] = {
                'type': str(type(self.llm_manager)),
                'has_process_query': hasattr(self.llm_manager, 'process_query'),
                'test_result': None,
                'test_error': None
            }
            
            try:
                result = self.llm_manager.process_query([sample_query])
                diagnosis['llm_manager_status']['test_result'] = result
            except Exception as e:
                diagnosis['llm_manager_status']['test_error'] = str(e)
        else:
            diagnosis['llm_manager_status'] = {'available': False}
        
        # Check Domain Config
        if domain_config:
            available_methods = [m for m in dir(domain_config) if not m.startswith('_')]
            required_methods = {}
            
            for intent, method in self.INTENT_HANDLERS.items():
                required_methods[intent] = {
                    'method': method,
                    'available': hasattr(domain_config, method)
                }
            
            diagnosis['domain_config_status'] = {
                'type': str(type(domain_config)),
                'available_methods': available_methods,
                'required_methods': required_methods
            }
        else:
            diagnosis['domain_config_status'] = {'available': False}
        
        # Test Classification
        try:
            classification = self.classify_query(sample_query)
            diagnosis['classification_test'] = {
                'success': True,
                'intent': classification.get('intent'),
                'confidence': classification.get('confidence')
            }
        except Exception as e:
            diagnosis['classification_test'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc() 
                if self.debug_mode else None
            }
        # Recommendations based on diagnostics