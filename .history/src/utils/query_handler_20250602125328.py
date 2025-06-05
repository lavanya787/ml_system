# src/utils/query_handler.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import hashlib
from utils.logger import Logger
from utils.llm_manager import LLMManager

class QueryHandler:
    """Handles query classification and routing to appropriate domain handlers."""
    
    # Define supported intents and their corresponding handler methods
    INTENT_HANDLERS = {
        'prediction': 'handle_prediction_query',
        'performance': 'handle_performance_query',
        'risk': 'handle_risk_query',
        'general': 'handle_general_query'
    }
    
    def __init__(self, logger: Logger, llm_manager: Optional[LLMManager] = None):
        self.logger = logger
        self.llm_manager = llm_manager
        
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