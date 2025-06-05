from typing import Dict, Any, Optional
from utils.logger import Logger
from utils.llm_manager import LLMManager

class QueryHandler:
    def __init__(self, logger: Logger, llm_manager: LLMManager):
        self.logger = logger
        self.llm_manager = llm_manager
    
    def classify_query(self, query: str) -> str:
        """Classify query intent using LLM."""
        try:
            query_result = self.llm_manager.process_query([query])[0]
            intent = query_result.get('intent', 'general')
            self.logger.log_info(f"Classified query intent: {intent}")
            return intent
        except Exception as e:
            self.logger.log_error(f"Query classification failed: {str(e)}")
            return 'general'
    
    def handle_query(self, query: str, domain_config: Any, raw_data: pd.DataFrame,
                     processed_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle query by routing to appropriate domain config method."""
        try:
            intent = self.classify_query(query)
            
            if intent == 'prediction':
                result = domain_config.handle_prediction_query(query, raw_data, models)
            elif intent == 'performance':
                result = domain_config.handle_performance_query(query, raw_data, processed_data)
            elif intent == 'risk':
                result = domain_config.handle_risk_query(query, raw_data, processed_data)
            else:
                result = domain_config.handle_general_query(query, raw_data, processed_data)
            
            self.logger.log_info(f"Handled query with intent: {intent}")
            return result
        except Exception as e:
            self.logger.log_error(f"Query handling failed: {str(e)}")
            return {'error': f"Query processing failed: {str(e)}"}