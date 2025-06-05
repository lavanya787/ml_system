# src/utils/llm_manager.py

from typing import List, Dict
from utils.logger import Logger

class LLMManager:
    def __init__(self, logger: Logger):
        self.logger = logger

    def process_query(self, queries: List[str]) -> List[Dict[str, str]]:
        """Mock LLM logic: Classifies intent based on keyword presence."""
        results = []
        for query in queries:
            intent = self._classify_intent(query)
            self.logger.log_info(f"LLM classified '{query}' as intent: {intent}")
            results.append({'query': query, 'intent': intent})
        return results

    def _classify_intent(self, query: str) -> str:
        """Private method to determine the intent based on keywords."""
        q = query.lower()
        if "predict" in q or "forecast" in q:
            return "prediction"
        elif "performance" in q or "accuracy" in q:
            return "performance"
        elif "risk" in q or "threat" in q:
            return "risk"
        else:
            return "general"
