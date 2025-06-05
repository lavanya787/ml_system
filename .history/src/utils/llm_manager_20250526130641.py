# src/utils/llm_manager.py

from typing import List, Dict

class LLMManager:
    def __init__(self):
        # Initialize LLM models, APIs, etc.
        pass

    def process_query(self, queries: List[str]) -> List[Dict[str, str]]:
        """Process the query using an LLM model and return structured result."""
        results = []
        for query in queries:
            # Dummy intent classification logic (replace with actual model/API call)
            if "predict" in query.lower():
                intent = 'prediction'
            elif "performance" in query.lower():
                intent = 'performance'
            elif "risk" in query.lower():
                intent = 'risk'
            else:
                intent = 'general'

            results.append({'query': query, 'intent': intent})

        return results

    # Optional: use lazy import if you ever need QueryHandler inside LLMManager
    def use_query_handler(self):
        from utils.query_handler import QueryHandler
        # You can now safely use QueryHandler inside this method
