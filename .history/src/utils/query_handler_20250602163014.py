
    
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
