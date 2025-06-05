# Enhanced QueryHandler to properly search and retrieve data from files

import pandas as pd
import re
from typing import Dict, Any, List, Optional
from fuzzywuzzy import fuzz

class DomainConfig:
    DOMAIN_KEYWORDS = [
        'student', 'marks', 'grade', 'score', 'attendance', 'exam', 'test',
        'math', 'physics', 'chemistry', 'english', 'subject', 'roll', 'class',
        'percentage', 'curriculum', 'academic', 'school', 'college'
    ]

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.log_info("Education DomainConfig initialized.")

    def detect_domain(self, df):
        if self.logger:
            self.logger.log_info("Running domain detection logic for education.")
        # Add your detection logic
        return {
            'confidence': 0.8,
            'detected_features': {
                'column_count': len(df.columns),
                'example_feature': 'marks'
            }
        }
    
    def handle_general_query(self, query: str, raw_data: pd.DataFrame, processed_data: dict) -> dict:
        """Enhanced general query handler that searches through data."""
        
        if not query or not query.strip():
            return {'error': 'Empty query provided'}
        
        if raw_data is None or raw_data.empty:
            return {'error': 'No data available to search'}
        
        # Try to extract student name and subject from query
        search_result = self._search_student_data(query, raw_data)
        
        if search_result['found']:
            return search_result
        else:
            # Fallback to general data search
            return self._general_data_search(query, raw_data)
    
    def _search_student_data(self, query: str, raw_data: pd.DataFrame) -> dict:
        """Search for student-specific information in the data."""
        
        query_lower = query.lower()
        
        # Extract potential student name (look for names in query)
        # Common patterns: "what is the chemistry mark of sai munshi"
        name_patterns = [
            r'(?:mark|score|grade)(?:\s+of\s+|\s+for\s+)([a-zA-Z\s]+?)(?:\?|$|in|for)',
            r'(?:of|for)\s+([a-zA-Z\s]+?)(?:\s+in|\s+chemistry|\s+physics|\?|$)',
            r'([a-zA-Z\s]+?)(?:\s+chemistry|\s+physics|\s+math|\s+mark|\s+score)'
        ]
        
        potential_names = []
        for pattern in name_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            potential_names.extend([match.strip() for match in matches if len(match.strip()) > 2])
        
        # Extract subject
        subjects = ['chemistry', 'physics', 'mathematics', 'math', 'biology', 'english', 'history']
        found_subject = None
        for subject in subjects:
            if subject in query_lower:
                found_subject = subject
                break
        
        print(f"DEBUG: Potential names found: {potential_names}")
        print(f"DEBUG: Subject found: {found_subject}")
        print(f"DEBUG: Available columns: {raw_data.columns.tolist()}")
        
        # Search through the dataframe
        results = []
        
        for potential_name in potential_names:
            if len(potential_name) < 3:  # Skip very short names
                continue
                
            # Search in all text columns for the name
            for col in raw_data.columns:
                if raw_data[col].dtype == 'object':  # Text columns
                    # Use fuzzy matching to find similar names
                    for idx, value in enumerate(raw_data[col]):
                        if pd.isna(value):
                            continue
                        
                        value_str = str(value).lower()
                        # Check if the potential name is in the cell value
                        if potential_name in value_str or self._fuzzy_match(potential_name, value_str):
                            # Found a potential match, get the entire row
                            row_data = raw_data.iloc[idx].to_dict()
                            
                            # If we found a subject, try to get that specific mark
                            if found_subject:
                                subject_data = self._find_subject_mark(row_data, found_subject)
                                if subject_data:
                                    results.append({
                                        'student_name': value_str,
                                        'subject': found_subject,
                                        'mark': subject_data['mark'],
                                        'column': subject_data['column'],
                                        'row_index': idx,
                                        'full_row': row_data
                                    })
                            else:
                                # Return all numeric data for this student
                                numeric_data = {k: v for k, v in row_data.items() 
                                              if isinstance(v, (int, float)) and not pd.isna(v)}
                                results.append({
                                    'student_name': value_str,
                                    'all_marks': numeric_data,
                                    'row_index': idx,
                                    'full_row': row_data
                                })
        
        if results:
            return {
                'found': True,
                'message': f'Found {len(results)} result(s) for query: "{query}"',
                'results': results,
                'query_type': 'student_search'
            }
        else:
            return {
                'found': False,
                'message': f'No results found for query: "{query}"',
                'searched_names': potential_names,
                'searched_subject': found_subject,
                'suggestions': self._get_search_suggestions(raw_data)
            }
    
    def _fuzzy_match(self, name1: str, name2: str, threshold: int = 80) -> bool:
        """Check if two names are similar using fuzzy matching."""
        try:
            return fuzz.partial_ratio(name1.lower(), name2.lower()) >= threshold
        except:
            # Fallback if fuzzywuzzy is not available
            return name1.lower() in name2.lower() or name2.lower() in name1.lower()
    
    def _find_subject_mark(self, row_data: dict, subject: str) -> Optional[dict]:
        """Find the mark for a specific subject in a row."""
        subject_variations = {
            'chemistry': ['chemistry', 'chem', 'chemistry_mark', 'chemistry_score'],
            'physics': ['physics', 'phy', 'physics_mark', 'physics_score'],
            'mathematics': ['mathematics', 'math', 'maths', 'math_mark', 'mathematics_score'],
            'biology': ['biology', 'bio', 'biology_mark', 'biology_score'],
            'english': ['english', 'eng', 'english_mark', 'english_score']
        }
        
        variations = subject_variations.get(subject, [subject])
        
        for col_name, value in row_data.items():
            col_lower = col_name.lower()
            if any(var in col_lower for var in variations):
                if isinstance(value, (int, float)) and not pd.isna(value):
                    return {'mark': value, 'column': col_name}
        
        return None
    
    def _general_data_search(self, query: str, raw_data: pd.DataFrame) -> dict:
        """General search through all data."""
        query_terms = query.lower().split()
        results = []
        
        # Search through all text columns
        for col in raw_data.columns:
            if raw_data[col].dtype == 'object':
                for idx, value in enumerate(raw_data[col]):
                    if pd.isna(value):
                        continue
                    
                    value_str = str(value).lower()
                    if any(term in value_str for term in query_terms if len(term) > 2):
                        row_data = raw_data.iloc[idx].to_dict()
                        results.append({
                            'matched_in_column': col,
                            'matched_value': str(value),
                            'row_index': idx,
                            'full_row': row_data
                        })
        
        if results:
            return {
                'found': True,
                'message': f'Found {len(results)} general result(s)',
                'results': results[:10],  # Limit to first 10 results
                'query_type': 'general_search'
            }
        else:
            return {
                'found': False,
                'message': 'No matching data found',
                'data_overview': self._get_data_overview(raw_data)
            }
    
    def _get_search_suggestions(self, raw_data: pd.DataFrame) -> List[str]:
        """Get suggestions based on available data."""
        suggestions = []
        
        # Get sample names from text columns
        text_columns = [col for col in raw_data.columns if raw_data[col].dtype == 'object']
        
        for col in text_columns[:3]:  # Check first 3 text columns
            sample_values = raw_data[col].dropna().head(5).tolist()
            for value in sample_values:
                if isinstance(value, str) and len(value) > 5:
                    suggestions.append(f"Try searching for: {value}")
        
        # Add column-based suggestions
        numeric_cols = [col for col in raw_data.columns if raw_data[col].dtype in ['int64', 'float64']]
        if numeric_cols:
            suggestions.append(f"Available subjects/marks: {', '.join(numeric_cols)}")
        
        return suggestions[:5]  # Limit suggestions
    
    def _get_data_overview(self, raw_data: pd.DataFrame) -> dict:
        """Get overview of available data."""
        return {
            'total_rows': len(raw_data),
            'total_columns': len(raw_data.columns),
            'columns': raw_data.columns.tolist(),
            'sample_data': raw_data.head(3).to_dict() if not raw_data.empty else 'No data',
            'data_types': raw_data.dtypes.to_dict()
        }
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: dict) -> dict:
        """Handle prediction queries."""
        return self.handle_general_query(query, raw_data, {})
    
    def handle_performance_query(self, query: str, raw_data: pd.DataFrame, processed_data: dict) -> dict:
        """Handle performance queries."""
        return self.handle_general_query(query, raw_data, processed_data)
    
    def handle_risk_query(self, query: str, raw_data: pd.DataFrame, processed_data: dict) -> dict:
        """Handle risk queries."""
        return self.handle_general_query(query, raw_data, processed_data)


# Test function to verify the enhanced query handler
def test_enhanced_query_handler():
    """Test the enhanced query handler with sample student data."""
    
    # Create sample student data
    sample_data = pd.DataFrame({
        'student_name': ['Sai Munshi', 'John Doe', 'Jane Smith', 'Raj Patel', 'Priya Kumar'],
        'chemistry': [85, 92, 78, 88, 95],
        'physics': [90, 87, 82, 91, 89],
        'mathematics': [88, 95, 85, 86, 92],
        'class': ['10A', '10B', '10A', '10C', '10B'],
        'roll_number': [101, 102, 103, 104, 105]
    })
    
    print("Sample Data:")
    print(sample_data)
    print("\n" + "="*50 + "\n")
    
    # Test the enhanced domain config
    domain_config = DomainConfig()
    
    # Test queries
    test_queries = [
        "what is the chemistry mark of sai munshi",
        "chemistry mark of Sai Munshi",
        "sai munshi chemistry",
        "show me John Doe physics score",
        "what are Jane Smith marks"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        result = domain_config.handle_general_query(query, sample_data, {})
        print(f"Result: {result}")
        print("-" * 40)


# Usage with your actual QueryHandler
def integrate_with_query_handler():
    """Show how to integrate with your existing QueryHandler."""
    
    from utils.query_handler import QueryHandler
    from utils.logger import Logger
    
    # Create logger (or use your existing one)
    try:
        logger = Logger()
    except:
        class MockLogger:
            def log_info(self, msg): print(f"INFO: {msg}")
            def log_warning(self, msg): print(f"WARNING: {msg}")
            def log_error(self, msg): print(f"ERROR: {msg}")
        logger = MockLogger()
    
    # Initialize query handler with enhanced domain config
    query_handler = QueryHandler(logger=logger)
    enhanced_domain_config = DomainConfig()
    query_handler.add_domain_config(enhanced_domain_config)