import pandas as pd
from typing import Dict, Any, List, Optional
from importlib import import_module
import streamlit as st
import concurrent.futures
from fuzzywuzzy import fuzz
from utils.logger import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DomainDetector:
    CONFIDENCE_THRESHOLD = 0.15  # Lowered threshold for better detection
    MIN_KEYWORD_MATCH_THRESHOLD = 0.1  # Minimum keyword match score

    def __init__(self, logger: Logger):
        self.logger = logger
        self.domains = [
            'customer_support', 'entertainment', 'gaming', 'legal', 'marketing',
            'logistics', 'manufacturing', 'real_estate', 'agriculture', 'energy',
            'hospitality', 'automobile', 'telecommunications', 'government',
            'food_beverage', 'it_services', 'event_management', 'insurance',
            'retail', 'hr_resources', 'education', 'generic'  # Added 'education' domain
        ]
        
        # Define fallback keywords for common domains
        self.fallback_keywords = {
            'education': ['student', 'marks', 'grade', 'score', 'attendance', 'exam', 'test', 
                         'math', 'physics', 'chemistry', 'english', 'subject', 'roll', 'class',
                         'percentage', 'curriculum', 'academic', 'school', 'college'],
            'hr_resources': ['employee', 'salary', 'department', 'position', 'hire', 'staff'],
            'customer_support': ['ticket', 'complaint', 'issue', 'support', 'customer', 'service'],
            'retail': ['product', 'price', 'sale', 'inventory', 'purchase', 'order'],
            'marketing': ['campaign', 'lead', 'conversion', 'revenue', 'marketing'],
            'finance': ['amount', 'transaction', 'payment', 'balance', 'account'],
            'generic': ['data', 'record', 'entry', 'information', 'details']
        }
        
        self.domain_configs = self._load_domain_configs()
        self.logger.log_info(f"Initialized DomainDetector with {len(self.domain_configs)} domain configs")

    def _load_domain_configs(self) -> Dict[str, Any]:
        configs = {}
        for domain in self.domains:
            try:
                module = import_module(f"domain_configs.{domain}")
                domain_config_instance = module.DomainConfig(logger=self.logger)

                if not hasattr(domain_config_instance, 'detect_domain'):
                    self.logger.log_error(f"Loaded domain config for '{domain}' does NOT have 'detect_domain' method.")
                else:
                    self.logger.log_info(f"Loaded domain config for '{domain}' is valid.")

                configs[domain] = domain_config_instance
            except Exception as e:
                self.logger.log_error(f"Failed to load domain '{domain}': {e}", exc_info=True)
        return configs

    def _get_domain_keywords(self, domain_name: str) -> List[str]:
        """Get keywords for a domain, with fallback to predefined keywords"""
        keywords = []
        
        # Try to get keywords from domain config
        if domain_name in self.domain_configs:
            config = self.domain_configs[domain_name]
            keywords = getattr(config, "DOMAIN_KEYWORDS", [])
        
        # If no keywords from config, use fallback
        if not keywords and domain_name in self.fallback_keywords:
            keywords = self.fallback_keywords[domain_name]
            self.logger.log_info(f"Using fallback keywords for domain '{domain_name}': {keywords[:5]}...")
        
        return keywords

    def _analyze_column_names(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze column names for domain-specific patterns"""
        column_text = " ".join(df.columns.str.lower())
        domain_scores = {}
        
        for domain_name in self.domains:
            keywords = self._get_domain_keywords(domain_name)
            if not keywords:
                continue
                
            # Count keyword matches in column names
            matches = sum(1 for keyword in keywords if keyword.lower() in column_text)
            score = matches / len(keywords) if keywords else 0
            domain_scores[domain_name] = score
            
            if matches > 0:
                self.logger.log_info(f"Domain '{domain_name}': {matches} keyword matches in columns, score: {score:.4f}")
        
        return domain_scores

    def _analyze_data_content(self, df: pd.DataFrame, text_columns: List[str]) -> Dict[str, float]:
        """Analyze actual data content using TF-IDF"""
        if not text_columns:
            return {}
            
        try:
            # Sample data for analysis
            sample_size = min(100, len(df))
            sample_df = df.head(sample_size)
            
            # Combine text from all text columns
            text_data = []
            for _, row in sample_df.iterrows():
                row_text = " ".join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                text_data.append(row_text)
            
            combined_text = " ".join(text_data).lower()
            self.logger.log_info(f"Analyzing content from {len(text_data)} rows, {len(combined_text)} characters")
            
            if len(combined_text.strip()) < 10:
                self.logger.log_warning("Insufficient text content for analysis")
                return {}
            
            domain_scores = {}
            for domain_name in self.domains:
                keywords = self._get_domain_keywords(domain_name)
                if not keywords:
                    continue
                
                domain_text = " ".join(keywords).lower()
                
                try:
                    # Use TF-IDF similarity
                    vectorizer = TfidfVectorizer(
                        max_features=1000,
                        stop_words='english',
                        ngram_range=(1, 2)
                    )
                    
                    # Fit on both texts
                    tfidf_matrix = vectorizer.fit_transform([combined_text, domain_text])
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    domain_scores[domain_name] = similarity
                    
                except Exception as e:
                    self.logger.log_error(f"TF-IDF analysis failed for domain '{domain_name}': {e}")
                    
                    # Fallback: simple keyword matching
                    keyword_matches = sum(1 for keyword in keywords if keyword.lower() in combined_text)
                    domain_scores[domain_name] = keyword_matches / len(keywords) if keywords else 0
            
            return domain_scores
            
        except Exception as e:
            self.logger.log_error(f"Content analysis failed: {e}", exc_info=True)
            return {}

    def _suggest_column_mappings(self, df_cols, domain_terms):
        suggestions = {}
        for col in df_cols:
            best_match = None
            best_score = 0
            for term in domain_terms:
                score = fuzz.token_set_ratio(col.lower(), term.lower())
                if score > best_score:
                    best_score = score
                    best_match = term
            if best_score >= 70 and best_match != col:
                suggestions[col] = best_match
                self.logger.log_info(f"Suggest mapping: '{col}' -> '{best_match}' (score={best_score})")
        return suggestions

    def _fallback_generic_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced fallback with better generic detection"""
        self.logger.log_info("Falling back to enhanced generic domain detection.")
        
        # Try generic config first
        if 'generic' in self.domain_configs:
            try:
                generic_result = self.domain_configs['generic'].detect_domain(df)
                if isinstance(generic_result, dict) and generic_result.get('confidence', 0) > 0:
                    return {
                        'domain': 'generic',
                        'confidence': generic_result.get('confidence', 0.0),
                        'config': self.domain_configs['generic'],
                        'features': generic_result.get('detected_features', {})
                    }
            except Exception as e:
                self.logger.log_error(f"Generic domain config failed: {e}")
        
        # Manual generic detection
        confidence = 0.2  # Base confidence for having structured data
        features = {
            'column_count': len(df.columns),
            'row_count': len(df),
            'data_types': df.dtypes.value_counts().to_dict()
        }
        
        # Boost confidence based on data structure
        if len(df.columns) > 3:
            confidence += 0.1
        if len(df) > 100:
            confidence += 0.1
        
        return {
            'domain': 'generic',
            'confidence': min(confidence, 1.0),
            'config': self.domain_configs.get('generic'),
            'features': features
        }

    def detect_domain(self, df: pd.DataFrame, text_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            self.logger.log_error(f"detect_domain expected pd.DataFrame but got {type(df)}")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        self.logger.log_info(f"Dataset columns: {df.columns.tolist()}")
        self.logger.log_info(f"Dataset shape: {df.shape}")
        
        # Log sample data more safely
        try:
            sample_str = df.head(3).to_string(max_cols=5)
            self.logger.log_info(f"Sample data:\n{sample_str}")
        except Exception as e:
            self.logger.log_warning(f"Could not log sample data: {e}")

        # Step 1: Analyze column names
        column_scores = self._analyze_column_names(df)
        self.logger.log_info(f"Column analysis scores: {column_scores}")

        # Step 2: Detect textual columns if not provided
        if text_columns is None:
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                    if unique_ratio > 0.1 or df[col].nunique() >= 5:  # More lenient criteria
                        text_columns.append(col)
            
            self.logger.log_info(f"Automatically detected text columns: {text_columns}")

        # Step 3: Analyze data content
        content_scores = {}
        if text_columns:
            content_scores = self._analyze_data_content(df, text_columns)
            self.logger.log_info(f"Content analysis scores: {content_scores}")

        # Step 4: Combine scores
        combined_scores = {}
        all_domains = set(column_scores.keys()) | set(content_scores.keys())
        
        for domain in all_domains:
            col_score = column_scores.get(domain, 0)
            content_score = content_scores.get(domain, 0)
            
            # Weighted combination: columns are more reliable than content
            combined_score = (col_score * 0.7) + (content_score * 0.3)
            combined_scores[domain] = combined_score
            
            if combined_score > 0:
                self.logger.log_info(f"Domain '{domain}': column_score={col_score:.4f}, content_score={content_score:.4f}, combined={combined_score:.4f}")

        # Step 5: Select best domain
        if not combined_scores:
            self.logger.log_warning("No domain scores calculated")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        best_domain = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[best_domain]
        config = self.domain_configs.get(best_domain)

        self.logger.log_info(f"Best domain match: '{best_domain}' with confidence: {confidence:.4f}")

        return {
            'domain': best_domain,
            'confidence': confidence,
            'config': config,
            'features': {
                'column_scores': column_scores,
                'content_scores': content_scores,
                'combined_scores': combined_scores,
                'text_columns': text_columns
            }
        }

    def detect_domain_with_fallback(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            self.logger.log_error(f"detect_domain_with_fallback expected pd.DataFrame but got {type(df)}")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        self.logger.log_info("Starting enhanced domain detection with fallback mechanism.")
        result = self.detect_domain(df)

        if result['domain'] is None or result['confidence'] < self.CONFIDENCE_THRESHOLD:
            self.logger.log_warning(f"Low confidence ({result['confidence']}) for detected domain '{result['domain']}'. Falling back to generic detection.")
            return self._fallback_generic_detection(df)

        self.logger.log_info(f"Successfully detected domain: {result['domain']} with confidence: {result['confidence']:.4f}")
        return result