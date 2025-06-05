import pandas as pd
from typing import Dict, Any
from importlib import import_module
import streamlit as st
import concurrent.futures
from fuzzywuzzy import fuzz
from utils.logger import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DomainDetector:
    CONFIDENCE_THRESHOLD = 0.3  # Configurable threshold for domain acceptance

    def __init__(self, logger: Logger):
        self.logger = logger
        self.domains = [
            'customer_support', 'entertainment', 'gaming', 'legal', 'marketing',
            'logistics', 'manufacturing', 'real_estate', 'agriculture', 'energy',
            'hospitality', 'automobile', 'telecommunications', 'government',
            'food_beverage', 'it_services', 'event_management', 'insurance',
            'retail', 'hr_resources', 'generic'
        ]
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
        if 'generic' in self.domain_configs:
            self.logger.log_info("Falling back to generic domain detection.")
            try:
                generic_result = self.domain_configs['generic'].detect_domain(df)
                if not isinstance(generic_result, dict):
                    self.logger.log_warning("Generic domain detection did not return a dict result.")
                    generic_result = {}
            except Exception as e:
                self.logger.log_error(f"Error during generic domain detection fallback: {str(e)}", exc_info=True)
                generic_result = {}
            return {
                'domain': 'generic',
                'confidence': generic_result.get('confidence', 0.0),
                'config': self.domain_configs['generic'],
                'features': generic_result.get('detected_features', {})
            }
        self.logger.log_warning("No generic domain config available for fallback.")
        return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

    def detect_domain(self, df: pd.DataFrame, text_columns: list[str] = None) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            self.logger.log_error(f"detect_domain expected pd.DataFrame but got {type(df)}")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        self.logger.log_info(f"Dataset columns: {df.columns.tolist()}")
        self.logger.log_info(f"Dataset shape: {df.shape}")
        self.logger.log_info(f"Sample data:\n{df.head().to_string()}")

        # Detect textual columns if not explicitly provided
        if text_columns is None:
            text_columns = [
                col for col in df.columns
                if df[col].dtype == 'object' and df[col].nunique() >= 10
            ]
            self.logger.log_info(f"Automatically detected text columns: {text_columns}")

        if not text_columns:
            self.logger.log_error("No suitable textual columns found for domain detection.")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        try:
            sample_text_rows = df[text_columns].astype(str).fillna("").agg(" ".join, axis=1).head(50)
        sample_text = "\n".join(sample_text_rows.tolist())
        self.logger.log_info(f"Sample combined text (first 500 chars):\n{sample_text[:500]}")
    except Exception as e:
        self.logger.log_error(f"Failed to combine textual columns: {e}", exc_info=True)
        return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

    scores = {}
    for name, module in self.domain_configs.items():
        keywords = getattr(module, "DOMAIN_KEYWORDS", [])
        if not keywords:
            continue
        domain_keywords_text = " ".join(keywords)
        try:
            vectorizer = TfidfVectorizer().fit([sample_text, domain_keywords_text])
            vectors = vectorizer.transform([sample_text, domain_keywords_text])
            score = cosine_similarity(vectors[0], vectors[1])[0][0]
            scores[name] = score
        except Exception as e:
            self.logger.log_error(f"TF-IDF domain detection failed for {name}: {e}")

    if not scores:
        return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

    best_domain, confidence = max(scores.items(), key=lambda x: x[1])
    config = self.domain_configs.get(best_domain)

    self.logger.log_info(f"Best domain match: '{best_domain}' with confidence: {confidence:.4f}")

    return {
        'domain': best_domain,
        'confidence': confidence,
        'config': config,
        'features': {'scores': scores}
    }


    def detect_domain_with_fallback(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            self.logger.log_error(f"detect_domain_with_fallback expected pd.DataFrame but got {type(df)}")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        self.logger.log_info("Starting domain detection with fallback mechanism.")
        result = self.detect_domain(df)

        if result['domain'] is None or result['confidence'] < self.CONFIDENCE_THRESHOLD:
            self.logger.log_warning(f"Low confidence ({result['confidence']}) for detected domain '{result['domain']}'. Falling back to generic detection.")
            return self._fallback_generic_detection(df)

        self.logger.log_info(f"Detected domain: {result['domain']} with confidence: {result['confidence']}")
        return result
    