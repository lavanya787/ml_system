import pandas as pd
from typing import Dict, Any, Optional, Tuple
from importlib import import_module
import streamlit as st
import concurrent.futures
from fuzzywuzzy import fuzz
from utils.logger import Logger

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
            
                self.logger.log_info(f"Loaded domain config for '{domain}' - type: {type(domain_config_instance)}")

                # Debug: log the type of loaded config
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
                # Use token_set_ratio for better fuzzy matching of multi-word terms
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

    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Added type check
        if not isinstance(df, pd.DataFrame):
            self.logger.log_error(f"detect_domain expected pd.DataFrame but got {type(df)}")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        self.logger.log_info(f"Dataset columns: {df.columns.tolist()}")
        self.logger.log_info(f"Dataset shape: {df.shape}")
        self.logger.log_info(f"Sample data:\n{df.head().to_string()}")

        if not self.domain_configs:
            self.logger.log_error("No domain configs available to detect domain.")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        results = {}

    def detect_single_domain(domain_config_tuple: Tuple[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        domain, config = domain_config_tuple
        try:
            # Added method existence check
            if not hasattr(config, 'detect_domain'):
                self.logger.log_error(f"Domain config '{domain}' missing 'detect_domain' method.")
                return None

            result = config.detect_domain(df)
                # Added return type check
            if not isinstance(result, dict):
                self.logger.log_warning(f"Domain '{domain}' detection returned non-dict result: {type(result)}")
                return None
            confidence = result.get('confidence')
            if confidence is None or not isinstance(confidence, (float, int)):
                self.logger.log_warning(f"Invalid or missing confidence from domain '{domain}'. Skipping.")
                return None
            self.logger.log_info(f"Domain '{domain}' confidence: {confidence:.2f}")
            return (domain, result)
    except Exception as e:
            self.logger.log_error(f"Error during domain '{domain}' detection: {str(e)}", exc_info=True)                return None

        # Parallel domain detection
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(detect_single_domain, item): item[0] for item in self.domain_configs.items()}
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                except Exception as e:
                    self.logger.log_error(f"Exception in future result: {str(e)}", exc_info=True)
                    res = None
                if res:
                    domain, result = res
                    results[domain] = result

        if not results:
            self.logger.log_warning("No valid detection results returned by any domain configs.")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        # Summarize confidence scores
        confidence_summary = {k: v['confidence'] for k, v in results.items()}
        self.logger.log_info(f"Detection results summary: {confidence_summary}")

        # Get the domain with highest confidence
        best_domain, best_result = max(results.items(), key=lambda x: x[1]['confidence'])

        # Check confidence threshold
        if best_result['confidence'] < self.CONFIDENCE_THRESHOLD:
            self.logger.log_warning(f"Low confidence ({best_result['confidence']:.2f}) for all domains.")

            # Aggregate all domain terms for suggestions
            all_domain_terms = set()
            for config in self.domain_configs.values():
                domain_terms = getattr(config, 'domain_terms', [])
                all_domain_terms.update(domain_terms)

            # Suggest column mappings based on fuzzy matching
            suggested_mappings = self._suggest_column_mappings(df.columns, all_domain_terms)

            # Show UI in Streamlit for user to confirm/override mappings
            st.warning("⚠️ Low confidence in domain detection. Please map your dataset columns to domain-specific terms below:")
            column_mappings = {}
            for col in df.columns:
                default_val = suggested_mappings.get(col, col)
                try:  # Added exception handling around user input
                    mapped_term = st.text_input(f"Map column '{col}' to:", key=f"map_{col}_{id(df)}", value=default_val)
                except Exception as e:
                    self.logger.log_error(f"Error during Streamlit input for column '{col}': {str(e)}", exc_info=True)
                    mapped_term = col
                if mapped_term and mapped_term != col:
                    column_mappings[col] = mapped_term

            if column_mappings:
                self.logger.log_info(f"User confirmed column mappings: {column_mappings}")
                renamed_df = df.rename(columns=column_mappings)
                # Recursive call to detect domain again with renamed columns
                return self.detect_domain(renamed_df)

            # If user declines to remap or no mappings, fallback to generic
            return self._fallback_generic_detection(df)

        # Confident detection
        self.logger.log_info(f"Best domain match: '{best_domain}' with confidence {best_result['confidence']:.2f}")
        return {
            'domain': best_domain,
            'confidence': best_result['confidence'],
            'config': self.domain_configs[best_domain],
            'features': best_result.get('detected_features', {})
        }
