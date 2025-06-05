import pandas as pd
from typing import Dict, Any
from importlib import import_module
import streamlit as st
import concurrent.futures
from fuzzywuzzy import fuzz
from utils.logger import Logger

class DomainDetector:
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
                configs[domain] = module.DomainConfig(logger=self.logger)
                self.logger.log_info(f"Successfully loaded domain config: {domain}")
            except ModuleNotFoundError as e:
                self.logger.log_error(f"Module not found for domain '{domain}': {str(e)}")
            except TypeError as e:
                self.logger.log_error(f"Type error for domain '{domain}': {str(e)}")
            except Exception as e:
                self.logger.log_error(f"Unexpected error loading domain '{domain}': {str(e)}")
        
        if not configs:
            self.logger.log_error("No domain configs were loaded! Domain detection cannot proceed.")
        else:
            self.logger.log_info(f"Loaded {len(configs)} domain configs: {list(configs.keys())}")
        
        return configs
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.logger.log_info(f"Dataset columns: {df.columns.tolist()}")
        self.logger.log_info(f"Dataset shape: {df.shape}")
        self.logger.log_info(f"Sample data:\n{df.head().to_string()}")

        if not self.domain_configs:
            self.logger.log_error("No domain configs available to detect domain.")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

        try:
            results = {}
            for domain, config in self.domain_configs.items():
                try:
                    result = config.detect_domain(df)
                    if 'confidence' not in result or not isinstance(result['confidence'], (float, int)):
                        self.logger.log_warning(f"Invalid confidence value returned by domain '{domain}'. Skipping.")
                        continue
                    results[domain] = result
                    self.logger.log_info(
                        f"Domain '{domain}' confidence: {result['confidence']:.2f}, "
                        f"features: {result.get('detected_features', {})}"
                    )
                except Exception as e:
                    self.logger.log_error(f"Error during domain '{domain}' detection: {str(e)}")

            if not results:
                self.logger.log_warning("No valid detection results returned by any domain configs.")
                return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

            confidence_summary = {k: v['confidence'] for k, v in results.items()}
            self.logger.log_info(f"Detection results summary: {confidence_summary}")

            best_domain, best_result = max(results.items(), key=lambda x: x[1]['confidence'])

            if best_result['confidence'] < 0.3:
                self.logger.log_warning(f"Low confidence ({best_result['confidence']:.2f}) for all domains.")
                
                column_mappings = {}
                st.write("Low confidence in domain detection. Please map dataset columns to domain-specific terms:")
                for col in df.columns:
                    mapped_term = st.text_input(f"Map column '{col}' to:", key=f"map_{col}_{id(df)}", value=col)
                    if mapped_term and mapped_term != col:
                        column_mappings[col] = mapped_term
                
                if column_mappings:
                    self.logger.log_info(f"Applying column mappings: {column_mappings}")
                    renamed_df = df.rename(columns=column_mappings)
                    return self.detect_domain(renamed_df)
                
                if 'generic' in self.domain_configs:
                    self.logger.log_info("Falling back to generic domain.")
                    generic_result = self.domain_configs['generic'].detect_domain(df)
                    return {
                        'domain': 'generic',
                        'confidence': generic_result['confidence'],
                        'config': self.domain_configs['generic'],
                        'features': generic_result.get('detected_features', {})
                    }
                
                return {
                    'domain': None,
                    'confidence': best_result['confidence'],
                    'config': None,
                    'features': best_result.get('detected_features', {})
                }

            self.logger.log_info(f"Best domain match: '{best_domain}' with confidence {best_result['confidence']:.2f}")
            return {
                'domain': best_domain,
                'confidence': best_result['confidence'],
                'config': self.domain_configs[best_domain],
                'features': best_result.get('detected_features', {})
            }

        except Exception as e:
            self.logger.log_error(f"Domain detection failed unexpectedly: {str(e)}")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}