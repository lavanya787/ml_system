import pandas as pd
from typing import Dict, Any, Optional
from utils.logger import Logger
from importlib import import_module

class DomainDetector:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.domains = [
            'customer_support', 'entertainment', 'gaming', 'legal', 'marketing',
            'logistics', 'manufacturing', 'real_estate', 'agriculture', 'energy',
            'hospitality', 'automobile', 'telecommunications', 'government',
            'food_beverage', 'it_services', 'event_management', 'insurance',
            'retail', 'hr_resources', 'banking'
        ]
        self.domain_configs = self._load_domain_configs()
    
    def _load_domain_configs(self) -> Dict[str, Any]:
        """Load all domain configuration classes."""
        configs = {}
        for domain in self.domains:
            try:
                module = import_module(f"domain_configs.{domain}")
                configs[domain] = module.DomainConfig()
                self.logger.log_info(f"Loaded domain config: {domain}")
            except Exception as e:
                self.logger.log_error(f"Failed to load domain config {domain}: {str(e)}")
        return configs
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect the most likely domain for the dataset."""
        try:
            results = {}
            for domain, config in self.domain_configs.items():
                result = config.detect_domain(df)
                results[domain] = result
                self.logger.log_info(f"Domain {domain} confidence: {result['confidence']}")
            
            # Select domain with highest confidence
            best_domain = max(results.items(), key=lambda x: x[1]['confidence'])
            best_domain_name, best_result = best_domain
            
            if best_result['confidence'] < 0.3:
                self.logger.log_warning("No domain detected with sufficient confidence")
                return {
                    'domain': None,
                    'confidence': best_result['confidence'],
                    'config': None,
                    'features': best_result['detected_features']
                }
            
            self.logger.log_info(f"Detected domain: {best_domain_name} with confidence {best_result['confidence']}")
            return {
                'domain': best_domain_name,
                'confidence': best_result['confidence'],
                'config': self.domain_configs[best_domain_name],
                'features': best_result['detected_features']
            }
        except Exception as e:
            self.logger.log_error(f"Domain detection failed: {str(e)}")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}