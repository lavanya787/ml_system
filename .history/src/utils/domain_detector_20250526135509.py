# domaindetector.py
import pandas as pd
from typing import Dict, Any
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
        configs = {}
        for domain in self.domains:
            try:
                module = import_module(f"domain_configs.{domain}")
                configs[domain] = module.DomainConfig()
                self.logger.log_info(f"✅ Loaded domain config: {domain}")
            except Exception as e:
                self.logger.log_error(f"❌ Failed to load domain config for '{domain}': {str(e)}")
        return configs
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            results = {}
            for domain, config in self.domain_configs.items():
                result = config.detect_domain(df)
                results[domain] = result
                self.logger.log_info(f"Domain '{domain}' confidence: {result['confidence']:.2f}")

            best_domain, best_result = max(results.items(), key=lambda x: x[1]['confidence'])
            if best_result['confidence'] < 0.3:
                self.logger.log_warning("⚠️ Low confidence for all domains.")
                return {
                    'domain': None,
                    'confidence': best_result['confidence'],
                    'config': None,
                    'features': best_result.get('detected_features', {})
                }

            self.logger.log_info(f"✅ Best match: {best_domain} with confidence {best_result['confidence']:.2f}")
            return {
                'domain': best_domain,
                'confidence': best_result['confidence'],
                'config': self.domain_configs[best_domain],
                'features': best_result.get('detected_features', {})
            }

        except Exception as e:
            self.logger.log_error(f"❌ Domain detection failed: {str(e)}")
            return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}
