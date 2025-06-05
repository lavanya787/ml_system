import pandas as pd
from typing import Dict, Any
from importlib import import_module
from utils.logger import Logger

class DomainDetector:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.domains = [
            'customer_support', 'entertainment', 'gaming', 'legal', 'marketing',
            'logistics', 'manufacturing', 'real_estate', 'agriculture', 'energy',
            'hospitality', 'automobile', 'telecommunications', 'government',
            'food_beverage', 'it_services', 'event_management', 'insurance',
            'retail', 'hr_resources'
        ]
        self.domain_configs = self._load_domain_configs()

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
                    self.logger.log_info(f"Domain '{domain}' confidence: {result['confidence']:.2f}")
                except Exception as e:
                    self.logger.log_error(f"Error during domain '{domain}' detection: {str(e)}")

            if not results:
                self.logger.log_warning("No valid detection results returned by any domain configs.")
                return {'domain': None, 'confidence': 0.0, 'config': None, 'features': {}}

            self.logger.log_info(f"Detection results summary: {{k: v['confidence'] for k, v in results.items()}}")

            best_domain, best_result = max(results.items(), key=lambda x: x[1]['confidence'])

            if best_result['confidence'] < 0.3:
                self.logger.log_warning(f"Low confidence ({best_result['confidence']:.2f}) for all domains.")
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