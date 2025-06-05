import logging
import os
from datetime import datetime
from typing import Optional

class Logger:
    def __init__(self, log_dir='logs', log_level=logging.INFO):
        self.log_dir = log_dir
        self.log_level = log_level
        self.logs = []

        # Create base logger
        self.logger = logging.getLogger("SystemLogger")
        self.logger.setLevel(self.log_level)

        # Create stream handler
        #handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        # Setup directory and file logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure root logger with file and console handlers."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"system_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Add file handler if not already added
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            self.logger.addHandler(file_handler)

    def log_query(self, query_type, query, dataset_info=None, response=None):
        """Log a structured query interaction into internal memory."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": query_type,
            "query": query,
            "dataset_info": dataset_info,
            "response": response
        }
        self.logs.append(entry)

    def get_logs(self):
        return self.logs

    def get_domain_logger(self, domain: str) -> logging.Logger:
        """Get or create a logger for a specific domain."""
        domain_log_file = os.path.join(self.log_dir, f"{domain}.log")
        domain_logger = logging.getLogger(f"Domain_{domain}")
        
        if not domain_logger.handlers:
            domain_logger.setLevel(self.log_level)
            file_handler = logging.FileHandler(domain_log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            domain_logger.addHandler(file_handler)

        return domain_logger

    def log_info(self, message, exc_info=False):
        self.logger.info(message, exc_info=exc_info)

    def log_warning(self, message, exc_info=False):
        self.logger.warning(message, exc_info=exc_info)

    def log_error(self, message, exc_info=False):
        self.logger.error(message, exc_info=exc_info)

    def log_debug(self, message: str, domain: Optional[str] = None) -> None:
        logger = self.get_domain_logger(domain) if domain else self.logger
        logger.debug(message)
