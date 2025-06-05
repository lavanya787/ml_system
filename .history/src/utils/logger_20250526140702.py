import logging
import os
from datetime import datetime
from typing import Optional

class Logger:
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        """Initialize logger with specified log directory and level."""
                self.logs = []

        self.log_dir = log_dir
        self.log_level = log_level
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure root logger with file and console handlers."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create a unique log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"system_{timestamp}.log")
        
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("SystemLogger")
        def log_query(self, query_type, query, dataset_info=None, response=None):
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    
    def log_info(self, message: str, domain: Optional[str] = None) -> None:
        """Log an info message."""
        logger = self.get_domain_logger(domain) if domain else self.logger
        logger.info(message)
    
    def log_error(self, message: str, domain: Optional[str] = None) -> None:
        """Log an error message."""
        logger = self.get_domain_logger(domain) if domain else self.logger
        logger.error(message)
    
    def log_warning(self, message: str, domain: Optional[str] = None) -> None:
        """Log a warning message."""
        logger = self.get_domain_logger(domain) if domain else self.logger
        logger.warning(message)
    
    def log_debug(self, message: str, domain: Optional[str] = None) -> None:
        """Log a debug message."""
        logger = self.get_domain_logger(domain) if domain else self.logger
        logger.debug(message)