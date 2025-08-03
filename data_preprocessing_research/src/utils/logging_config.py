"""
Logging configuration for the research framework.
"""

import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "experiments/results/logs",
    console_output: bool = True
):
    """
    Setup logging configuration for the research framework.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files
        console_output: Whether to output logs to console
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"preprocessing_research_{timestamp}.log"
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'filename': str(log_file),
                'mode': 'w',
                'formatter': 'detailed',
                'level': log_level
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['file'],
                'level': log_level,
                'propagate': False
            }
        }
    }
    
    # Add console handler if requested
    if console_output:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': log_level
        }
        config['loggers']['']['handlers'].append('console')
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    logger.info(f"Log level: {log_level}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
