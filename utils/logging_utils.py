import logging
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        if log_file == 'auto':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"
        else:
            log_file = log_dir / log_file
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_exception(logger, prefix="An error occurred"):
    """Log exception with traceback"""
    import traceback
    logger.error(f"{prefix}: {traceback.format_exc()}")