import logging
import os
import sys

def setup_logger(name: str = "B2R", save_dir: str = None, level: int = logging.INFO):
    """
    Set up a logger that outputs to both console and (optionally) a file.
    
    Args:
        name (str): Logger name.
        save_dir (str): Directory to save log file. If None, no file log will be created.
        level (int): Logging level.
        
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{name}.log")
        file_handler = logging.FileHandler(file_path, mode="w")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
