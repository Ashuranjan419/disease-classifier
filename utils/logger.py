"""
Logging utilities for training.
"""

import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOGS_DIR


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logs directory if needed
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(LOGS_DIR, f'{name}_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Logger specifically for training metrics."""
    
    def __init__(self, experiment_name='experiment'):
        self.experiment_name = experiment_name
        self.logger = setup_logger(experiment_name)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """Log metrics for an epoch."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(lr)
        
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f}"
        )
    
    def log_info(self, message):
        """Log general info."""
        self.logger.info(message)
    
    def log_config(self, config_dict):
        """Log configuration."""
        self.logger.info("=" * 50)
        self.logger.info("Configuration:")
        for key, value in config_dict.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)
    
    def get_history(self):
        """Return training history."""
        return self.history
