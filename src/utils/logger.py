"""
Modular logging utility for Sign Language Recognition project.
Provides consistent logging across all scripts with file and console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ProjectLogger:
    """
    Centralized logging system for the project.
    Creates both file and console handlers with consistent formatting.
    """
    
    _loggers = {}  # Cache loggers to avoid duplicates
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        log_dir: str = "logs",
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        console_output: bool = True
    ) -> logging.Logger:
        """
        Get or create a logger with file and optional console output.
        
        Args:
            name: Name of the logger (typically __name__ of the calling module)
            log_dir: Directory to store log files
            log_file: Specific log file name. If None, uses name + timestamp
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Whether to also output to console
            
        Returns:
            Configured logger instance
        """
        # Return cached logger if exists
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # Prevent duplicate logs
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Generate log file name
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            module_name = name.split('.')[-1]  # Get last part of module name
            log_file = f"{module_name}_{timestamp}.log"
        
        file_path = log_path / log_file
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            fmt='%(levelname)s: %(message)s'
        )
        
        # File handler (detailed)
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Console handler (simplified)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # Cache logger
        cls._loggers[name] = logger
        
        logger.info(f"Logger initialized. Logging to: {file_path}")
        
        return logger
    
    @classmethod
    def get_simple_logger(
        cls,
        script_name: str,
        log_dir: str = "logs"
    ) -> logging.Logger:
        """
        Convenience method to get a logger with sensible defaults.
        
        Args:
            script_name: Name of the script (e.g., 'video2npy', 'train')
            log_dir: Directory to store log files
            
        Returns:
            Configured logger instance
        """
        return cls.get_logger(
            name=script_name,
            log_dir=log_dir,
            level=logging.INFO,
            console_output=True
        )


class ProgressLogger:
    """
    Helper class for logging progress in batch operations.
    """
    
    def __init__(self, logger: logging.Logger, total: int, desc: str = "Processing"):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance to use
            total: Total number of items to process
            desc: Description of the operation
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
        
    def update(self, n: int = 1, msg: Optional[str] = None):
        """
        Update progress.
        
        Args:
            n: Number of items completed
            msg: Optional message to log
        """
        self.current += n
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        items_per_sec = self.current / elapsed if elapsed > 0 else 0
        eta_seconds = (self.total - self.current) / items_per_sec if items_per_sec > 0 else 0
        
        status = f"{self.desc}: {self.current}/{self.total} ({percentage:.1f}%) - "
        status += f"{items_per_sec:.2f} items/s - ETA: {eta_seconds:.0f}s"
        
        if msg:
            status += f" | {msg}"
            
        self.logger.info(status)
    
    def finish(self, msg: Optional[str] = None):
        """
        Log completion.
        
        Args:
            msg: Optional completion message
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        completion_msg = f"{self.desc} completed: {self.current}/{self.total} in {elapsed:.2f}s"
        
        if msg:
            completion_msg += f" | {msg}"
            
        self.logger.info(completion_msg)


def log_system_info(logger: logging.Logger):
    """
    Log system information (Python version, platform, etc.)
    
    Args:
        logger: Logger instance to use
    """
    import platform
    import torch
    
    logger.info("=" * 70)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 70)
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 70)


def log_arguments(logger: logging.Logger, args):
    """
    Log script arguments in a formatted way.
    
    Args:
        logger: Logger instance to use
        args: Arguments namespace from argparse
    """
    logger.info("=" * 70)
    logger.info("SCRIPT ARGUMENTS")
    logger.info("=" * 70)
    
    if hasattr(args, '__dict__'):
        for key, value in sorted(vars(args).items()):
            logger.info(f"  {key}: {value}")
    else:
        logger.info(f"  {args}")
    
    logger.info("=" * 70)


# Convenience function for quick setup
def setup_logger(script_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Quick setup function for getting a logger.
    
    Args:
        script_name: Name of the script
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    return ProjectLogger.get_simple_logger(script_name, log_dir)


if __name__ == "__main__":
    # Example usage
    logger = setup_logger("example_script")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Progress logging example
    progress = ProgressLogger(logger, total=100, desc="Processing videos")
    for i in range(100):
        # Simulate work
        import time
        time.sleep(0.01)
        progress.update(1, msg=f"Item {i+1}")
    progress.finish()
