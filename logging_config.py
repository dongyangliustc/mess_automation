"""
Logging configuration for MESS automation package.

This module provides centralized logging configuration with different
verbosity levels, log rotation, and structured log formats.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any

# Log levels mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"


class StructuredLogger:
    """Logger that supports structured logging with context."""
    
    def __init__(self, name: str):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def add_context(self, **kwargs):
        """
        Add context to log messages.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context."""
        self.context.clear()
    
    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """
        Format message with context.
        
        Args:
            message: Base message
            extra: Extra context for this log entry
            
        Returns:
            Formatted message
        """
        if not self.context and not extra:
            return message
        
        all_context = self.context.copy()
        if extra:
            all_context.update(extra)
        
        context_str = " | ".join(f"{k}={v}" for k, v in all_context.items())
        return f"{message} | {context_str}"
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if self.logger.isEnabledFor(logging.WARNING):
            self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        if self.logger.isEnabledFor(logging.ERROR):
            self.logger.error(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        if self.logger.isEnabledFor(logging.CRITICAL):
            self.logger.critical(self._format_message(message, kwargs))
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        if self.logger.isEnabledFor(logging.ERROR):
            self.logger.exception(self._format_message(message, kwargs))


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    verbose: bool = False
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, log to console only)
        log_dir: Directory for log files (used if log_file is None)
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        verbose: If True, use detailed format
        
    Returns:
        Root logger
    """
    # Get log level
    level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    format_str = DETAILED_FORMAT if verbose else DEFAULT_FORMAT
    formatter = logging.Formatter(format_str)
    
    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file or log_dir:
        try:
            if log_file:
                log_path = Path(log_file)
            elif log_dir:
                # Create log file with timestamp
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = get_timestamp()
                log_path = log_dir / f"mess_automation_{timestamp}.log"
            else:
                raise ValueError("Either log_file or log_dir must be provided")
            
            # Ensure log directory exists
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            root_logger.info(f"Logging to file: {log_path}")
            
        except Exception as e:
            root_logger.error(f"Failed to setup file logging: {e}")
    
    # Log startup information
    root_logger.info(f"Logging initialized at level: {log_level}")
    root_logger.info(f"Python version: {sys.version}")
    root_logger.info(f"Working directory: {Path.cwd()}")
    
    return root_logger


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


def get_timestamp() -> str:
    """
    Get current timestamp for log file names.
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_function_call(func):
    """
    Decorator to log function calls with arguments and timing.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        arg_str = ", ".join([str(arg) for arg in args])
        kwarg_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        call_info = f"{func.__name__}({arg_str}{', ' if arg_str and kwarg_str else ''}{kwarg_str})"
        
        logger.debug(f"Calling {call_info}")
        
        # Time the function
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log success
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"{func.__name__} failed after {execution_time:.3f}s",
                error=str(e),
                error_type=type(e).__name__
            )
            
            raise
    
    return wrapper


def log_class_methods(cls):
    """
    Class decorator to log all public method calls.
    
    Args:
        cls: Class to decorate
        
    Returns:
        Decorated class
    """
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("_"):
            # Skip special methods
            if attr_name not in ["__init__", "__str__", "__repr__"]:
                setattr(cls, attr_name, log_function_call(attr_value))
    
    return cls