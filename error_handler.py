"""
Global error handler for MESS automation package.

This module provides centralized error handling, logging, and recovery
mechanisms for the entire application.
"""

import sys
import logging
import traceback
from typing import Optional, Dict, Any, Callable, Type
from pathlib import Path
import time

try:
    from .exceptions import MESSAutomationError, ConfigurationError, FileIOError, GaussianParserError
except ImportError:
    from exceptions import MESSAutomationError, ConfigurationError, FileIOError, GaussianParserError


class ErrorHandler:
    """Global error handler for MESS automation."""
    
    def __init__(self, log_file: Optional[Path] = None, log_level: str = "INFO"):
        """
        Initialize error handler.
        
        Args:
            log_file: Optional path to log file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_file = log_file
        self.log_level = log_level
        self.error_count = 0
        self.warning_count = 0
        self.recovery_attempts = {}
        
        # Configure logging
        self._setup_logging()
        
        # Register global exception handler
        sys.excepthook = self._global_exception_handler
        
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if log file specified)
        if self.log_file:
            try:
                # Ensure log directory exists
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(detailed_formatter)
                root_logger.addHandler(file_handler)
                
                logging.info(f"Logging to file: {self.log_file}")
            except Exception as e:
                logging.error(f"Failed to setup file logging: {e}")
    
    def _global_exception_handler(self, exc_type, exc_value, exc_traceback):
        """
        Global exception handler for uncaught exceptions.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Traceback object
        """
        # Don't log KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the exception
        logger = logging.getLogger(__name__)
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Format user-friendly error message
        if isinstance(exc_value, MESSAutomationError):
            error_msg = str(exc_value)
        else:
            error_msg = f"Unexpected error: {exc_value}"
        
        print("\n" + "="*80, file=sys.stderr)
        print("❌ FATAL ERROR", file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print("\nPlease check the log file for details.", file=sys.stderr)
        print("="*80 + "\n", file=sys.stderr)
        
        self.error_count += 1
    
    def handle_error(self, 
                     error: Exception, 
                     context: Optional[Dict[str, Any]] = None,
                     severity: str = "ERROR",
                     recoverable: bool = False) -> bool:
        """
        Handle an error with context and recovery options.
        
        Args:
            error: Exception to handle
            context: Additional context information
            severity: Error severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            recoverable: Whether this error is recoverable
            
        Returns:
            True if error was handled successfully, False if it should be re-raised
        """
        # Determine if it's one of our custom exceptions
        is_custom_exception = isinstance(error, MESSAutomationError)
        
        # Get logger
        logger = logging.getLogger(__name__)
        
        # Format error message
        if is_custom_exception:
            error_msg = str(error)
            error_details = error.details
        else:
            error_msg = str(error)
            error_details = context or {}
        
        # Combine with context
        if context:
            error_details.update(context)
        
        # Log based on severity
        log_method = getattr(logger, severity.lower(), logger.error)
        
        log_message = f"{error_msg}"
        if error_details:
            log_message += f" | Details: {error_details}"
        
        if severity == "CRITICAL":
            log_method(log_message, exc_info=True)
        else:
            log_method(log_message)
        
        # Update counters
        if severity in ["ERROR", "CRITICAL"]:
            self.error_count += 1
        elif severity == "WARNING":
            self.warning_count += 1
        
        # Try recovery if applicable
        if recoverable and self._can_recover(error, error_details):
            recovery_success = self._attempt_recovery(error, error_details)
            if recovery_success:
                logger.info(f"Recovered from error: {error_msg}")
                return True
        
        # Return False for non-recoverable errors or failed recovery
        return False
    
    def _can_recover(self, error: Exception, details: Dict[str, Any]) -> bool:
        """
        Determine if an error is recoverable.
        
        Args:
            error: Exception to check
            details: Error details
            
        Returns:
            True if error is recoverable
        """
        # Check error type
        if isinstance(error, FileNotFoundError):
            # File not found might be recoverable if it's optional
            filepath = details.get("filepath", "")
            optional_files = details.get("optional_files", [])
            return str(filepath) in optional_files
        
        elif isinstance(error, ConfigurationError):
            # Configuration errors might be recoverable with defaults
            return details.get("has_defaults", False)
        
        elif isinstance(error, GaussianParserError):
            # Parser errors might be recoverable if file can be skipped
            return details.get("skip_on_error", False)
        
        return False
    
    def _attempt_recovery(self, error: Exception, details: Dict[str, Any]) -> bool:
        """
        Attempt to recover from an error.
        
        Args:
            error: Exception to recover from
            details: Error details
            
        Returns:
            True if recovery succeeded
        """
        error_key = f"{type(error).__name__}:{str(error)}"
        
        # Limit recovery attempts per error type
        current_attempts = self.recovery_attempts.get(error_key, 0)
        if current_attempts >= 3:  # Max 3 attempts per error type
            logging.warning(f"Max recovery attempts reached for: {error_key}")
            return False
        
        self.recovery_attempts[error_key] = current_attempts + 1
        
        logger = logging.getLogger(__name__)
        logger.info(f"Attempting recovery (attempt {current_attempts + 1}) for: {error}")
        
        # Recovery strategies based on error type
        if isinstance(error, FileNotFoundError):
            return self._recover_file_not_found(error, details)
        
        elif isinstance(error, ConfigurationError):
            return self._recover_configuration_error(error, details)
        
        elif isinstance(error, GaussianParserError):
            return self._recover_parser_error(error, details)
        
        return False
    
    def _recover_file_not_found(self, error: Exception, details: Dict[str, Any]) -> bool:
        """Recover from file not found error."""
        filepath = details.get("filepath")
        default_content = details.get("default_content")
        
        if not filepath or not default_content:
            return False
        
        try:
            # Create directory if it doesn't exist
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file with default content
            with open(path, 'w', encoding='utf-8') as f:
                f.write(default_content)
            
            logging.info(f"Created missing file with defaults: {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create file {filepath}: {e}")
            return False
    
    def _recover_configuration_error(self, error: Exception, details: Dict[str, Any]) -> bool:
        """Recover from configuration error."""
        config_key = details.get("key")
        default_value = details.get("default_value")
        
        if not config_key or default_value is None:
            return False
        
        # This would typically update the config in memory
        # The actual implementation depends on the config management system
        logging.info(f"Using default value for configuration key '{config_key}': {default_value}")
        return True
    
    def _recover_parser_error(self, error: Exception, details: Dict[str, Any]) -> bool:
        """Recover from parser error."""
        filepath = details.get("filepath")
        
        if not filepath:
            return False
        
        # Mark file as skipped
        logging.warning(f"Skipping file due to parse error: {filepath}")
        return True
    
    def wrap_function(self, func: Callable, context: Optional[Dict[str, Any]] = None):
        """
        Decorator wrapper for function-level error handling.
        
        Args:
            func: Function to wrap
            context: Default context for errors
            
        Returns:
            Wrapped function
        """
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Use provided context or create default
                error_context = context or {}
                
                # Add function name to context
                error_context["function"] = func.__name__
                
                # Add args info if available
                if args:
                    error_context["args_count"] = len(args)
                    # Don't include actual args in log to avoid sensitive data
                
                # Handle the error
                handled = self.handle_error(e, error_context, recoverable=True)
                
                # Re-raise if not handled
                if not handled:
                    raise
        
        return wrapper
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get error handling statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "recovery_attempts": len(self.recovery_attempts),
            "successful_recoveries": sum(1 for v in self.recovery_attempts.values() if v > 0)
        }
    
    def reset_statistics(self):
        """Reset error statistics."""
        self.error_count = 0
        self.warning_count = 0
        self.recovery_attempts = {}


# Global error handler instance
_global_handler: Optional[ErrorHandler] = None


def setup_global_error_handler(log_file: Optional[Path] = None, log_level: str = "INFO") -> ErrorHandler:
    """
    Setup global error handler.
    
    Args:
        log_file: Optional path to log file
        log_level: Logging level
        
    Returns:
        Global error handler instance
    """
    global _global_handler
    
    if _global_handler is None:
        _global_handler = ErrorHandler(log_file, log_level)
    
    return _global_handler


def get_global_error_handler() -> ErrorHandler:
    """
    Get global error handler instance.
    
    Returns:
        Global error handler instance
    """
    global _global_handler
    
    if _global_handler is None:
        # Create default handler if not set up
        _global_handler = setup_global_error_handler()
    
    return _global_handler


def handle_error(error: Exception, **kwargs) -> bool:
    """
    Convenience function to handle an error using the global handler.
    
    Args:
        error: Exception to handle
        **kwargs: Additional arguments for ErrorHandler.handle_error
        
    Returns:
        True if error was handled successfully
    """
    handler = get_global_error_handler()
    return handler.handle_error(error, **kwargs)


def wrap_with_error_handler(func: Callable, **kwargs) -> Callable:
    """
    Convenience function to wrap a function with error handling.
    
    Args:
        func: Function to wrap
        **kwargs: Additional arguments for ErrorHandler.wrap_function
        
    Returns:
        Wrapped function
    """
    handler = get_global_error_handler()
    return handler.wrap_function(func, **kwargs)