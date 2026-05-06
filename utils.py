"""
Utility functions for MESS automation package.

This module provides common utility functions for error handling,
validation, file operations, and other shared functionality.
"""

import os
import sys
import json
import yaml
import logging
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime
import traceback

from .exceptions import MESSAutomationError, ConfigurationError, FileIOError
from .error_handler import handle_error


def validate_file_path(filepath: Union[str, Path], 
                      must_exist: bool = True,
                      must_be_file: bool = True,
                      must_be_readable: bool = True) -> Tuple[bool, str]:
    """
    Validate file path.
    
    Args:
        filepath: Path to validate
        must_exist: If True, file must exist
        must_be_file: If True, path must be a file (not directory)
        must_be_readable: If True, file must be readable
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(filepath).resolve()
        
        if must_exist and not path.exists():
            return False, f"File does not exist: {path}"
        
        if must_be_file and path.exists() and not path.is_file():
            return False, f"Path is not a file: {path}"
        
        if must_be_readable and path.exists() and not os.access(path, os.R_OK):
            return False, f"File is not readable: {path}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error validating path {filepath}: {e}"


def validate_directory_path(dirpath: Union[str, Path],
                          must_exist: bool = True,
                          must_be_writable: bool = False) -> Tuple[bool, str]:
    """
    Validate directory path.
    
    Args:
        dirpath: Directory path to validate
        must_exist: If True, directory must exist
        must_be_writable: If True, directory must be writable
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(dirpath).resolve()
        
        if must_exist and not path.exists():
            return False, f"Directory does not exist: {path}"
        
        if path.exists() and not path.is_dir():
            return False, f"Path is not a directory: {path}"
        
        if must_be_writable and path.exists() and not os.access(path, os.W_OK):
            return False, f"Directory is not writable: {path}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error validating directory {dirpath}: {e}"


def safe_read_file(filepath: Union[str, Path], 
                  encoding: str = 'utf-8',
                  errors: str = 'ignore') -> Tuple[Optional[str], Optional[str]]:
    """
    Safely read file with error handling.
    
    Args:
        filepath: Path to file
        encoding: File encoding
        errors: Error handling for encoding
        
    Returns:
        Tuple of (content, error_message)
    """
    try:
        path = Path(filepath)
        
        # Validate file first
        is_valid, error_msg = validate_file_path(path, must_exist=True, must_be_file=True)
        if not is_valid:
            return None, error_msg
        
        # Read file
        with open(path, 'r', encoding=encoding, errors=errors) as f:
            content = f.read()
        
        return content, None
        
    except Exception as e:
        return None, f"Error reading file {filepath}: {e}"


def safe_write_file(filepath: Union[str, Path], 
                   content: str,
                   encoding: str = 'utf-8',
                   backup: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Safely write file with error handling and optional backup.
    
    Args:
        filepath: Path to file
        content: Content to write
        encoding: File encoding
        backup: If True, create backup of existing file
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        path = Path(filepath)
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file if requested
        if backup and path.exists():
            backup_path = path.with_suffix(f"{path.suffix}.backup")
            shutil.copy2(path, backup_path)
        
        # Write file
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        
        return True, None
        
    except Exception as e:
        return False, f"Error writing file {filepath}: {e}"


def safe_parse_yaml(content: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Safely parse YAML content with error handling.
    
    Args:
        content: YAML content as string
        
    Returns:
        Tuple of (parsed_dict, error_message)
    """
    try:
        parsed = yaml.safe_load(content)
        
        # Ensure result is a dictionary
        if not isinstance(parsed, dict):
            return None, f"YAML content is not a dictionary: {type(parsed)}"
        
        return parsed, None
        
    except yaml.YAMLError as e:
        return None, f"YAML parsing error: {e}"
    except Exception as e:
        return None, f"Error parsing YAML: {e}"


def safe_parse_json(content: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Safely parse JSON content with error handling.
    
    Args:
        content: JSON content as string
        
    Returns:
        Tuple of (parsed_dict, error_message)
    """
    try:
        parsed = json.loads(content)
        
        # Ensure result is a dictionary
        if not isinstance(parsed, dict):
            return None, f"JSON content is not a dictionary: {type(parsed)}"
        
        return parsed, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"
    except Exception as e:
        return None, f"Error parsing JSON: {e}"


def calculate_file_hash(filepath: Union[str, Path], 
                       algorithm: str = 'md5',
                       chunk_size: int = 8192) -> Tuple[Optional[str], Optional[str]]:
    """
    Calculate file hash for integrity checking.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        chunk_size: Chunk size for reading large files
        
    Returns:
        Tuple of (hash_value, error_message)
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            return None, f"File does not exist: {path}"
        
        # Select hash algorithm
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            return None, f"Unsupported hash algorithm: {algorithm}"
        
        # Calculate hash
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        
        return hasher.hexdigest(), None
        
    except Exception as e:
        return None, f"Error calculating file hash: {e}"


def create_backup(filepath: Union[str, Path],
                 backup_dir: Optional[Union[str, Path]] = None,
                 timestamp: bool = True) -> Tuple[Optional[Path], Optional[str]]:
    """
    Create backup of a file.
    
    Args:
        filepath: Path to file to backup
        backup_dir: Backup directory (default: same directory as file)
        timestamp: If True, add timestamp to backup filename
        
    Returns:
        Tuple of (backup_path, error_message)
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            return None, f"File does not exist: {path}"
        
        # Determine backup directory
        if backup_dir:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
        else:
            backup_path = path.parent
        
        # Create backup filename
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{path.stem}_{timestamp_str}{path.suffix}"
        else:
            backup_name = f"{path.stem}.backup{path.suffix}"
        
        backup_file = backup_path / backup_name
        
        # Copy file
        shutil.copy2(path, backup_file)
        
        return backup_file, None
        
    except Exception as e:
        return None, f"Error creating backup: {e}"


def retry_operation(operation: Callable,
                   max_attempts: int = 3,
                   delay: float = 1.0,
                   exponential_backoff: bool = True,
                   retry_on: Optional[List[type]] = None) -> Any:
    """
    Retry an operation on failure.
    
    Args:
        operation: Function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        exponential_backoff: If True, delay doubles after each attempt
        retry_on: List of exception types to retry on (None = all exceptions)
        
    Returns:
        Operation result
        
    Raises:
        Exception: If all attempts fail
    """
    logger = logging.getLogger(__name__)
    
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
            
        except Exception as e:
            # Check if we should retry this exception
            if retry_on and not any(isinstance(e, exc_type) for exc_type in retry_on):
                raise
            
            if attempt < max_attempts:
                # Calculate delay
                current_delay = delay * (2 ** (attempt - 1)) if exponential_backoff else delay
                
                logger.warning(
                    f"Operation failed (attempt {attempt}/{max_attempts}): {e}. "
                    f"Retrying in {current_delay:.1f}s..."
                )
                
                import time
                time.sleep(current_delay)
            else:
                logger.error(f"Operation failed after {max_attempts} attempts: {e}")
                raise


def with_timeout(timeout: float, operation: Callable, default: Any = None) -> Any:
    """
    Execute operation with timeout.
    
    Args:
        timeout: Timeout in seconds
        operation: Function to execute
        default: Value to return on timeout
        
    Returns:
        Operation result or default value on timeout
    """
    import threading
    import queue
    
    def worker(result_queue):
        try:
            result = operation()
            result_queue.put(result)
        except Exception as e:
            result_queue.put(e)
    
    result_queue = queue.Queue()
    thread = threading.Thread(target=worker, args=(result_queue,))
    thread.daemon = True
    thread.start()
    
    try:
        result = result_queue.get(timeout=timeout)
        if isinstance(result, Exception):
            raise result
        return result
    except queue.Empty:
        logger = logging.getLogger(__name__)
        logger.warning(f"Operation timed out after {timeout}s")
        return default


def format_error_details(error: Exception, include_traceback: bool = True) -> Dict[str, Any]:
    """
    Format error details for logging or reporting.
    
    Args:
        error: Exception to format
        include_traceback: If True, include formatted traceback
        
    Returns:
        Dictionary with error details
    """
    error_details = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_module": error.__class__.__module__,
        "timestamp": datetime.now().isoformat()
    }
    
    if include_traceback:
        error_details["traceback"] = traceback.format_exc()
    
    # Add additional details for custom exceptions
    if hasattr(error, 'details'):
        error_details["additional_details"] = getattr(error, 'details', {})
    
    return error_details


def cleanup_temp_files(temp_dir: Optional[Union[str, Path]] = None,
                      pattern: str = "mess_temp_*",
                      max_age_hours: float = 24.0) -> Tuple[int, Optional[str]]:
    """
    Clean up temporary files.
    
    Args:
        temp_dir: Temporary directory (default: system temp)
        pattern: File pattern to match
        max_age_hours: Maximum age of files to keep (hours)
        
    Returns:
        Tuple of (files_deleted, error_message)
    """
    try:
        if temp_dir:
            base_dir = Path(temp_dir)
        else:
            base_dir = Path(tempfile.gettempdir())
        
        if not base_dir.exists():
            return 0, f"Directory does not exist: {base_dir}"
        
        files_deleted = 0
        max_age_seconds = max_age_hours * 3600
        current_time = datetime.now().timestamp()
        
        for temp_file in base_dir.glob(pattern):
            try:
                # Check file age
                file_age = current_time - temp_file.stat().st_mtime
                
                if file_age > max_age_seconds:
                    if temp_file.is_file():
                        temp_file.unlink()
                        files_deleted += 1
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                        files_deleted += 1
                        
            except Exception as e:
                # Log but continue with other files
                logging.getLogger(__name__).warning(f"Error cleaning up {temp_file}: {e}")
        
        return files_deleted, None
        
    except Exception as e:
        return 0, f"Error cleaning up temporary files: {e}"


def validate_numeric_range(value: Any, 
                          min_value: Optional[float] = None,
                          max_value: Optional[float] = None,
                          name: str = "value") -> Tuple[bool, str]:
    """
    Validate numeric value is within range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Value name for error message
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        numeric_value = float(value)
        
        if min_value is not None and numeric_value < min_value:
            return False, f"{name} ({numeric_value}) is below minimum ({min_value})"
        
        if max_value is not None and numeric_value > max_value:
            return False, f"{name} ({numeric_value}) is above maximum ({max_value})"
        
        return True, ""
        
    except (ValueError, TypeError):
        return False, f"{name} is not a valid number: {value}"