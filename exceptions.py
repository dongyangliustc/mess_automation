"""
Custom exceptions for MESS automation package.

This module defines custom exception classes for better error handling
and more informative error messages.
"""

import traceback
from typing import Optional, Dict, Any, List
from pathlib import Path


class MESSAutomationError(Exception):
    """Base exception for all MESS automation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Human-readable error message
            details: Additional error details (e.g., file paths, values)
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.traceback = traceback.format_exc()
        
    def __str__(self):
        """Return formatted error string."""
        parts = [f"❌ {self.message}"]
        
        if self.details:
            parts.append("\nDetails:")
            for key, value in self.details.items():
                parts.append(f"  • {key}: {value}")
                
        return "\n".join(parts)


# ============================================================================
# Configuration errors
# ============================================================================

class ConfigurationError(MESSAutomationError):
    """Configuration file related errors."""
    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Configuration file not found."""
    
    def __init__(self, config_path: Path):
        super().__init__(
            f"Configuration file not found: {config_path}",
            details={"config_path": str(config_path)}
        )


class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""
    
    def __init__(self, errors: List[str], config_path: Optional[Path] = None):
        details = {"validation_errors": errors}
        if config_path:
            details["config_path"] = str(config_path)
        
        super().__init__(
            f"Configuration validation failed with {len(errors)} error(s)",
            details=details
        )


class MissingConfigKeyError(ConfigurationError):
    """Required configuration key is missing."""
    
    def __init__(self, key: str, config_path: Optional[Path] = None):
        details = {"missing_key": key}
        if config_path:
            details["config_path"] = str(config_path)
        
        super().__init__(
            f"Missing required configuration key: '{key}'",
            details=details
        )


class InvalidConfigValueError(ConfigurationError):
    """Configuration value is invalid."""
    
    def __init__(self, key: str, value: Any, expected: str):
        super().__init__(
            f"Invalid value for '{key}': {value} (expected: {expected})",
            details={"key": key, "value": value, "expected": expected}
        )


# ============================================================================
# File and I/O errors
# ============================================================================

class FileIOError(MESSAutomationError):
    """File input/output related errors."""
    pass


class FileNotFoundError(FileIOError):
    """File not found."""
    
    def __init__(self, filepath: Path):
        super().__init__(
            f"File not found: {filepath}",
            details={"filepath": str(filepath)}
        )


class FileReadError(FileIOError):
    """Error reading file."""
    
    def __init__(self, filepath: Path, error: str):
        super().__init__(
            f"Cannot read file: {filepath}",
            details={"filepath": str(filepath), "error": str(error)}
        )


class FileWriteError(FileIOError):
    """Error writing file."""
    
    def __init__(self, filepath: Path, error: str):
        super().__init__(
            f"Cannot write file: {filepath}",
            details={"filepath": str(filepath), "error": str(error)}
        )


class FileFormatError(FileIOError):
    """File format is incorrect or unsupported."""
    
    def __init__(self, filepath: Path, expected_format: str, actual_format: Optional[str] = None):
        details = {
            "filepath": str(filepath),
            "expected_format": expected_format
        }
        if actual_format:
            details["actual_format"] = actual_format
            
        msg = f"Invalid file format for {filepath}"
        if actual_format:
            msg = f"File format mismatch: expected {expected_format}, got {actual_format}"
            
        super().__init__(msg, details=details)


# ============================================================================
# Gaussian parser errors
# ============================================================================

class GaussianParserError(MESSAutomationError):
    """Gaussian output parser related errors."""
    pass


class GaussianFileParseError(GaussianParserError):
    """Failed to parse Gaussian output file."""
    
    def __init__(self, filepath: Path, error: str, line_number: Optional[int] = None):
        details = {"filepath": str(filepath), "error": error}
        if line_number:
            details["line_number"] = line_number
            
        super().__init__(
            f"Failed to parse Gaussian output file: {filepath}",
            details=details
        )


class GaussianConvergenceError(GaussianParserError):
    """Gaussian calculation did not converge."""
    
    def __init__(self, filepath: Path, reason: str):
        super().__init__(
            f"Gaussian calculation did not converge: {filepath}",
            details={"filepath": str(filepath), "reason": reason}
        )


class NoGeometryFoundError(GaussianParserError):
    """No molecular geometry found in Gaussian output."""
    
    def __init__(self, filepath: Path):
        super().__init__(
            f"No molecular geometry found in {filepath}",
            details={"filepath": str(filepath)}
        )


class NoFrequenciesFoundError(GaussianParserError):
    """No vibrational frequencies found in Gaussian output."""
    
    def __init__(self, filepath: Path):
        super().__init__(
            f"No vibrational frequencies found in {filepath}",
            details={"filepath": str(filepath)}
        )


class NoEnergyFoundError(GaussianParserError):
    """No energy data found in Gaussian output."""
    
    def __init__(self, filepath: Path, energy_type: str):
        super().__init__(
            f"No {energy_type} found in {filepath}",
            details={"filepath": str(filepath), "energy_type": energy_type}
        )


class InvalidFrequencyError(GaussianParserError):
    """Invalid or suspicious frequency value found."""
    
    def __init__(self, filepath: Path, frequency: float, reason: str):
        super().__init__(
            f"Invalid frequency in {filepath}: {frequency} cm^-1",
            details={"filepath": str(filepath), "frequency": frequency, "reason": reason}
        )


# ============================================================================
# Quantum data processing errors
# ============================================================================

class QuantumDataError(MESSAutomationError):
    """Quantum data processing related errors."""
    pass


class DataValidationError(QuantumDataError):
    """Quantum data validation failed."""
    
    def __init__(self, errors: List[str], filename: Optional[str] = None):
        details = {"validation_errors": errors}
        if filename:
            details["filename"] = filename
            
        super().__init__(
            f"Quantum data validation failed with {len(errors)} error(s)",
            details=details
        )


class ScalingFactorError(QuantumDataError):
    """Invalid scaling factor."""
    
    def __init__(self, scaling_factor: float):
        super().__init__(
            f"Invalid scaling factor: {scaling_factor} (must be > 0)",
            details={"scaling_factor": scaling_factor}
        )


class UnitConversionError(QuantumDataError):
    """Unit conversion failed."""
    
    def __init__(self, from_unit: str, to_unit: str, value: Any):
        super().__init__(
            f"Cannot convert {value} from {from_unit} to {to_unit}",
            details={"from_unit": from_unit, "to_unit": to_unit, "value": value}
        )


class ImaginaryFrequencyError(QuantumDataError):
    """Error handling imaginary frequencies."""
    
    def __init__(self, count: int, max_allowed: int = 3):
        super().__init__(
            f"Too many imaginary frequencies: {count} (max allowed: {max_allowed})",
            details={"count": count, "max_allowed": max_allowed}
        )


# ============================================================================
# MESS assembly errors
# ============================================================================

class MESSAssemblyError(MESSAutomationError):
    """MESS input file assembly related errors."""
    pass


class TemplateError(MESSAssemblyError):
    """Template rendering error."""
    
    def __init__(self, template_name: str, error: str):
        super().__init__(
            f"Failed to render template '{template_name}'",
            details={"template_name": template_name, "error": error}
        )


class MissingTemplateError(MESSAssemblyError):
    """Required template not found."""
    
    def __init__(self, template_name: str):
        super().__init__(
            f"Required template not found: '{template_name}'",
            details={"template_name": template_name}
        )


class SpeciesDefinitionError(MESSAssemblyError):
    """Species definition error."""
    
    def __init__(self, species_name: str, error: str):
        super().__init__(
            f"Invalid species definition for '{species_name}'",
            details={"species_name": species_name, "error": error}
        )


class ReactionNetworkError(MESSAssemblyError):
    """Reaction network validation error."""
    
    def __init__(self, error: str, reaction_info: Optional[Dict[str, Any]] = None):
        details = {"error": error}
        if reaction_info:
            details.update(reaction_info)
            
        super().__init__(
            f"Reaction network error: {error}",
            details=details
        )


class BarrierDepthError(MESSAssemblyError):
    """Barrier depth calculation error."""
    
    def __init__(self, barrier_name: str, error: str):
        super().__init__(
            f"Barrier depth calculation error for '{barrier_name}': {error}",
            details={"barrier_name": barrier_name, "error": error}
        )


# ============================================================================
# Utility functions
# ============================================================================

def wrap_exception(func):
    """
    Decorator to wrap exceptions with custom error types.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MESSAutomationError:
            # Already our custom exception, re-raise
            raise
        except FileNotFoundError as e:
            raise FileNotFoundError(Path(str(e))) from e
        except IOError as e:
            # Extract filepath if possible
            filepath = Path(str(e)).resolve() if hasattr(e, 'filename') else None
            if filepath and filepath.exists():
                raise FileReadError(filepath, str(e)) from e
            else:
                raise FileIOError(str(e)) from e
        except ValueError as e:
            # Try to determine error type based on context
            error_msg = str(e)
            if "scaling" in error_msg.lower():
                raise ScalingFactorError(0.0) from e
            elif "unit" in error_msg.lower():
                raise UnitConversionError("unknown", "unknown", "unknown") from e
            else:
                raise MESSAutomationError(f"Value error: {error_msg}") from e
        except Exception as e:
            # Generic exception wrapper
            raise MESSAutomationError(f"Unexpected error: {str(e)}") from e
            
    return wrapper