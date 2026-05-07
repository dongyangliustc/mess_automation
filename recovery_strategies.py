"""
Recovery strategies for MESS automation package.

This module provides specific recovery strategies for different error types,
allowing the application to gracefully handle errors and continue operation.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import shutil

from .exceptions import (
    ConfigurationError, ConfigFileNotFoundError, ConfigValidationError,
    FileNotFoundError, FileReadError, GaussianFileParseError,
    NoGeometryFoundError, NoFrequenciesFoundError, NoEnergyFoundError,
    DataValidationError, ScalingFactorError, ImaginaryFrequencyError,
    TemplateError, MissingTemplateError, SpeciesDefinitionError
)


class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def __init__(self, name: str, error_types: List[type]):
        """
        Initialize recovery strategy.
        
        Args:
            name: Strategy name
            error_types: List of exception types this strategy can handle
        """
        self.name = name
        self.error_types = error_types
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, error: Exception) -> bool:
        """
        Check if this strategy can handle the error.
        
        Args:
            error: Exception to check
            
        Returns:
            True if strategy can handle the error
        """
        return isinstance(error, tuple(self.error_types))
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Attempt to recover from the error.
        
        Args:
            error: Exception to recover from
            context: Context information
            
        Returns:
            Tuple of (success, recovery_context)
        """
        raise NotImplementedError("Subclasses must implement recover()")
    
    def __str__(self) -> str:
        """String representation."""
        return f"RecoveryStrategy({self.name})"


class SkipFileRecovery(RecoveryStrategy):
    """Recovery strategy for skipping problematic files."""
    
    def __init__(self):
        super().__init__(
            name="skip_file",
            error_types=[
                FileNotFoundError, FileReadError, GaussianFileParseError,
                NoGeometryFoundError, NoFrequenciesFoundError, NoEnergyFoundError,
                DataValidationError, ImaginaryFrequencyError
            ]
        )
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Skip the problematic file."""
        filepath = context.get("filepath")
        if filepath:
            self.logger.warning(f"Skipping file: {filepath} (error: {error})")
            return True, {"skipped_file": str(filepath), "skip_reason": str(error)}
        
        return False, {"reason": "No filepath in context"}


class UseDefaultValueRecovery(RecoveryStrategy):
    """Recovery strategy for using default values."""
    
    def __init__(self):
        super().__init__(
            name="use_default",
            error_types=[ConfigurationError, ConfigValidationError]
        )
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Use default value for missing configuration."""
        config_key = context.get("key")
        default_value = context.get("default_value")
        
        if config_key is not None and default_value is not None:
            self.logger.info(f"Using default value for '{config_key}': {default_value}")
            return True, {
                "config_key": config_key,
                "used_default": default_value,
                "original_error": str(error)
            }
        
        return False, {"reason": "Missing key or default value"}


class ScaleFactorAdjustmentRecovery(RecoveryStrategy):
    """Recovery strategy for adjusting invalid scaling factors."""
    
    def __init__(self):
        super().__init__(
            name="adjust_scale_factor",
            error_types=[ScalingFactorError]
        )
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Adjust invalid scaling factor to a reasonable default."""
        invalid_factor = context.get("scaling_factor")
        
        if invalid_factor is not None:
            # Choose appropriate default based on typical quantum chemistry methods
            if invalid_factor <= 0:
                adjusted_factor = 0.971  # Default for harmonic frequencies
                reason = "Negative or zero scaling factor"
            elif invalid_factor > 2.0:
                adjusted_factor = 1.0  # Unscaled
                reason = "Excessively high scaling factor"
            else:
                # Factor might be reasonable, just slightly out of bounds
                adjusted_factor = max(0.9, min(1.1, invalid_factor))
                reason = "Slightly out of bounds scaling factor"
            
            self.logger.warning(
                f"Adjusting scaling factor from {invalid_factor} to {adjusted_factor}: {reason}"
            )
            
            return True, {
                "original_factor": invalid_factor,
                "adjusted_factor": adjusted_factor,
                "adjustment_reason": reason
            }
        
        return False, {"reason": "No scaling factor in context"}


class TemplateFallbackRecovery(RecoveryStrategy):
    """Recovery strategy for template errors."""
    
    def __init__(self):
        super().__init__(
            name="template_fallback",
            error_types=[TemplateError, MissingTemplateError]
        )
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Fall back to a default template."""
        template_name = context.get("template_name")
        template_type = context.get("template_type")
        
        if template_name:
            # Try to find a fallback template
            fallback_result = self._get_fallback_template(template_name, template_type)
            if fallback_result:
                fallback_name, fallback_content = fallback_result
                self.logger.warning(f"Using fallback template '{fallback_name}' instead of '{template_name}'")
                
                return True, {
                    "original_template": template_name,
                    "fallback_template": fallback_name,
                    "template_content": fallback_content
                }
        
        return False, {"reason": "No template name or fallback available"}
    
    def _get_fallback_template(self, template_name: str, template_type: Optional[str]) -> Optional[Tuple[str, str]]:
        """Get fallback template content."""
        # Define minimal fallback templates for common template types
        fallback_templates = {
            "global": {
                "name": "minimal_global",
                "content": """TemperatureList[K] {temperatures}
PressureList[atm] {pressures}
EnergyStepOverTemperature {energy_step}
ExcessEnergyOverTemperature {excess_energy}
ModelEnergyLimit[kcal/mol] {model_limit}
CalculationMethod {method}
WellCutoff[kcal/mol] {well_cutoff}
ChemicalEigenvalueMax {eigenvalue_max}
ReductionMethod {reduction_method}"""
            },
            "species": {
                "name": "minimal_species",
                "content": """{species_name}
  Core RigidRotor
    SymmetryFactor {symmetry}
  End
  Geometry[angstrom] {num_atoms}
{geometry}
  Frequencies[1/cm] {num_frequencies}
{frequencies}
  ElectronicLevels[1/cm] {num_levels}
{electronic_levels}
  ZeroEnergy[kcal/mol] {ZeroEnergy}"""
            },
            "barrier": {
                "name": "minimal_barrier",
                "content": """{barrier_name}
  WellDepth[kcal/mol] {well_depth}"""
            }
        }
        
        if template_type in fallback_templates:
            return fallback_templates[template_type]["name"], fallback_templates[template_type]["content"]
        
        return None


class DirectoryCleanupRecovery(RecoveryStrategy):
    """Recovery strategy for cleaning up output directories."""
    
    def __init__(self):
        super().__init__(
            name="directory_cleanup",
            error_types=[Exception]  # Can handle any error during cleanup
        )
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Clean up output directory to allow retry."""
        output_dir = context.get("output_dir")
        backup_dir = context.get("backup_dir")
        
        if output_dir:
            output_path = Path(output_dir)
            
            try:
                # Create backup if requested
                if backup_dir and output_path.exists():
                    backup_path = Path(backup_dir) / f"backup_{output_path.name}"
                    shutil.copytree(output_path, backup_path)
                    self.logger.info(f"Created backup at {backup_path}")
                
                # Clean up output directory
                if output_path.exists():
                    shutil.rmtree(output_path)
                    self.logger.info(f"Cleaned up output directory: {output_path}")
                
                # Recreate directory
                output_path.mkdir(parents=True, exist_ok=True)
                
                return True, {
                    "output_dir": str(output_path),
                    "backup_dir": str(backup_path) if backup_dir else None,
                    "cleanup_performed": True
                }
                
            except Exception as cleanup_error:
                self.logger.error(f"Failed to cleanup directory {output_path}: {cleanup_error}")
                return False, {"reason": str(cleanup_error)}
        
        return False, {"reason": "No output directory in context"}


class RecoveryManager:
    """Manages recovery strategies and orchestrates recovery attempts."""
    
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = []
        self.logger = logging.getLogger(__name__)
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        self.register_strategy(SkipFileRecovery())
        self.register_strategy(UseDefaultValueRecovery())
        self.register_strategy(ScaleFactorAdjustmentRecovery())
        self.register_strategy(TemplateFallbackRecovery())
        self.register_strategy(DirectoryCleanupRecovery())
    
    def register_strategy(self, strategy: RecoveryStrategy):
        """
        Register a recovery strategy.
        
        Args:
            strategy: RecoveryStrategy instance
        """
        self.strategies.append(strategy)
        self.logger.debug(f"Registered recovery strategy: {strategy.name}")
    
    def get_strategies_for_error(self, error: Exception) -> List[RecoveryStrategy]:
        """
        Get strategies that can handle the given error.
        
        Args:
            error: Exception to handle
            
        Returns:
            List of applicable recovery strategies
        """
        return [s for s in self.strategies if s.can_handle(error)]
    
    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Attempt to recover from an error using registered strategies.
        
        Args:
            error: Exception to recover from
            context: Context information
            
        Returns:
            Tuple of (success, recovery_context)
        """
        applicable_strategies = self.get_strategies_for_error(error)
        
        if not applicable_strategies:
            self.logger.debug(f"No recovery strategies available for {type(error).__name__}")
            return False, {"reason": "No applicable recovery strategies"}
        
        self.logger.info(f"Attempting recovery for {type(error).__name__} with {len(applicable_strategies)} strategies")
        
        # Try strategies in order
        for strategy in applicable_strategies:
            try:
                self.logger.debug(f"Trying recovery strategy: {strategy.name}")
                success, recovery_context = strategy.recover(error, context)
                
                if success:
                    self.logger.info(f"Recovery successful with strategy: {strategy.name}")
                    return True, recovery_context
                    
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                continue
        
        self.logger.warning(f"All recovery strategies failed for {type(error).__name__}")
        return False, {"reason": "All recovery strategies failed"}
    
    def add_context_for_recovery(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Add recovery-specific context.
        
        Args:
            context: Existing context
            **kwargs: Additional context items
            
        Returns:
            Updated context
        """
        # Add recovery-related defaults
        recovery_context = context.copy()
        
        # Add common defaults
        if "default_scaling_factor" not in recovery_context:
            recovery_context["default_scaling_factor"] = 0.971
        
        if "skip_on_error" not in recovery_context:
            recovery_context["skip_on_error"] = True
        
        # Add provided kwargs
        recovery_context.update(kwargs)
        
        return recovery_context


# Global recovery manager instance
_global_recovery_manager: Optional[RecoveryManager] = None


def get_global_recovery_manager() -> RecoveryManager:
    """
    Get global recovery manager instance.
    
    Returns:
        RecoveryManager instance
    """
    global _global_recovery_manager
    
    if _global_recovery_manager is None:
        _global_recovery_manager = RecoveryManager()
    
    return _global_recovery_manager


def attempt_recovery(error: Exception, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to attempt recovery using the global manager.
    
    Args:
        error: Exception to recover from
        **kwargs: Context information
        
    Returns:
        Tuple of (success, recovery_context)
    """
    manager = get_global_recovery_manager()
    return manager.attempt_recovery(error, kwargs)


def with_recovery(func: Callable) -> Callable:
    """
    Decorator to add automatic recovery to a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            # Create context from function arguments
            context = {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args)
            }
            
            # Add kwargs to context (excluding sensitive data)
            for key, value in kwargs.items():
                if not key.startswith("_") and not isinstance(value, (bytes, bytearray)):
                    context[key] = value
            
            # Attempt recovery
            manager = get_global_recovery_manager()
            recovery_context = manager.add_context_for_recovery(context)
            success, recovery_result = manager.attempt_recovery(error, recovery_context)
            
            if success:
                logging.getLogger(__name__).info(
                    f"Recovered from error in {func.__name__}: {error}",
                    extra={"recovery_strategy": recovery_result.get("strategy", "unknown")}
                )
                
                # Return recovery result or continue with default behavior
                return recovery_result.get("recovery_result", None)
            else:
                # Re-raise if recovery failed
                raise
    
    return wrapper