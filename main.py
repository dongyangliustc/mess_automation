#!/usr/bin/env python3
"""
Main module for MESS automation package - Command line interface.
Fixed to work as standalone script.

Enhanced with comprehensive error handling and logging.
"""

import os
import sys
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import yaml

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import local modules - handle both package and standalone cases
try:
    # Try package imports first
    from .parser import GaussianParser, QuantumData
    from .processor import FrequencyCorrector, CorrectionResult, UnitConverter, create_molecule_object
    from .assembler import MESSAssembler, MESSGlobalSettings, MESSModelSettings, MESSSpeciesConfig, MESSBarrierConfig
    from .exceptions import (
        ConfigurationError, ConfigFileNotFoundError, ConfigValidationError,
        FileNotFoundError, FileReadError, GaussianFileParseError,
        DataValidationError, ScalingFactorError, UnitConversionError
    )
    from .error_handler import setup_global_error_handler, get_global_error_handler, handle_error
    from .logging_config import setup_logging, get_logger, log_function_call
except ImportError:
    # Fall back to direct imports for standalone use
    try:
        import parser
        import processor
        import assembler
        
        from parser import GaussianParser, QuantumData
        from processor import FrequencyCorrector, CorrectionResult, UnitConverter, create_molecule_object
        from assembler import MESSAssembler, MESSGlobalSettings, MESSModelSettings, MESSSpeciesConfig, MESSBarrierConfig
        
        # Try to import custom modules
        try:
            from exceptions import (
                ConfigurationError, ConfigFileNotFoundError, ConfigValidationError,
                FileNotFoundError, FileReadError, GaussianFileParseError,
                DataValidationError, ScalingFactorError, UnitConversionError
            )
            from error_handler import setup_global_error_handler, get_global_error_handler, handle_error
            from logging_config import setup_logging, get_logger, log_function_call
        except ImportError:
            # Create dummy implementations for backward compatibility
            class ConfigurationError(Exception): pass
            class ConfigFileNotFoundError(ConfigurationError): pass
            class ConfigValidationError(ConfigurationError): pass
            class FileNotFoundError(Exception): pass
            class FileReadError(Exception): pass
            class GaussianFileParseError(Exception): pass
            class DataValidationError(Exception): pass
            class ScalingFactorError(Exception): pass
            class UnitConversionError(Exception): pass
            
            def setup_global_error_handler(*args, **kwargs):
                pass
            def get_global_error_handler():
                return None
            def handle_error(error, **kwargs):
                return False
            def setup_logging(**kwargs):
                return logging.getLogger()
            def get_logger(name):
                return logging.getLogger(name)
            def log_function_call(func):
                return func
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure parser.py, processor.py, and assembler.py are in the same directory.")
        sys.exit(1)


# Configure logging - will be properly setup in main()
logger = get_logger(__name__)

# Module-level dict that stores processed quantum data (populated during a run).
# Exposed here so that tests can patch it with mock data.
processed_files: Dict[str, Dict[str, Any]] = {}


def setup_corrector(config: Dict[str, Any]) -> "FrequencyCorrector":
    """
    Create and return a FrequencyCorrector from *config*.

    Args:
        config: Configuration dictionary (uses ``quantum`` sub-section).

    Returns:
        Configured FrequencyCorrector instance.
    """
    quantum_cfg = config.get("quantum", {})
    scaling_factor = quantum_cfg.get("frequency_scaling_factor", 0.971)
    handle_imaginary = quantum_cfg.get("handle_imaginary", "abs")
    return FrequencyCorrector(
        scaling_factor=scaling_factor,
        handle_imaginary=handle_imaginary,
    )


def process_gaussian_file(
    file_path: Union[str, Path],
    corrector: "FrequencyCorrector",
) -> Optional[Dict[str, Any]]:
    """
    Parse a single Gaussian output file and apply frequency corrections.

    Args:
        file_path: Path to the Gaussian ``.out`` file.
        corrector: Pre-configured FrequencyCorrector instance.

    Returns:
        Dict ``{"qdata": QuantumData, "correction": CorrectionResult}``
        or ``None`` if parsing / correction failed.
    """
    parser_instance = GaussianParser()
    qdata = parser_instance.parse_file(str(file_path))

    if qdata is None:
        logger.warning(f"Parser returned None for {file_path}")
        return None

    if not qdata.convergence_status:
        logger.warning(f"Unconverged calculation skipped: {file_path}")
        return None

    result = corrector.correct_frequencies(qdata)
    if result is None or not result.success:
        logger.warning(
            f"Frequency correction failed for {file_path}: "
            f"{getattr(result, 'error_message', 'unknown error')}"
        )
        return None

    return {"qdata": qdata, "correction": result}


def find_quantum_data(file_key: str) -> Optional[Dict[str, Any]]:
    """
    Look up processed quantum data by file path in the module-level cache.

    Args:
        file_key: Original file path used as key (exact or normalised match).

    Returns:
        Data dict or ``None`` if not found.
    """
    # Exact match
    if file_key in processed_files:
        return processed_files[file_key]

    # Try normalised path comparison
    try:
        target = Path(file_key).resolve()
        for key, data in processed_files.items():
            if Path(key).resolve() == target:
                return data
    except Exception:
        pass

    return None


def parse_arguments() -> argparse.Namespace:
    """
    Parse and return command-line arguments.

    Exposed as a standalone function so that tests can call it with a patched
    ``sys.argv`` without triggering the full ``main()`` workflow.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    arg_parser = argparse.ArgumentParser(
        description="MESS Automation Tool - Generate MESS input files from quantum chemistry data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    arg_parser.add_argument("-c", "--config", required=True,
                            help="Path to configuration YAML file")
    arg_parser.add_argument("-o", "--output", required=True,
                            help="Path to output MESS input file")
    arg_parser.add_argument("--scaling", type=float,
                            help="Frequency scaling factor (overrides config)")
    arg_parser.add_argument("--overwrite", action="store_true",
                            help="Overwrite output file if it exists")
    arg_parser.add_argument("-v", "--verbose", action="store_true",
                            help="Verbose output")
    arg_parser.add_argument("--log-level", dest="log_level", default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                            help="Logging level")

    return arg_parser.parse_args()


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary with configuration
    """
    config_file = Path(config_file)
    
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        raise
    
    logger.info(f"Loaded configuration from {config_file}")
    return config


def setup_global_settings(config: Dict[str, Any]) -> MESSGlobalSettings:
    """
    Setup global settings from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MESSGlobalSettings object
    """
    global_config = config.get("mess_global", {})
    
    # Extract temperature list
    temp_list = global_config.get("temperature_list", 
                                 [200, 225, 250, 275, 300, 350, 400, 450, 500, 
                                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    
    # Extract pressure list
    pressure_list = global_config.get("pressure_list", [1.0])
    
    # Create settings object
    settings = MESSGlobalSettings(
        temperature_list=temp_list,
        pressure_list=pressure_list,
        energy_step_over_temperature=global_config.get("energy_step_over_temperature", 0.2),
        excess_energy_over_temperature=global_config.get("excess_energy_over_temperature", 50),
        model_energy_limit=global_config.get("model_energy_limit", 400),
        calculation_method=global_config.get("calculation_method", "direct"),
        well_cutoff=global_config.get("well_cutoff", 20),
        chemical_eigenvalue_max=global_config.get("chemical_eigenvalue_max", 0.2),
        reduction_method=global_config.get("reduction_method", "diagonalization"),
        rate_output=global_config.get("rate_output", "mess.out"),
        log_output=global_config.get("log_output"),
        eigenvalue_output=global_config.get("eigenvalue_output"),
        eigenvector_output=global_config.get("eigenvector_output"),
        ped_output=global_config.get("ped_output")
    )
    
    logger.info(f"Global settings: {len(settings.temperature_list)} temperatures, "
               f"{len(settings.pressure_list)} pressures")
    return settings


def setup_model_settings(config: Dict[str, Any]) -> MESSModelSettings:
    """
    Setup model settings from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MESSModelSettings object
    """
    model_config = config.get("mess_model", {})
    
    # Default values
    energy_relaxation = {
        "model": "Exponential",
        "factor": 350.0,
        "power": 0.85,
        "exponent_cutoff": 10
    }
    
    collision_frequency = {
        "model": "LennardJones",
        "epsilons": [10.0, 595.38],
        "sigmas": [2.55, 4.89],
        "masses": [4.0, 78.05]
    }
    
    # Override with config values
    if "energy_relaxation" in model_config:
        energy_relaxation.update(model_config["energy_relaxation"])
    
    if "collision_frequency" in model_config:
        collision_frequency.update(model_config["collision_frequency"])
    
    settings = MESSModelSettings(
        energy_relaxation=energy_relaxation,
        collision_frequency=collision_frequency
    )
    
    logger.info("Model settings configured")
    return settings


def process_quantum_data(
    config: Dict[str, Any],
    scaling_factor: Optional[float] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Process all quantum chemistry data files.
    
    Args:
        config: Configuration dictionary
        scaling_factor: Optional scaling factor (overrides config)
        
    Returns:
        Dictionary mapping filenames to (QuantumData, CorrectionResult) tuples
    """
    input_config = config.get("input", {})
    quantum_config = config.get("quantum", {})
    processing_config = config.get("processing", {})
    
    # Get scaling factor
    if scaling_factor is None:
        scaling_factor = quantum_config.get("frequency_scaling_factor", 0.971)
    
    # Determine files to process
    files_to_process = []
    
    # Check for explicit file list
    explicit_files = input_config.get("files", [])
    for file_path in explicit_files:
        path = Path(file_path)
        if path.exists():
            files_to_process.append(path)
        else:
            logger.warning(f"File not found: {file_path}")
    
    # Check for directory pattern
    gaussian_outputs = input_config.get("gaussian_outputs")
    if gaussian_outputs and not explicit_files:
        pattern = gaussian_outputs
        directory = Path(pattern).parent
        file_pattern = Path(pattern).name
        
        # Find files
        if directory.exists():
            files = list(directory.glob(file_pattern))
            files_to_process.extend(files)
            logger.info(f"Found {len(files)} files matching pattern: {pattern}")
        else:
            logger.warning(f"Directory not found: {directory}")
    
    if not files_to_process:
        logger.error("No files found to process")
        return {}
    
    # Process files
    parser = GaussianParser(
        skip_unconverged=processing_config.get("skip_unconverged", True),
        validate=processing_config.get("validate_frequencies", True)
    )
    
    corrector = FrequencyCorrector(
        scaling_factor=scaling_factor,
        handle_imaginary="abs",
        validate_input=True
    )
    
    results = {}
    
    for filepath in files_to_process:
        logger.info(f"Processing {filepath}")
        
        try:
            # Parse Gaussian output
            qdata = parser.parse_file(filepath)
            
            if not qdata.success:
                logger.warning(f"Failed to parse {filepath}: {qdata.error_message}")
                continue
            
            # Apply frequency scaling
            correction = corrector.correct_frequencies(qdata, scaling_factor)
            
            if not correction.success:
                logger.warning(f"Failed to correct frequencies for {filepath}: {correction.error_message}")
                continue
            
            results[str(filepath)] = {
                "qdata": qdata,
                "correction": correction,
                "filename": filepath.name
            }
            
            logger.info(f"Successfully processed {filepath}: {qdata.num_atoms} atoms, "
                       f"{qdata.num_frequencies} frequencies")
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
    
    logger.info(f"Processed {len(results)} out of {len(files_to_process)} files")
    return results


def create_species_and_barriers(
    config: Dict[str, Any],
    quantum_results: Dict[str, Dict[str, Any]]
) -> tuple[Dict[str, MESSSpeciesConfig], Dict[str, MESSBarrierConfig]]:
    """
    Create species and barrier configurations from processed data.
    
    Args:
        config: Configuration dictionary
        quantum_results: Processed quantum data
        
    Returns:
        Tuple of (species_dict, barriers_dict)
    """
    network_config = config.get("reaction_network", {})
    species_list = network_config.get("species", [])
    
    species_configs = {}
    barrier_configs = {}
    
    # Helper to find quantum data by filename
    def find_quantum_data(file_key: str) -> Optional[Dict[str, Any]]:
        # Try exact match first
        if file_key in quantum_results:
            return quantum_results[file_key]
        
        # Try partial match
        for key, data in quantum_results.items():
            if Path(key).name == file_key:
                return data
        
        # Try without extension
        for key, data in quantum_results.items():
            if Path(key).stem == Path(file_key).stem:
                return data
        
        return None
    
    # Process species - separate into wells and barriers
    for species_def in species_list:
        name = species_def.get("name")
        species_type = species_def.get("type", "well")
        gaussian_file = species_def.get("gaussian_file")
        from_species = species_def.get("from_species")
        to_species = species_def.get("to_species")
        
        if not name or not gaussian_file:
            logger.warning(f"Incomplete species definition: {species_def}")
            continue
        
        # Find corresponding quantum data
        quantum_data = find_quantum_data(gaussian_file)
        if not quantum_data:
            logger.warning(f"No quantum data found for {gaussian_file} (species: {name})")
            continue
        
        # If it's a barrier with from/to species, create a barrier config
        if species_type == "barrier" and from_species and to_species:
            # Create barrier configuration
            barrier_config = MESSBarrierConfig(
                name=name,
                from_species=from_species,
                to_species=to_species,
                quantum_data=quantum_data["qdata"],
                correction=quantum_data["correction"],
                stoichiometry=species_def.get("stoichiometry"),
                symmetry_factor=species_def.get("symmetry_factor", 1.0),
                comment=species_def.get("comment", ""),
                method=species_def.get("method", "RRHO"),
                core_type=species_def.get("core_type", "RigidRotor"),
                electronic_levels=species_def.get("electronic_levels", [
                    {"energy": 0.0, "degeneracy": 1.0}
                ]),
                is_bimolecular=species_def.get("is_bimolecular", False),
                potential_prefactor=species_def.get("potential_prefactor"),
                potential_power_exponent=species_def.get("potential_power_exponent"),
                # Calculate barrier depths if energies are available
                forward_barrier=None,
                reverse_barrier=None
            )
            
            barrier_configs[name] = barrier_config
            logger.info(f"Created barrier {name} ({from_species} -> {to_species}) from {gaussian_file}")
        else:
            # Create species configuration (well or other non-barrier)
            species_config = MESSSpeciesConfig(
                name=name,
                type=species_type,
                quantum_data=quantum_data["qdata"],
                correction=quantum_data["correction"],
                stoichiometry=species_def.get("stoichiometry"),
                symmetry_factor=species_def.get("symmetry_factor", 1.0),
                ground_energy=species_def.get("ground_energy", 0.0),
                comment=species_def.get("comment", ""),
                method=species_def.get("method", "RRHO"),
                core_type=species_def.get("core_type", "RigidRotor"),
                electronic_levels=species_def.get("electronic_levels", [
                    {"energy": 0.0, "degeneracy": 1.0}
                ])
            )
            
            species_configs[name] = species_config
            logger.info(f"Created species {name} ({species_type}) from {gaussian_file}")
    
    # Process barriers (simplified - in real implementation, this would be more complex)
    # For now, we'll assume barriers are defined similarly to species
    barrier_defs = network_config.get("barriers", [])
    
    for barrier_def in barrier_defs:
        name = barrier_def.get("name")
        from_species = barrier_def.get("from_species")
        to_species = barrier_def.get("to_species")
        gaussian_file = barrier_def.get("gaussian_file")
        
        if not all([name, from_species, to_species, gaussian_file]):
            logger.warning(f"Incomplete barrier definition: {barrier_def}")
            continue
        
        # Find corresponding quantum data
        quantum_data = find_quantum_data(gaussian_file)
        if not quantum_data:
            logger.warning(f"No quantum data found for {gaussian_file} (barrier: {name})")
            continue
        
        # Check if from/to species exist
        if from_species not in species_configs:
            logger.warning(f"From species {from_species} not found for barrier {name}")
            continue
        
        if to_species not in species_configs:
            logger.warning(f"To species {to_species} not found for barrier {name}")
            continue
        
        # Create barrier configuration
        barrier_config = MESSBarrierConfig(
            name=name,
            from_species=from_species,
            to_species=to_species,
            quantum_data=quantum_data["qdata"],
            correction=quantum_data["correction"],
            stoichiometry=barrier_def.get("stoichiometry"),
            symmetry_factor=barrier_def.get("symmetry_factor", 1.0),
            comment=barrier_def.get("comment", ""),
            method=barrier_def.get("method", "RRHO"),
            core_type=barrier_def.get("core_type", "RigidRotor"),
            electronic_levels=barrier_def.get("electronic_levels", [
                {"energy": 0.0, "degeneracy": 1.0}
            ]),
            is_bimolecular=barrier_def.get("is_bimolecular", False),
            potential_prefactor=barrier_def.get("potential_prefactor"),
            potential_power_exponent=barrier_def.get("potential_power_exponent")
        )
        
        barrier_configs[name] = barrier_config
        logger.info(f"Created barrier {name} ({from_species} -> {to_species}) from {gaussian_file}")
    
    logger.info(f"Created {len(species_configs)} species and {len(barrier_configs)} barriers")
    return species_configs, barrier_configs


def parse_config_file(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Parse a YAML configuration file and return its contents as a dictionary.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with configuration, or None if parsing fails
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading {config_path}: {e}")
        return None


def validate_config(config: Dict[str, Any]) -> tuple:
    """
    Validate a configuration dictionary for required fields and valid values.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors = []
    
    if not config:
        return False, ["Configuration is empty"]
    
    # --- input section ---
    input_cfg = config.get("input", {})
    files = input_cfg.get("files", [])
    if not files:
        errors.append("input.files must contain at least one file path")
    
    # --- quantum section ---
    quantum_cfg = config.get("quantum", {})
    if not quantum_cfg:
        errors.append("Missing required section: quantum")
    else:
        sf = quantum_cfg.get("frequency_scaling_factor")
        if sf is not None and sf <= 0:
            errors.append(
                f"quantum.frequency_scaling_factor must be positive, got {sf}"
            )
    
    # --- mess_global section ---
    global_cfg = config.get("mess_global", {})
    if not global_cfg:
        errors.append("Missing required section: mess_global")
    else:
        temp_list = global_cfg.get("temperature_list", [])
        if not temp_list:
            errors.append("mess_global.temperature_list must not be empty")
        elif any(t <= 0 for t in temp_list):
            errors.append(
                "mess_global.temperature_list contains negative or zero temperatures"
            )
        
        pressure_list = global_cfg.get("pressure_list", [])
        if not pressure_list:
            errors.append("mess_global.pressure_list must not be empty")
    
    # --- reaction_network species reference check (optional deeper validation) ---
    network_cfg = config.get("reaction_network", {})
    species_list = network_cfg.get("species", [])
    species_names = {s["name"] for s in species_list if "name" in s}
    for sp in species_list:
        if sp.get("type") == "barrier":
            from_sp = sp.get("from_species")
            to_sp = sp.get("to_species")
            if from_sp and from_sp not in species_names:
                errors.append(
                    f"Barrier '{sp.get('name')}' references unknown from_species: {from_sp}"
                )
            if to_sp and to_sp not in species_names:
                errors.append(
                    f"Barrier '{sp.get('name')}' references unknown to_species: {to_sp}"
                )
    
    return len(errors) == 0, errors


@dataclass
class Config:
    """Structured configuration object parsed from YAML config file."""
    
    # Input
    input_files: List[str]
    
    # Quantum settings
    frequency_scaling_factor: float = 0.971
    geometry_units: str = "angstrom"
    frequency_units: str = "1/cm"
    energy_units: str = "kcal/mol"
    
    # MESS global settings
    temperature_list: List[float] = None
    pressure_list: List[float] = None
    energy_step_over_temperature: float = 0.2
    excess_energy_over_temperature: float = 50
    model_energy_limit: float = 400
    calculation_method: str = "direct"
    well_cutoff: float = 20
    chemical_eigenvalue_max: float = 0.2
    reduction_method: str = "diagonalization"
    
    # Processing options
    skip_unconverged: bool = False
    validate_frequencies: bool = True
    create_backups: bool = True
    verbose: bool = False
    
    # Reaction network
    reaction_network: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.temperature_list is None:
            self.temperature_list = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        if self.pressure_list is None:
            self.pressure_list = [1.0]
        if self.reaction_network is None:
            self.reaction_network = []
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a parsed configuration dictionary."""
        input_cfg = config_dict.get("input", {})
        quantum_cfg = config_dict.get("quantum", {})
        global_cfg = config_dict.get("mess_global", {})
        processing_cfg = config_dict.get("processing", {})
        network_cfg = config_dict.get("reaction_network", {})
        
        return cls(
            input_files=input_cfg.get("files", []),
            frequency_scaling_factor=quantum_cfg.get("frequency_scaling_factor", 0.971),
            geometry_units=quantum_cfg.get("geometry_units", "angstrom"),
            frequency_units=quantum_cfg.get("frequency_units", "1/cm"),
            energy_units=quantum_cfg.get("energy_units", "kcal/mol"),
            temperature_list=global_cfg.get("temperature_list"),
            pressure_list=global_cfg.get("pressure_list"),
            energy_step_over_temperature=global_cfg.get("energy_step_over_temperature", 0.2),
            excess_energy_over_temperature=global_cfg.get("excess_energy_over_temperature", 50),
            model_energy_limit=global_cfg.get("model_energy_limit", 400),
            calculation_method=global_cfg.get("calculation_method", "direct"),
            well_cutoff=global_cfg.get("well_cutoff", 20),
            chemical_eigenvalue_max=global_cfg.get("chemical_eigenvalue_max", 0.2),
            reduction_method=global_cfg.get("reduction_method", "diagonalization"),
            skip_unconverged=processing_cfg.get("skip_unconverged", False),
            validate_frequencies=processing_cfg.get("validate_frequencies", True),
            create_backups=processing_cfg.get("create_backups", True),
            verbose=processing_cfg.get("verbose", False),
            reaction_network=network_cfg.get("species", []),
        )






def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="MESS Automation Tool - Generate MESS input files from quantum chemistry data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate MESS input with default configuration
  python main_fixed.py -c config.yaml -o mess_input.inp
  
  # Use custom scaling factor
  python main_fixed.py -c config.yaml -o mess_input.inp --scaling 0.971
  
  # Verbose output
  python main_fixed.py -c config.yaml -o mess_input.inp -v
  
  # Overwrite existing output file
  python main_fixed.py -c config.yaml -o mess_input.inp --overwrite
        """
    )
    
    parser.add_argument("-c", "--config", required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("-o", "--output", required=True,
                       help="Path to output MESS input file")
    parser.add_argument("--scaling", type=float,
                       help="Frequency scaling factor (overrides config)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite output file if it exists")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if output file exists
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        logger.error(f"Output file {output_path} already exists. Use --overwrite to overwrite.")
        sys.exit(1)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup global settings
        global_settings = setup_global_settings(config)
        
        # Setup model settings
        model_settings = setup_model_settings(config)
        
        # Process quantum data
        logger.info("Processing quantum chemistry data...")
        quantum_results = process_quantum_data(config, args.scaling)
        
        if not quantum_results:
            logger.error("No quantum data processed successfully. Check input files.")
            sys.exit(1)
        
        # Create species and barriers
        logger.info("Creating reaction network...")
        species_configs, barrier_configs = create_species_and_barriers(config, quantum_results)
        
        if not species_configs:
            logger.error("No species created. Check reaction network definition.")
            sys.exit(1)
        
        # Create assembler
        assembler = MESSAssembler()
        assembler.set_global_settings(global_settings)
        assembler.set_model_settings(model_settings)
        
        # Add species
        for species in species_configs.values():
            assembler.add_species(species)
        
        # Add barriers
        for barrier in barrier_configs.values():
            assembler.add_barrier(barrier)
        
        # Generate MESS input
        logger.info(f"Generating MESS input file: {output_path}")
        assembler.write_to_file(output_path, overwrite=args.overwrite)
        
        logger.info(f"Successfully generated MESS input file: {output_path}")
        logger.info(f"Total species: {len(species_configs)}")
        logger.info(f"Total barriers: {len(barrier_configs)}")
        
        # Print summary
        print("\n" + "="*60)
        print("MESS AUTOMATION - GENERATION COMPLETE")
        print("="*60)
        print(f"Output file: {output_path}")
        print(f"Species: {len(species_configs)}")
        print(f"Barriers: {len(barrier_configs)}")
        print(f"Quantum files processed: {len(quantum_results)}")
        
        # List species
        print("\nSpecies:")
        for name, species in species_configs.items():
            qdata = species.quantum_data
            print(f"  {name:10s} {species.type:12s} {qdata.num_atoms:3d} atoms, "
                  f"{qdata.num_frequencies:3d} frequencies")
        
        print("="*60)
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()