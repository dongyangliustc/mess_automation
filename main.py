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
        FileNotFoundError as MESSFileNotFoundError, FileReadError, GaussianFileParseError,
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
                FileNotFoundError as MESSFileNotFoundError, FileReadError, GaussianFileParseError,
                DataValidationError, ScalingFactorError, UnitConversionError
            )
            from error_handler import setup_global_error_handler, get_global_error_handler, handle_error
            from logging_config import setup_logging, get_logger, log_function_call
        except ImportError:
            # Create dummy implementations for backward compatibility
            class ConfigurationError(Exception): pass
            class ConfigFileNotFoundError(ConfigurationError): pass
            class ConfigValidationError(ConfigurationError): pass
            class MESSFileNotFoundError(Exception): pass
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


def get_config_value(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-None value from a config mapping."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def get_quantum_factors(quantum_cfg: Dict[str, Any]) -> Tuple[float, float]:
    """
    Read frequency and ZPE scaling factors from the quantum config.

    ``frequency_scaling_factor`` and ``zpe_scaling_factor`` are accepted as
    backward-compatible aliases.
    """
    frequency_factor = get_config_value(
        quantum_cfg,
        "Frequency_factor", "frequency_factor", "frequency_scaling_factor",
        default=1.0,
    )
    zpe_factor = get_config_value(
        quantum_cfg,
        "zpe_factor", "ZPE_factor", "zpe_scaling_factor",
        default=1.0,
    )
    return float(frequency_factor), float(zpe_factor)


def find_quantum_data(
    file_key: str,
    quantum_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Find processed quantum data by exact path, file name, or path stem."""
    data_source = quantum_results if quantum_results is not None else processed_files
    if file_key in data_source:
        return data_source[file_key]

    target = Path(file_key)
    for key, data in data_source.items():
        candidate = Path(key)
        if candidate.name == target.name:
            return data

    for key, data in data_source.items():
        candidate = Path(key)
        if candidate.stem == target.stem:
            return data

    return None


def setup_corrector(config: Dict[str, Any]) -> "FrequencyCorrector":
    """
    Create and return a FrequencyCorrector from *config*.

    Args:
        config: Configuration dictionary (uses ``quantum`` sub-section).

    Returns:
        Configured FrequencyCorrector instance.
    """
    quantum_cfg = config.get("quantum", {})
    frequency_factor, zpe_factor = get_quantum_factors(quantum_cfg)
    handle_imaginary = quantum_cfg.get("handle_imaginary", "abs")
    return FrequencyCorrector(
        Frequency_factor=frequency_factor,
        zpe_factor=zpe_factor,
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


def create_species_and_barriers(
        config: Dict[str, Any],
        quantum_results: Dict[str, Dict[str, Any]]
) -> tuple[Dict[str, MESSSpeciesConfig], Dict[str, MESSBarrierConfig]]:
    """
    创建物种和能垒配置，自动计算 ZeroEnergy [kcal/mol]。

    ZeroEnergy = (YAML 中的 EleEnergy - EleEnergy_baseline) [Hartree] + scaled_ZPE [Hartree]
    其中 scaled_ZPE = 原始 ZPE × zpe_factor
    最后乘以 627.509474 转换为 kcal/mol。
    """
    network_config = config.get("reaction_network", {})
    species_list = network_config.get("species", [])
    quantum_cfg = config.get("quantum", {})

    # ---------- 读取全局能量和 ZPE 校正参数 ----------
    _, zpe_factor = get_quantum_factors(quantum_cfg)
    energy_baseline = float(
        get_config_value(network_config, "Energy_baseline", "energy_baseline", default=0.0)
    )

    HARTREE_TO_KCAL = 627.509474

    species_configs = {}
    barrier_configs = {}

    # ---------- 辅助：根据文件名查找量子数据 ----------
    # ---------- 辅助：提取校正后的 ZPE（Hartree） ----------
    def extract_scaled_zpe(qdata_dict: dict) -> Optional[float]:
        """
        尝试从 CorrectionResult 或 QuantumData 中获取校正后的零点能 (Hartree)。
        返回 None 表示找不到。
        """
        correction = qdata_dict.get("correction")
        if correction is not None and getattr(correction, "scaled_zpe", None) is not None:
            return correction.scaled_zpe

        qdata = qdata_dict.get("qdata")
        if qdata is not None and qdata.zero_point_energy is not None:
            return qdata.zero_point_energy * zpe_factor
        return None

    # ---------- 核心：计算 ZeroEnergy ----------
    def compute_zero_energy(species_def: dict, qdata_dict: dict) -> float:
        """
        从 YAML 的 EleEnergy 和量子数据计算 ZeroEnergy (kcal/mol)。
        """
        manual_energy = get_config_value(
            species_def,
            "ZeroEnergy", "zero_energy", "GroundEnergy", "ground_energy",
        )
        ele_energy = get_config_value(
            species_def,
            "EleEnergy", "ele_energy", "electronic_energy", "scf_energy_hartree",
        )
        if ele_energy is None:
            if manual_energy is not None:
                return float(manual_energy)
            logger.error(
                f"Species/Barrier '{species_def.get('name')}' missing 'EleEnergy' in YAML. "
                "Setting ZeroEnergy to 0.0 kcal/mol."
            )
            return 0.0
        ele_energy = float(ele_energy)
        corrected_ele_energy = ele_energy - energy_baseline

        # 提取校正后的 ZPE（Hartree）
        scaled_zpe = extract_scaled_zpe(qdata_dict)
        if scaled_zpe is None:
            logger.warning(
                f"Cannot find ZPE in quantum data for '{species_def.get('name')}'. "
                "ZeroEnergy = baseline-corrected EleEnergy only (no ZPE added)."
            )
            total_hartree = corrected_ele_energy
        else:
            total_hartree = corrected_ele_energy + scaled_zpe
            logger.info(
                f"  {species_def.get('name')}: EleEnergy={ele_energy:.8f} Ha, "
                f"EleEnergy_baseline={energy_baseline:.8f} Ha, "
                f"corrected_EleEnergy={corrected_ele_energy:.8f} Ha, "
                f"scaled_zpe={scaled_zpe:.8f} Ha → "
                f"ZeroEnergy={total_hartree * HARTREE_TO_KCAL:.6f} kcal/mol"
            )

        zero_energy_kcal = total_hartree * HARTREE_TO_KCAL
        return zero_energy_kcal

    # ---------- 处理所有条目 ----------
    for species_def in species_list:
        name = species_def.get("name")
        species_type = str(species_def.get("type", "well")).lower()
        gaussian_file = get_config_value(species_def, "gaussian_file", "file", "path")
        if not name or not gaussian_file:
            continue

        qdata_dict = find_quantum_data(gaussian_file, quantum_results)
        if qdata_dict is None:
            logger.warning(f"No quantum data found for {gaussian_file} (species: {name})")
            continue

        # 计算 ZeroEnergy
        zero_energy = compute_zero_energy(species_def, qdata_dict)

        # 兼容手动覆盖：如果 YAML 显式给出了 GroundEnergy，则优先使用
        manual_zero_energy = get_config_value(species_def, "ZeroEnergy", "zero_energy")
        manual_ground_energy = get_config_value(species_def, "GroundEnergy", "ground_energy")
        if manual_zero_energy is not None:
            logger.info(f"  {name}: using manual ZeroEnergy={manual_zero_energy} kcal/mol")
            zero_energy = float(manual_zero_energy)
        elif manual_ground_energy is not None:
            logger.info(f"  {name}: using manual GroundEnergy={manual_ground_energy} kcal/mol")
            zero_energy = float(manual_ground_energy)

        connects = species_def.get("connects") or []
        from_species = get_config_value(species_def, "from_species")
        to_species = get_config_value(species_def, "to_species")
        if (not from_species or not to_species) and len(connects) >= 2:
            from_species, to_species = connects[0], connects[1]

        if species_type == "barrier" and from_species and to_species:
            # ---- 能垒 ----
            barrier_config = MESSBarrierConfig(
                name=name,
                from_species=from_species,
                to_species=to_species,
                quantum_data=qdata_dict["qdata"],
                correction=qdata_dict["correction"],
                ZeroEnergy=zero_energy,  # 注意大写 Z
                stoichiometry=species_def.get("stoichiometry"),
                symmetry_factor=species_def.get("symmetry_factor", 1.0),
                comment=species_def.get("comment", ""),
                method=species_def.get("method", "RRHO"),
                core_type=species_def.get("core_type", "RigidRotor"),
                electronic_levels=species_def.get("electronic_levels", [{"energy": 0.0, "degeneracy": 1.0}]),
                is_bimolecular=species_def.get("is_bimolecular", False),
                potential_prefactor=species_def.get("potential_prefactor"),
                potential_power_exponent=species_def.get("potential_power_exponent"),
            )
            barrier_configs[name] = barrier_config
            logger.info(
                f"Created barrier {name} ({from_species} -> {to_species})")
        else:
            # ---- 物种 ----
            species_config = MESSSpeciesConfig(
                name=name,
                type=species_type,
                quantum_data=qdata_dict["qdata"],
                correction=qdata_dict["correction"],
                ZeroEnergy=zero_energy,
                stoichiometry=species_def.get("stoichiometry"),
                symmetry_factor=species_def.get("symmetry_factor", 1.0),
                comment=species_def.get("comment", ""),
                method=species_def.get("method", "RRHO"),
                core_type=species_def.get("core_type", "RigidRotor"),
                electronic_levels=species_def.get("electronic_levels", [{"energy": 0.0, "degeneracy": 1.0}]),
            )
            species_configs[name] = species_config
            logger.info(f"Created species {name} ({species_type})")

    logger.info(f"Created {len(species_configs)} species and {len(barrier_configs)} barriers")
    return species_configs, barrier_configs

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
                            help="Frequency_factor override for vibrational frequencies")
    arg_parser.add_argument("--zpe-factor", type=float,
                            help="zpe_factor override for zero-point energy")
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
        raise ConfigFileNotFoundError(config_file)
    
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
    model_config = config.get("mess_model") or config.get("mess_global", {})
    
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
    scaling_factor: Optional[float] = None,
    zpe_factor: Optional[float] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Process all quantum chemistry data files.
    
    Args:
        config: Configuration dictionary
        scaling_factor: Optional frequency scaling factor (overrides config)
        zpe_factor: Optional ZPE scaling factor (overrides config)
        
    Returns:
        Dictionary mapping filenames to (QuantumData, CorrectionResult) tuples
    """
    input_config = config.get("input", {})
    quantum_config = config.get("quantum", {})
    processing_config = config.get("processing", {})
    
    # Get correction factors
    frequency_factor, config_zpe_factor = get_quantum_factors(quantum_config)
    if scaling_factor is None:
        scaling_factor = frequency_factor
    if zpe_factor is None:
        zpe_factor = config_zpe_factor
    
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
        Frequency_factor=scaling_factor,
        zpe_factor=zpe_factor,
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
            correction = corrector.correct_frequencies(
                qdata,
                Frequency_factor=scaling_factor,
                zpe_factor=zpe_factor,
            )
            
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
    
    processed_files.clear()
    processed_files.update(results)

    logger.info(f"Processed {len(results)} out of {len(files_to_process)} files")
    return results





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
    gaussian_outputs = input_cfg.get("gaussian_outputs")
    if not files and not gaussian_outputs:
        errors.append("input.files or input.gaussian_outputs must contain at least one file path")
    
    # --- quantum section ---
    quantum_cfg = config.get("quantum", {})
    if not quantum_cfg:
        errors.append("Missing required section: quantum")
    else:
        try:
            frequency_factor, zpe_factor = get_quantum_factors(quantum_cfg)
        except (TypeError, ValueError) as exc:
            errors.append(f"quantum scaling factors must be numeric: {exc}")
        else:
            if frequency_factor <= 0:
                errors.append(
                    f"quantum.Frequency_factor must be positive, got {frequency_factor}"
                )
            if zpe_factor <= 0:
                errors.append(
                    f"quantum.zpe_factor must be positive, got {zpe_factor}"
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
    baseline = get_config_value(network_cfg, "EleEnergy_baseline", "ele_energy_baseline", default=0.0)
    try:
        float(baseline)
    except (TypeError, ValueError):
        errors.append(f"reaction_network.EleEnergy_baseline must be numeric, got {baseline}")

    species_list = network_cfg.get("species", [])
    species_names = {s["name"] for s in species_list if "name" in s}
    for sp in species_list:
        if str(sp.get("type", "")).lower() == "barrier":
            connects = sp.get("connects") or []
            from_sp = sp.get("from_species")
            to_sp = sp.get("to_species")
            if (not from_sp or not to_sp) and len(connects) >= 2:
                from_sp, to_sp = connects[0], connects[1]
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
    frequency_scaling_factor: float = 1.0  # Deprecated alias for Frequency_factor
    Frequency_factor: float = 1.0
    zpe_factor: float = 1.0
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
        frequency_factor, zpe_factor = get_quantum_factors(quantum_cfg)
        
        return cls(
            input_files=(
                input_cfg.get("files", [])
                or ([input_cfg["gaussian_outputs"]] if input_cfg.get("gaussian_outputs") else [])
            ),
            frequency_scaling_factor=frequency_factor,
            Frequency_factor=frequency_factor,
            zpe_factor=zpe_factor,
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
                       help="Frequency_factor override for vibrational frequencies")
    parser.add_argument("--zpe-factor", type=float,
                       help="zpe_factor override for zero-point energy")
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
        quantum_results = process_quantum_data(config, args.scaling, args.zpe_factor)
        
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
        
    except (ConfigFileNotFoundError, MESSFileNotFoundError, FileNotFoundError) as e:
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
