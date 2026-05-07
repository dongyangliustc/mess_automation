"""
Processor module for frequency scaling and quantum data correction.

This module provides functionality to:
- Apply scaling factors to harmonic frequencies
- Convert units between different conventions
- Handle imaginary frequencies appropriately
- Perform basic data validation and cleaning
- Interface with the parser module for data flow
"""

import re
import copy
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .parser import QuantumData, GaussianParser
    from .exceptions import (
        DataValidationError, ScalingFactorError, UnitConversionError,
        ImaginaryFrequencyError, GaussianFileParseError
    )
    from .error_handler import wrap_with_error_handler, handle_error
except ImportError:
    from parser import QuantumData, GaussianParser
    from exceptions import (
        DataValidationError, ScalingFactorError, UnitConversionError,
        ImaginaryFrequencyError, GaussianFileParseError
    )
    from error_handler import wrap_with_error_handler, handle_error

# Configure logging
logger = logging.getLogger(__name__)


def get_first_defined(*values: Optional[float], default: float) -> float:
    """Return the first non-None float value, or *default*."""
    for value in values:
        if value is not None:
            return float(value)
    return default


@dataclass
class CorrectionResult:
    """Container for corrected quantum chemistry data."""
    # Original data reference
    original_data: QuantumData
    
    # Corrected frequencies
    scaled_frequencies: List[float]
    
    # Applied corrections
    scaling_factor: float
    frequency_factor: Optional[float] = None
    zpe_factor: float = 1.0
    corrections_applied: List[str] = field(default_factory=list)
    
    # Derived properties
    scaled_zpe: Optional[float] = None  # Hartree, matching QuantumData.zero_point_energy
    scaled_thermal: Optional[float] = None
    total_energy: Optional[float] = None  # Total energy (SCF + ZPE) in kcal/mol
    
    # Imaginary frequency handling
    imaginary_frequency: Optional[float] = None  # Absolute value of imaginary frequency for TS
    real_frequencies: List[float] = field(default_factory=list)  # Real frequencies only (for TS)
    
    # Barrier depth calculations
    forward_barrier: Optional[float] = None  # TS energy - reactant energy (kcal/mol)
    reverse_barrier: Optional[float] = None  # TS energy - product energy (kcal/mol)
    
    # Metadata
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Compute derived properties after initialization."""
        if self.frequency_factor is None:
            self.frequency_factor = self.scaling_factor
        else:
            self.scaling_factor = self.frequency_factor

        # Compute scaled ZPE if original ZPE exists
        if self.original_data.zero_point_energy is not None:
            self._compute_scaled_zpe()
        
        # Extract imaginary frequency if present (for transition states)
        self._extract_imaginary_frequency()
        
        # Compute total energy (SCF + ZPE) in kcal/mol
        self._compute_total_energy()
    
    def _compute_scaled_zpe(self):
        """Compute scaled ZPE in Hartree using the dedicated ZPE factor."""
        original_zpe = self.original_data.zero_point_energy
        if original_zpe is not None:
            self.scaled_zpe = original_zpe * self.zpe_factor
            logger.debug(
                f"Scaled ZPE: raw={original_zpe:.8f} Hartree, "
                f"zpe_factor={self.zpe_factor}, scaled={self.scaled_zpe:.8f} Hartree"
            )
        else:
            self.scaled_zpe = None
            
        if "zpe_scaling" not in self.corrections_applied:
            self.corrections_applied.append("zpe_scaling")
    
    def _extract_imaginary_frequency(self):
        """Extract imaginary frequency for transition states."""
        imaginary_modes = [
            (index, frequency)
            for index, frequency in enumerate(self.original_data.frequencies)
            if frequency < 0
        ]

        if imaginary_modes:
            _, original_imag = min(imaginary_modes, key=lambda item: item[1])
            self.imaginary_frequency = abs(original_imag) * self.frequency_factor

            if len(self.scaled_frequencies) == len(self.original_data.frequencies):
                self.real_frequencies = [
                    scaled_freq
                    for index, scaled_freq in enumerate(self.scaled_frequencies)
                    if self.original_data.frequencies[index] >= 0
                ]
            else:
                self.real_frequencies = [f for f in self.scaled_frequencies if f >= 0]

            logger.debug(
                f"Extracted scaled imaginary frequency: {self.imaginary_frequency:.2f} cm^-1 "
                f"(Frequency_factor: {self.frequency_factor})"
            )
            logger.debug(
                f"Real frequencies count: {len(self.real_frequencies)} "
                f"(original had {len(self.original_data.frequencies)})"
            )
        else:
            # No imaginary frequencies, all are real
            self.real_frequencies = [f for f in self.scaled_frequencies if f >= 0]
            logger.debug(f"No imaginary frequencies found, real frequencies count: {len(self.real_frequencies)}")
    
    def _compute_total_energy(self):
        """Compute total energy (SCF + ZPE) in kcal/mol."""
        # Get SCF energy in Hartree
        scf_energy_hartree = self.original_data.scf_energy
        
        # Get ZPE in Hartree (either original or scaled)
        zpe_hartree = None
        if self.scaled_zpe is not None:
            zpe_hartree = self.scaled_zpe
        elif self.original_data.zero_point_energy is not None:
            zpe_hartree = self.original_data.zero_point_energy
        
        # If we have both SCF energy and ZPE, compute total energy
        if scf_energy_hartree is not None and zpe_hartree is not None:
            total_hartree = scf_energy_hartree + zpe_hartree
            self.total_energy = UnitConverter.convert_energy(
                total_hartree, "hartree", "kcal/mol"
            )

            logger.debug(
                f"Computed total energy: SCF={scf_energy_hartree:.8f} Ha + "
                f"ZPE={zpe_hartree:.8f} Ha = {self.total_energy:.2f} kcal/mol"
            )
        else:
            # If missing data, use available energy information
            if scf_energy_hartree is not None:
                # Only SCF energy available
                self.total_energy = UnitConverter.convert_energy(
                    scf_energy_hartree, "hartree", "kcal/mol"
                )
                logger.debug(f"Using SCF energy only: {self.total_energy:.2f} kcal/mol")
            elif zpe_hartree is not None:
                # Only ZPE available
                self.total_energy = UnitConverter.convert_energy(
                    zpe_hartree, "hartree", "kcal/mol"
                )
                logger.debug(f"Using ZPE only: {self.total_energy:.2f} kcal/mol")
            else:
                logger.warning("No energy information available for total energy calculation")
    
    def get_frequencies_string(self, units: str = "1/cm") -> str:
        """Return corrected frequencies as formatted string for MESS input."""
        # For transition states with imaginary frequencies, only output real frequencies
        # Frequency count should be N-1 if there's an imaginary frequency
        if self.imaginary_frequency is not None:
            freq_list = self.real_frequencies
        else:
            freq_list = [f for f in self.scaled_frequencies if f >= 0]
        
        lines = [f"Frequencies[{units}] {len(freq_list):>15}"]
        
        # Format frequencies in columns of 3
        for i in range(0, len(freq_list), 3):
            chunk = freq_list[i:i+3]
            line = "  " + " ".join(f"{f:>10.2f}" for f in chunk)
            lines.append(line)
        
        return "\n".join(lines) + "\n"
    



class FrequencyCorrector:
    """
    Main class for frequency scaling and correction operations.
    
    This class handles:
    - Application of scaling factors to harmonic frequencies
    - Unit conversions for frequencies and energies
    - Special handling of imaginary frequencies
    - Data validation and error checking
    """
    
    def __init__(self, scaling_factor: Optional[float] = None,
                 handle_imaginary: str = "abs",  # Take absolute value for MESS output
                 validate_input: bool = True,
                 Frequency_factor: Optional[float] = None,
                 zpe_factor: Optional[float] = None,
                 frequency_factor: Optional[float] = None):
        """
        Initialize frequency corrector.
        
        Args:
            scaling_factor: Deprecated alias for ``Frequency_factor``.
            Frequency_factor: Multiplicative factor to apply to frequencies.
            zpe_factor: Multiplicative factor to apply to zero-point energy.
            frequency_factor: Lower-case alias for ``Frequency_factor``.
            handle_imaginary: How to handle imaginary frequencies:
                "abs": Take absolute value (make positive)
                "keep": Keep negative values
                "remove": Remove imaginary frequencies
                "warn": Keep but warn
            validate_input: Validate input data before processing
        """
        resolved_frequency_factor = get_first_defined(
            Frequency_factor, frequency_factor, scaling_factor, default=1.0
        )
        resolved_zpe_factor = 1.0 if zpe_factor is None else float(zpe_factor)

        self.Frequency_factor = resolved_frequency_factor
        self.frequency_factor = resolved_frequency_factor
        self.scaling_factor = resolved_frequency_factor
        self.zpe_factor = resolved_zpe_factor
        self.handle_imaginary = handle_imaginary
        self.validate_input = validate_input
        
        # Allowed imaginary handling methods
        self._allowed_imaginary_methods = ["abs", "keep", "remove", "warn"]
        
        if handle_imaginary not in self._allowed_imaginary_methods:
            raise ValueError(
                f"Invalid handle_imaginary: {handle_imaginary}. "
                f"Allowed: {self._allowed_imaginary_methods}"
            )

    
    @wrap_with_error_handler
    def correct_frequencies(self, qdata: QuantumData,
                           scaling_factor: Optional[float] = None,
                           Frequency_factor: Optional[float] = None,
                           zpe_factor: Optional[float] = None,
                           frequency_factor: Optional[float] = None) -> CorrectionResult:
        """
        Apply frequency scaling to a QuantumData object.
        
        Args:
            qdata: QuantumData object with frequencies to correct
            scaling_factor: Deprecated alias for ``Frequency_factor``.
            Frequency_factor: Optional frequency scaling factor.
            zpe_factor: Optional ZPE scaling factor.
            frequency_factor: Lower-case alias for ``Frequency_factor``.
            
        Returns:
            CorrectionResult object with scaled frequencies. Returns a failed
            CorrectionResult (success=False) if validation fails.
        """
        resolved_frequency_factor = get_first_defined(
            Frequency_factor, frequency_factor, scaling_factor,
            default=self.frequency_factor,
        )
        resolved_zpe_factor = self.zpe_factor if zpe_factor is None else float(zpe_factor)
        
        # Validate scaling factors
        if resolved_frequency_factor <= 0:
            raise ScalingFactorError(resolved_frequency_factor)
        if resolved_zpe_factor <= 0:
            raise ScalingFactorError(resolved_zpe_factor)

        # Validate input if requested
        if self.validate_input:
            validation_errors = self._validate_input(qdata)
            if validation_errors:
                error_msg = "; ".join(validation_errors)
                logger.warning(f"Input validation failed: {error_msg}")
                return CorrectionResult(
                    original_data=qdata,
                    scaled_frequencies=[],
                    scaling_factor=resolved_frequency_factor,
                    frequency_factor=resolved_frequency_factor,
                    zpe_factor=resolved_zpe_factor,
                    corrections_applied=[],
                    success=False,
                    error_message=f"Validation failed: {error_msg}",
                )
        
        # Apply scaling to frequencies
        scaled_freqs = self._apply_scaling(qdata.frequencies, resolved_frequency_factor)
        
        # Handle imaginary frequencies
        scaled_freqs = self._handle_imaginary_frequencies(scaled_freqs, qdata.frequencies)
        
        # Create correction result
        result = CorrectionResult(
            original_data=qdata,
            scaled_frequencies=scaled_freqs,
            scaling_factor=resolved_frequency_factor,
            frequency_factor=resolved_frequency_factor,
            zpe_factor=resolved_zpe_factor,
            corrections_applied=["frequency_scaling"]
        )
        
        logger.info(
            f"Applied Frequency_factor={resolved_frequency_factor} to "
            f"{len(scaled_freqs)} frequencies"
        )
        if qdata.zero_point_energy is not None:
            logger.info(
                f"Applied zpe_factor={resolved_zpe_factor}: "
                f"{qdata.zero_point_energy:.8f} -> {result.scaled_zpe:.8f} Hartree"
            )
        logger.info(f"Imaginary frequencies: {sum(1 for f in scaled_freqs if f < 0)}")
        
        return result
    
    def calculate_barrier_depths(self, ts_correction: CorrectionResult,
                               reactant_energy: Optional[float] = None,
                               product_energy: Optional[float] = None,
                               energy_units: str = "hartree") -> CorrectionResult:
        """
        Calculate forward and reverse barrier depths for a transition state.
        
        Args:
            ts_correction: CorrectionResult for the transition state
            reactant_energy: Energy of reactant species (in specified units)
            product_energy: Energy of product species (in specified units)
            energy_units: Units of input energies (hartree, kcal/mol, etc.)
            
        Returns:
            Updated CorrectionResult with forward_barrier and reverse_barrier fields
        """
        # Ensure we have SCF energy for TS
        if ts_correction.original_data.scf_energy is None:
            logger.warning("Transition state has no SCF energy, cannot calculate barriers")
            return ts_correction
        
        ts_energy_hartree = ts_correction.original_data.scf_energy
        
        # Use UnitConverter from current module
        if reactant_energy is not None:
            reactant_energy_hartree = UnitConverter.convert_energy(
                reactant_energy, energy_units, "hartree"
            )
            # Calculate forward barrier (TS - reactant) and convert to kcal/mol
            forward_barrier_hartree = ts_energy_hartree - reactant_energy_hartree
            ts_correction.forward_barrier = UnitConverter.convert_energy(
                forward_barrier_hartree, "hartree", "kcal/mol"
            )
            logger.debug(f"Forward barrier: TS={ts_energy_hartree} - Reactant={reactant_energy_hartree} = {forward_barrier_hartree} hartree -> {ts_correction.forward_barrier} kcal/mol")
        
        if product_energy is not None:
            product_energy_hartree = UnitConverter.convert_energy(
                product_energy, energy_units, "hartree"
            )
            # Calculate reverse barrier (TS - product) and convert to kcal/mol
            reverse_barrier_hartree = ts_energy_hartree - product_energy_hartree
            ts_correction.reverse_barrier = UnitConverter.convert_energy(
                reverse_barrier_hartree, "hartree", "kcal/mol"
            )
            logger.debug(f"Reverse barrier: TS={ts_energy_hartree} - Product={product_energy_hartree} = {reverse_barrier_hartree} hartree -> {ts_correction.reverse_barrier} kcal/mol")
        
        logger.info(f"Barrier depths calculated: forward={ts_correction.forward_barrier}, "
                   f"reverse={ts_correction.reverse_barrier} kcal/mol")
        
        return ts_correction
    
    def _apply_scaling(self, frequencies: List[float], 
                      scaling_factor: float) -> List[float]:
        """Apply scaling factor to frequency list."""
        return [f * scaling_factor for f in frequencies]
    
    def _handle_imaginary_frequencies(self, scaled_freqs: List[float],
                                     original_freqs: List[float]) -> List[float]:
        """Handle imaginary frequencies according to specified method."""
        processed_freqs = scaled_freqs.copy()
        
        for i, (scaled, original) in enumerate(zip(scaled_freqs, original_freqs)):
            if scaled < 0:  # This is an imaginary frequency
                if self.handle_imaginary == "abs":
                    processed_freqs[i] = abs(scaled)
                    logger.debug(f"Converted imaginary frequency {scaled:.2f} to {abs(scaled):.2f}")
                
                elif self.handle_imaginary == "remove":
                    # Note: This changes the frequency count
                    logger.warning(f"Removing imaginary frequency {scaled:.2f}")
                    # Mark for removal by setting to None
                    processed_freqs[i] = None
                
                elif self.handle_imaginary == "warn":
                    logger.warning(f"Keeping imaginary frequency {scaled:.2f}")
                    # Keep as is
                
                elif self.handle_imaginary == "keep":
                    # Keep as is, no warning
                    pass
        
        # Remove frequencies marked for removal
        if self.handle_imaginary == "remove":
            processed_freqs = [f for f in processed_freqs if f is not None]
        
        return processed_freqs
    
    def _validate_input(self, qdata: QuantumData) -> List[str]:
        """Validate input QuantumData object."""
        errors = []

        if not qdata.convergence_status:
            errors.append("Calculation did not converge")

        if not qdata.atoms:
            errors.append("No atoms found in input data")
        
        # Check if frequencies exist
        if not qdata.frequencies:
            errors.append("No frequencies found in input data")
        
        # Check for reasonable frequency values
        for freq in qdata.frequencies:
            if abs(freq) > 10000:  # Unusually high
                errors.append(f"Unusually high frequency: {freq} cm^-1")
                break
        
        # Check scaling factor
        if self.scaling_factor <= 0:
            errors.append(f"Invalid scaling factor: {self.scaling_factor}")
        
        # Check for too many imaginary frequencies
        num_imaginary = sum(1 for f in qdata.frequencies if f < 0)
        if num_imaginary > 3:
            errors.append(f"Too many imaginary frequencies: {num_imaginary}")
        
        return errors
    
    def correct_file(self, filepath: Union[str, Path],
                    scaling_factor: Optional[float] = None,
                    zpe_factor: Optional[float] = None) -> Tuple[QuantumData, CorrectionResult]:
        """
        Parse a Gaussian file and correct its frequencies.
        
        Args:
            filepath: Path to Gaussian output file
            scaling_factor: Optional frequency scaling factor (overrides default)
            zpe_factor: Optional ZPE scaling factor (overrides default)
            
        Returns:
            Tuple of (parsed QuantumData, CorrectionResult)
        """
        # Parse the file
        parser = GaussianParser()
        qdata = parser.parse_file(filepath)
        
        if not qdata.success:
            logger.error(f"Failed to parse {filepath}: {qdata.error_message}")
            return qdata, CorrectionResult(
                original_data=qdata,
                scaled_frequencies=[],
                scaling_factor=scaling_factor or self.scaling_factor,
                frequency_factor=scaling_factor or self.frequency_factor,
                zpe_factor=zpe_factor or self.zpe_factor,
                success=False,
                error_message=f"Parse error: {qdata.error_message}"
            )
        
        # Correct frequencies
        result = self.correct_frequencies(qdata, scaling_factor, zpe_factor=zpe_factor)
        
        return qdata, result
    
    def process_directory(self, directory: Union[str, Path], 
                         pattern: str = "*.out", 
                         recursive: bool = False,
                         scaling_factor: Optional[float] = None,
                         zpe_factor: Optional[float] = None) -> Dict[str, Tuple[QuantumData, CorrectionResult]]:
        """
        Process all Gaussian files in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Search recursively
            scaling_factor: Optional frequency scaling factor
            zpe_factor: Optional ZPE scaling factor
            
        Returns:
            Dictionary mapping filenames to (QuantumData, CorrectionResult) tuples
        """
        directory = Path(directory)
        results = {}
        
        # Find files
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        logger.info(f"Found {len(files)} files matching pattern '{pattern}'")
        
        # Process each file
        for filepath in files:
            try:
                qdata, correction = self.correct_file(filepath, scaling_factor, zpe_factor)
                if correction.success:
                    results[str(filepath)] = (qdata, correction)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        logger.info(f"Successfully processed {len(results)} out of {len(files)} files")
        return results


class UnitConverter:
    """
    Utility class for unit conversions in quantum chemistry.
    
    Supports conversions between:
    - Energy units: Hartree, kcal/mol, kJ/mol, eV, cm^-1
    - Length units: Angstrom, Bohr
    - Frequency units: cm^-1, THz, eV, Hartree
    """
    
    # Conversion constants
    HARTREE_TO_KCAL_MOL = 627.509474
    HARTREE_TO_KJ_MOL = 2625.49962
    HARTREE_TO_EV = 27.21138602
    HARTREE_TO_CM = 219474.6313702
    
    KCAL_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_MOL
    KJ_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KJ_MOL
    EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
    CM_TO_HARTREE = 1.0 / HARTREE_TO_CM
    
    # Length conversions
    BOHR_TO_ANGSTROM = 0.52917721067
    ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
    
    # Frequency conversions
    CM_TO_THZ = 0.0299792458  # 1 cm^-1 = 0.0299792458 THz
    THZ_TO_CM = 1.0 / CM_TO_THZ
    
    @staticmethod
    def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
        """Convert energy between different units."""
        # First convert to Hartree (internal standard)
        if from_unit.lower() in ["hartree", "au", "a.u."]:
            hartree = value
        elif from_unit.lower() in ["kcal/mol", "kcal", "kcalmol"]:
            hartree = value * UnitConverter.KCAL_MOL_TO_HARTREE
        elif from_unit.lower() in ["kj/mol", "kj", "kjmol"]:
            hartree = value * UnitConverter.KJ_MOL_TO_HARTREE
        elif from_unit.lower() in ["ev", "electronvolt"]:
            hartree = value * UnitConverter.EV_TO_HARTREE
        elif from_unit.lower() in ["cm^-1", "cm-1", "cm", "wavenumber"]:
            hartree = value * UnitConverter.CM_TO_HARTREE
        else:
            raise ValueError(f"Unknown from_unit: {from_unit}")
        
        # Convert from Hartree to target unit
        if to_unit.lower() in ["hartree", "au", "a.u."]:
            return hartree
        elif to_unit.lower() in ["kcal/mol", "kcal", "kcalmol"]:
            return hartree * UnitConverter.HARTREE_TO_KCAL_MOL
        elif to_unit.lower() in ["kj/mol", "kj", "kjmol"]:
            return hartree * UnitConverter.HARTREE_TO_KJ_MOL
        elif to_unit.lower() in ["ev", "electronvolt"]:
            return hartree * UnitConverter.HARTREE_TO_EV
        elif to_unit.lower() in ["cm^-1", "cm-1", "cm", "wavenumber"]:
            return hartree * UnitConverter.HARTREE_TO_CM
        else:
            raise ValueError(f"Unknown to_unit: {to_unit}")
    
    @staticmethod
    def convert_length(value: float, from_unit: str, to_unit: str) -> float:
        """Convert length between Angstrom and Bohr."""
        if from_unit.lower() in ["angstrom", "ang", "a"]:
            angstrom = value
        elif from_unit.lower() in ["bohr", "b", "au", "a.u."]:
            angstrom = value * UnitConverter.BOHR_TO_ANGSTROM
        else:
            raise ValueError(f"Unknown from_unit: {from_unit}")
        
        if to_unit.lower() in ["angstrom", "ang", "a"]:
            return angstrom
        elif to_unit.lower() in ["bohr", "b", "au", "a.u."]:
            return angstrom * UnitConverter.ANGSTROM_TO_BOHR
        else:
            raise ValueError(f"Unknown to_unit: {to_unit}")
    
    @staticmethod
    def convert_frequency(value: float, from_unit: str, to_unit: str) -> float:
        """Convert frequency between different units."""
        # First convert to cm^-1 (internal standard for frequencies)
        if from_unit.lower() in ["cm^-1", "cm-1", "cm", "wavenumber"]:
            cm = value
        elif from_unit.lower() in ["thz", "terahertz"]:
            cm = value * UnitConverter.THZ_TO_CM
        elif from_unit.lower() in ["ev", "electronvolt"]:
            # Convert eV to cm^-1 via Hartree
            hartree = value * UnitConverter.EV_TO_HARTREE
            cm = hartree * UnitConverter.HARTREE_TO_CM
        elif from_unit.lower() in ["hartree", "au", "a.u."]:
            cm = value * UnitConverter.HARTREE_TO_CM
        else:
            raise ValueError(f"Unknown from_unit: {from_unit}")
        
        # Convert from cm^-1 to target unit
        if to_unit.lower() in ["cm^-1", "cm-1", "cm", "wavenumber"]:
            return cm
        elif to_unit.lower() in ["thz", "terahertz"]:
            return cm * UnitConverter.CM_TO_THZ
        elif to_unit.lower() in ["ev", "electronvolt"]:
            hartree = cm * UnitConverter.CM_TO_HARTREE
            return hartree * UnitConverter.HARTREE_TO_EV
        elif to_unit.lower() in ["hartree", "au", "a.u."]:
            return cm * UnitConverter.CM_TO_HARTREE
        else:
            raise ValueError(f"Unknown to_unit: {to_unit}")


def create_molecule_object(qdata: QuantumData, correction: Optional[CorrectionResult] = None,
                           name: str = "Molecule", species_type: str = "RRHO",
                           symmetry_factor: float = 1.0, GroundEnergy: Optional[float] = None,
                           ZeroEnergy: Optional[float] = None,
                           ground_energy: Optional[float] = None,
                           zero_energy: Optional[float] = None) -> Dict[str, Any]:
    """
    Create a standardized molecule dictionary for template rendering.

    Args:
        qdata: Parsed quantum data
        correction: Optional frequency correction results
        name: Molecule/species name
        species_type: Type in MESS (RRHO, Bimolecular, etc.)
        symmetry_factor: Rotational symmetry factor
        GroundEnergy: Ground state energy relative to reference
        ZeroEnergy/zero_energy: Species zero energy (kcal/mol) from YAML.
                                This takes priority over computed energies.

    Returns:
        Dictionary with all data needed for template rendering
    """
    effective_ground_energy = (
        ground_energy if ground_energy is not None
        else (GroundEnergy if GroundEnergy is not None else 0.0)
    )
    effective_zero_energy = zero_energy if zero_energy is not None else ZeroEnergy

    # Use corrected frequencies if available, otherwise original.
    # Parser stores Gaussian zero-point correction in Hartree.
    if correction and correction.success:
        frequencies = correction.scaled_frequencies
        zpe_hartree = (
            correction.scaled_zpe
            if correction.scaled_zpe is not None
            else qdata.zero_point_energy
        )
        total_energy = correction.total_energy
    else:
        frequencies = qdata.frequencies
        zpe_hartree = qdata.zero_point_energy
        total_energy = None

        # Compute total energy if possible even without correction
        if qdata.scf_energy is not None:
            if zpe_hartree is not None:
                total_energy = UnitConverter.convert_energy(
                    qdata.scf_energy + zpe_hartree, "hartree", "kcal/mol"
                )
            else:
                total_energy = UnitConverter.convert_energy(qdata.scf_energy, "hartree", "kcal/mol")
        elif zpe_hartree is not None:
            total_energy = UnitConverter.convert_energy(zpe_hartree, "hartree", "kcal/mol")

    zpe_kcal_mol = (
        UnitConverter.convert_energy(zpe_hartree, "hartree", "kcal/mol")
        if zpe_hartree is not None
        else None
    )

    # Count imaginary frequencies (in scaled frequencies, imaginary has been converted to abs)
    num_imaginary = sum(1 for f in qdata.frequencies if f < 0)

    # Build real and imaginary frequency lists from correction result
    if correction and correction.success:
        # Real frequencies: scaled, without imaginary
        real_freqs = correction.real_frequencies
        # Imaginary frequency: scaled absolute value (already computed in correction)
        imag_freq = correction.imaginary_frequency
    else:
        real_freqs = [f for f in frequencies if f >= 0]
        imag_freq = None

    # Create molecule dictionary
    molecule = {
        "name": name,
        "type": species_type,
        "atoms": qdata.atoms,
        "num_atoms": qdata.num_atoms,
        "frequencies": frequencies,
        "num_frequencies": len(frequencies),
        "num_imaginary": num_imaginary,
        "real_frequencies": real_freqs,
        "imaginary_frequency": imag_freq,
        "imaginary_frequencies": list(qdata.imaginary_frequencies),
        "scf_energy": qdata.scf_energy,
        "zero_point_energy": zpe_hartree,
        "zero_point_energy_hartree": zpe_hartree,
        "zero_point_energy_kcal_mol": zpe_kcal_mol,
        "total_energy": total_energy,
        "symmetry_factor": symmetry_factor,
        "GroundEnergy": effective_ground_energy,
        "ground_energy": effective_ground_energy,
        "multiplicity": qdata.multiplicity,
        "charge": qdata.charge,
        "convergence_status": qdata.convergence_status,
        "method_basis": qdata.method_basis,
        # ZeroEnergy from YAML takes highest priority; this is what the user specified.
        "ZeroEnergy": effective_zero_energy,
        "zero_energy": effective_zero_energy,
    }

    return molecule


if __name__ == "__main__":
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply frequency scaling to Gaussian output files"
    )
    parser.add_argument("input", help="Gaussian output file or directory")
    parser.add_argument("--factor", "-f", type=float, default=1.0,
                       help="Frequency_factor for vibrational frequencies (default: 1.0)")
    parser.add_argument("--zpe-factor", type=float, default=1.0,
                       help="zpe_factor for zero-point energy (default: 1.0)")
    parser.add_argument("--imaginary", "-i", default="abs",
                       choices=["abs", "keep", "remove", "warn"],
                       help="How to handle imaginary frequencies (default: abs)")
    parser.add_argument("--output", "-o", help="Output summary file")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Search directories recursively")
    parser.add_argument("--pattern", "-p", default="*.out",
                       help="File pattern (default: *.out)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Create corrector
    corrector = FrequencyCorrector(
        Frequency_factor=args.factor,
        zpe_factor=args.zpe_factor,
        handle_imaginary=args.imaginary
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        qdata, correction = corrector.correct_file(input_path)
        
        if correction.success:
            output = []
            output.append(f"File: {qdata.filename}")
            output.append(f"Original frequencies: {len(qdata.frequencies)}")
            output.append(f"Scaled frequencies: {len(correction.scaled_frequencies)}")
            output.append(f"Frequency_factor: {correction.frequency_factor}")
            output.append(f"zpe_factor: {correction.zpe_factor}")
            output.append(f"Imaginary frequencies: {sum(1 for f in correction.scaled_frequencies if f < 0)}")
            
            if qdata.zero_point_energy and correction.scaled_zpe:
                output.append(f"Original ZPE: {qdata.zero_point_energy:.8f} Hartree")
                output.append(f"Scaled ZPE: {correction.scaled_zpe:.8f} Hartree")
            
            result = "\n".join(output)
        else:
            result = f"Error: {correction.error_message}"
    
    elif input_path.is_dir():
        results = corrector.process_directory(
            input_path, 
            pattern=args.pattern,
            recursive=args.recursive
        )
        
        output = []
        output.append(f"Processed {len(results)} files:")
        for filename, (qdata, correction) in results.items():
            output.append(f"\n{filename}:")
            output.append(f"  Atoms: {qdata.num_atoms}")
            output.append(f"  Frequencies: {len(correction.scaled_frequencies)}")
            output.append(f"  Frequency_factor: {correction.frequency_factor}")
            output.append(f"  zpe_factor: {correction.zpe_factor}")
        
        result = "\n".join(output)
    else:
        result = f"Error: {args.input} is not a valid file or directory"
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
    else:
        print(result)
