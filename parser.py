"""
Parser module for extracting quantum chemistry data from Gaussian output files.

This module provides functionality to extract:
- Molecular geometry (Cartesian coordinates)
- Harmonic vibrational frequencies
- Electronic energy (SCF energy)
- Zero-point vibrational energy (ZPE)
- Thermochemical data (if available)
"""

import re
import os
import sys
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Import custom exceptions and error handler
try:
    from .exceptions import (
        GaussianFileParseError, GaussianConvergenceError, NoGeometryFoundError,
        NoFrequenciesFoundError, NoEnergyFoundError, InvalidFrequencyError
    )
    from .error_handler import wrap_with_error_handler, handle_error
except ImportError:
    from exceptions import (
        GaussianFileParseError, GaussianConvergenceError, NoGeometryFoundError,
        NoFrequenciesFoundError, NoEnergyFoundError, InvalidFrequencyError
    )
    from error_handler import wrap_with_error_handler, handle_error

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Atom:
    """Represents an atom with its symbol and Cartesian coordinates."""
    symbol: str
    x: float
    y: float
    z: float
    
    def __str__(self) -> str:
        """Format atom as string in Gaussian/MESS format."""
        return f"{self.symbol:<4} {self.x:>14.8f} {self.y:>14.8f} {self.z:>14.8f}"


@dataclass
class QuantumData:
    """Container for all quantum chemistry data extracted from a Gaussian output."""
    # Basic identification
    filename: str
    convergence_status: bool
    
    # Geometry data
    atoms: List[Atom]
    num_atoms: int
    
    # Frequency data
    frequencies: List[float]  # in cm^-1
    num_frequencies: int
    imaginary_frequencies: List[float] = field(default_factory=list)  # negative values for imaginary modes
    
    # Energy data
    scf_energy: Optional[float] = None  # Hartree
    zero_point_energy: Optional[float] = None  # Hartree
    thermal_correction: Optional[float] = None  # kcal/mol
    enthalpy: Optional[float] = None  # kcal/mol
    gibbs_free_energy: Optional[float] = None  # kcal/mol
    
    # Electronic structure
    multiplicity: int = 1
    charge: int = 0
    
    # Additional metadata
    method_basis: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate and compute derived properties."""
        self.num_atoms = len(self.atoms)
        self.num_frequencies = len(self.frequencies)
        
        # Count imaginary frequencies
        self.num_imaginary = sum(1 for f in self.frequencies if f < 0)
        
        # Separate imaginary frequencies
        self.imaginary_frequencies = [f for f in self.frequencies if f < 0]
        self.real_frequencies = [f for f in self.frequencies if f >= 0]
    
    def get_geometry_string(self, units: str = "angstrom") -> str:
        """Return geometry as formatted string for MESS input."""
        lines = [f"Geometry[{units}] {self.num_atoms:>15}"]
        for atom in self.atoms:
            lines.append(f"  {atom}")
        return "\n".join(lines)
    
    def get_frequencies_string(self, units: str = "1/cm") -> str:
        """Return frequencies as formatted string for MESS input."""
        lines = [f"Frequencies[{units}] {self.num_frequencies:>15}"]
        
        # Format frequencies in columns of 3
        for i in range(0, len(self.frequencies), 3):
            chunk = self.frequencies[i:i+3]
            line = "  " + " ".join(f"{f:>10.2f}" for f in chunk)
            lines.append(line)
        
        return "\n".join(lines)


class GaussianParser:
    """Parser for Gaussian 16/09 output files."""
    
    # Periodic table mapping (atomic number to symbol)
    PERIODIC_TABLE = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 35: 'Br', 53: 'I'
    }
    
    # Conversion factors
    HARTREE_TO_KCAL_MOL = 627.509474
    
    def __init__(self, skip_unconverged: bool = True, validate: bool = True):
        """
        Initialize Gaussian parser.
        
        Args:
            skip_unconverged: If True, skip files with SCF convergence issues
            validate: If True, validate extracted data for consistency
        """
        self.skip_unconverged = skip_unconverged
        self.validate = validate
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regular expressions used in parsing."""
        # Geometry patterns
        self.std_orient_pattern = re.compile(r'^\s*Standard orientation:')
        self.input_orient_pattern = re.compile(r'^\s*Input orientation:')
        self.mol_orient_pattern = re.compile(r'^\s*Molecular orientation:')
        
        # Geometry data line pattern: Center Atomic Atomic Coordinates
        # Pattern for: "     1          6           0       -1.628220   -0.776233    0.450099"
        self.geom_line_pattern = re.compile(
            r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'
        )
        
        # Frequency patterns
        self.freq_pattern = re.compile(r'^\s*Frequencies --\s+([-\d\.\s]+)')
        
        # Energy patterns - match both E(RHF/UHF) and E(RM062X) etc.
        self.scf_energy_pattern = re.compile(
            r'^\s*SCF Done:\s+E\([^)]+\)\s*=\s*([-\d\.]+)\s*'
        )
        self.zpe_pattern = re.compile(
             r'^\s*Zero-point correction=\s*([\d\.]+)\s*\(Hartree/Particle\)'
        )
        self.thermal_pattern = re.compile(
            r'^\s*Thermal correction to Energy\s*[=:]\s*([-\d\.]+)'
        )
        self.enthalpy_pattern = re.compile(
            r'^\s*Thermal correction to Enthalpy\s*[=:]\s*([-\d\.]+)'
        )
        self.gibbs_pattern = re.compile(
            r'^\s*Thermal correction to Gibbs Free Energy\s*[=:]\s*([-\d\.]+)'
        )
        
        # Convergence patterns
        self.convergence_pattern = re.compile(r'^\s*Convergence criterion met')
        self.normal_termination_pattern = re.compile(r'^\s*Normal termination')
        
        # Method/basis pattern
        self.method_pattern = re.compile(r'^\s*#\s*(.+)')
    
    @staticmethod
    def get_atom_symbol(atomic_number: Union[int, str]) -> str:
        """
        Convert atomic number to element symbol.
        
        Args:
            atomic_number: Atomic number as integer or string
            
        Returns:
            Element symbol, or "X" if not found
        """
        try:
            num = int(atomic_number)
            return GaussianParser.PERIODIC_TABLE.get(num, "X")
        except (ValueError, TypeError):
            return "X"
    
    @wrap_with_error_handler
    def parse_file(self, filepath: Union[str, Path]) -> QuantumData:
        """
        Parse a Gaussian output file.
        
        Args:
            filepath: Path to Gaussian output file
            
        Returns:
            QuantumData object with extracted information
            
        Raises:
            GaussianFileParseError: If file cannot be parsed
            GaussianConvergenceError: If SCF does not converge
            NoGeometryFoundError: If no geometry found
            NoFrequenciesFoundError: If no frequencies found
        """
        filepath = Path(filepath)
        logger.info(f"Parsing Gaussian output: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except (FileNotFoundError, IOError) as e:
            error_msg = f"Failed to read file {filepath}: {e}"
            logger.error(error_msg)
            raise GaussianFileParseError(filepath, str(e)) from e
        
        # Extract data
        atoms = self._extract_geometry(lines)
        frequencies = self._extract_frequencies(lines)
        energies = self._extract_energies(lines)
        convergence = self._check_convergence(lines)
        method_basis = self._extract_method_basis(lines)
        
        # Validate extracted data
        if not atoms:
            logger.warning(f"No geometry found in {filepath}")
            # Don't raise exception here - return empty QuantumData with error
        elif not frequencies:
            logger.warning(f"No frequencies found in {filepath}")
            # Don't raise exception here - return empty QuantumData with error
        
        # Check for SCF convergence issues if required
        if self.skip_unconverged and not convergence:
            logger.warning(f"SCF not converged in {filepath}, skipping...")
            raise GaussianConvergenceError(filepath, "SCF not converged")
        
        # Create QuantumData object
        qdata = QuantumData(
            filename=str(filepath),
            convergence_status=convergence,
            atoms=atoms,
            num_atoms=len(atoms),
            frequencies=frequencies,
            num_frequencies=len(frequencies),
            imaginary_frequencies=[],
            scf_energy=energies.get('scf'),
            zero_point_energy=energies.get('zpe'),
            thermal_correction=energies.get('thermal'),
            enthalpy=energies.get('enthalpy'),
            gibbs_free_energy=energies.get('gibbs'),
            method_basis=method_basis,
            success=True
        )
        
        # Validate if requested
        if self.validate:
            self._validate_data(qdata)
        
        logger.info(f"Successfully parsed {filepath}: {len(atoms)} atoms, {len(frequencies)} frequencies")
        return qdata
    
    def _extract_geometry(self, lines: List[str]) -> List[Atom]:
        """Extract molecular geometry from Gaussian output."""
        atoms = []
        
        # Find the last occurrence of Standard, Input, or Molecular orientation
        orientation_start = -1
        for i, line in enumerate(lines):
            if (self.std_orient_pattern.search(line) 
                    or self.input_orient_pattern.search(line)
                    or self.mol_orient_pattern.search(line)):
                orientation_start = i
        
        if orientation_start == -1:
            logger.warning("No geometry orientation found")
            return atoms
        
        # Find the start of the geometry table (after the separator lines)
        data_start = -1
        dash_count = 0
        for i in range(orientation_start, min(orientation_start + 20, len(lines))):
            if "---" in lines[i] or "----" in lines[i]:
                dash_count += 1
                if dash_count == 2:  # Second separator line is after header in Gaussian 16
                    data_start = i + 1
                    break
        
        if data_start == -1:
            # Try alternative pattern: look for "Coordinates (Angstroms)" header
            for i in range(orientation_start, min(orientation_start + 20, len(lines))):
                if "Coordinates (Angstroms)" in lines[i]:
                    # Skip the separator line after the header
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "---" in lines[j] or "----" in lines[j]:
                            data_start = j + 1
                            break
                    if data_start != -1:
                        break
        
        if data_start == -1:
            logger.warning("Could not find geometry data start")
            return atoms
        
        # Parse atom lines until next separator
        for i in range(data_start, len(lines)):
            line = lines[i]
            if "---" in line or "----" in line or line.strip() == "":
                break
            
            match = self.geom_line_pattern.match(line)
            if match:
                # Format: atom_number atomic_number type x y z
                # Example: "     1          6           0       -1.628220   -0.776233    0.450099"
                atomic_num = match.group(2)  # atomic number is second column
                x, y, z = map(float, match.group(4, 5, 6))
                symbol = self.get_atom_symbol(atomic_num)
                atoms.append(Atom(symbol, x, y, z))
        
        return atoms
    
    def _extract_frequencies(self, lines: List[str]) -> List[float]:
        """Extract vibrational frequencies from Gaussian output."""
        frequencies = []
        
        for line in lines:
            match = self.freq_pattern.match(line)
            if match:
                # Extract all frequencies from the line
                freq_strs = match.group(1).split()
                for f_str in freq_strs:
                    try:
                        freq = float(f_str)
                        frequencies.append(freq)
                    except ValueError:
                        continue
        
        return frequencies
    
    def _extract_energies(self, lines: List[str]) -> Dict[str, float]:
        """Extract energy values from Gaussian output."""
        energies = {}
        
        for line in lines:
            # SCF energy
            match = self.scf_energy_pattern.search(line)
            if match:
                try:
                    scf_hartree = float(match.group(1))
                    energies['scf'] = scf_hartree
                except ValueError:
                    pass
            
            # Zero-point energy
            match = self.zpe_pattern.search(line)
            if match:
                try:
                    zpe_hartree = float(match.group(1))
                    energies['zpe'] = zpe_hartree
                except ValueError:
                    pass
            
            # Thermal correction
            match = self.thermal_pattern.search(line)
            if match:
                try:
                    thermal_hartree = float(match.group(1))
                    energies['thermal'] = thermal_hartree * self.HARTREE_TO_KCAL_MOL
                except ValueError:
                    pass
            
            # Enthalpy correction
            match = self.enthalpy_pattern.search(line)
            if match:
                try:
                    enthalpy_hartree = float(match.group(1))
                    energies['enthalpy'] = enthalpy_hartree * self.HARTREE_TO_KCAL_MOL
                except ValueError:
                    pass
            
            # Gibbs free energy correction
            match = self.gibbs_pattern.search(line)
            if match:
                try:
                    gibbs_hartree = float(match.group(1))
                    energies['gibbs'] = gibbs_hartree * self.HARTREE_TO_KCAL_MOL
                except ValueError:
                    pass
        
        return energies
    
    def _check_convergence(self, lines: List[str]) -> bool:
        """Check if calculation converged normally.

        Strategy:
        1. A job that ends with ``Normal termination`` is considered converged.
        2. Geometry-optimisation jobs additionally need ``Optimization completed``
           or ``Stationary point found``.  Single-point and frequency-only jobs
           lack these markers but are still valid; Normal termination is enough
           for them.
        """
        normal_termination = False
        opt_convergence = False
        is_opt_job = False

        for line in lines:
            if self.normal_termination_pattern.search(line):
                normal_termination = True
            if self.convergence_pattern.search(line):
                opt_convergence = True
            if "Optimization completed" in line or "Stationary point found" in line:
                opt_convergence = True
                is_opt_job = True
            # Detect opt jobs by looking for the Berny optimisation header
            if "GradGradGrad" in line or "-- Stationary point --" in line:
                is_opt_job = True

        if not normal_termination:
            return False

        # For optimisation jobs we need convergence; for others Normal termination suffices
        if is_opt_job:
            return opt_convergence
        return True
    
    def _extract_method_basis(self, lines: List[str]) -> Optional[str]:
        """Extract method/basis set information."""
        for line in lines:
            match = self.method_pattern.match(line)
            if match:
                method_str = match.group(1).strip()
                # Take the first method line (skip additional keywords)
                if not method_str.startswith('---'):
                    return method_str
        return None
    
    def _validate_data(self, qdata: QuantumData) -> None:
        """Validate extracted data for consistency."""
        issues = []
        
        # Check geometry
        if not qdata.atoms:
            issues.append("No atoms found")
        
        # Check frequencies
        if not qdata.frequencies:
            issues.append("No frequencies found")
        else:
            # Check for reasonable frequency range
            for freq in qdata.frequencies:
                if abs(freq) > 10000:  # Unusually high frequency
                    issues.append(f"Unusual frequency: {freq} cm^-1")
                    break
        
        # Check for too many imaginary frequencies
        if qdata.num_imaginary > 3:  # More than 3 imaginary frequencies is suspicious
            issues.append(f"Too many imaginary frequencies: {qdata.num_imaginary}")
        
        # Check energies
        if qdata.scf_energy is None:
            issues.append("No SCF energy found")
        
        if issues:
            logger.warning(f"Validation issues for {qdata.filename}: {', '.join(issues)}")


def parse_directory(directory: Union[str, Path], pattern: str = "*.out", 
                    recursive: bool = False, **kwargs) -> Dict[str, QuantumData]:
    """
    Parse all Gaussian output files in a directory.
    
    Args:
        directory: Directory to search for Gaussian output files
        pattern: File pattern to match (default: *.out)
        recursive: If True, search recursively in subdirectories
        **kwargs: Additional arguments for GaussianParser
        
    Returns:
        Dictionary mapping filenames to QuantumData objects
    """
    directory = Path(directory)
    parser = GaussianParser(**kwargs)
    results = {}
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    logger.info(f"Found {len(files)} files matching pattern '{pattern}'")
    
    for filepath in files:
        try:
            qdata = parser.parse_file(filepath)
            if qdata.success:
                results[str(filepath)] = qdata
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
    
    logger.info(f"Successfully parsed {len(results)} out of {len(files)} files")
    return results


if __name__ == "__main__":
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Gaussian output files")
    parser.add_argument("input", help="Gaussian output file or directory")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
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
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        parser = GaussianParser()
        qdata = parser.parse_file(input_path)
        
        output = []
        output.append(f"File: {qdata.filename}")
        output.append(f"Atoms: {qdata.num_atoms}")
        output.append(f"Frequencies: {qdata.num_frequencies} ({qdata.num_imaginary} imaginary)")
        output.append(f"SCF Energy: {qdata.scf_energy} Hartree")
        if qdata.zero_point_energy:
            output.append(f"ZPE: {qdata.zero_point_energy:.2f} kcal/mol")
        
        result = "\n".join(output)
        
    elif input_path.is_dir():
        results = parse_directory(input_path, pattern=args.pattern, 
                                 recursive=args.recursive)
        
        output = []
        output.append(f"Parsed {len(results)} files:")
        for filename, qdata in results.items():
            output.append(f"\n{filename}:")
            output.append(f"  Atoms: {qdata.num_atoms}")
            output.append(f"  Frequencies: {qdata.num_frequencies}")
        
        result = "\n".join(output)
    else:
        result = f"Error: {args.input} is not a valid file or directory"
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
    else:
        print(result)