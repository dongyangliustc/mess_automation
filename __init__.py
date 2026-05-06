"""
MESS Automation Package
Automated generation of MESS input files from quantum chemistry data.
"""

__version__ = "1.0.0"
__author__ = "MESS Automation Team"

from .parser import GaussianParser, QuantumData, parse_directory
from .processor import FrequencyCorrector, CorrectionResult, UnitConverter, create_molecule_object
from .assembler import MESSAssembler, MESSGlobalSettings, MESSModelSettings, MESSSpeciesConfig, MESSBarrierConfig

__all__ = [
    "GaussianParser",
    "QuantumData",
    "parse_directory",
    "FrequencyCorrector",
    "CorrectionResult",
    "UnitConverter",
    "create_molecule_object",
    "MESSAssembler",
    "MESSGlobalSettings",
    "MESSModelSettings",
    "MESSSpeciesConfig",
    "MESSBarrierConfig"
]