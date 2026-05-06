"""
Assembler module for generating MESS input files using Jinja2 templates.

This module provides functionality to:
- Generate MESS input files from quantum chemistry data
- Use Jinja2 templates for flexible formatting
- Ensure proper indentation and formatting for MESS
- Combine global settings, model parameters, and species data
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict

import jinja2

try:
    from .parser import QuantumData, Atom
    from .processor import CorrectionResult, create_molecule_object
except ImportError:
    from parser import QuantumData, Atom
    from processor import CorrectionResult, create_molecule_object

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MESSGlobalSettings:
    """Global settings for MESS input file."""
    temperature_list: List[float] = field(default_factory=lambda: 
        [200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 
         750, 800, 850, 900, 950, 1000])
    pressure_list: List[float] = field(default_factory=lambda: [1.0])
    energy_step_over_temperature: float = 0.2
    excess_energy_over_temperature: float = 50
    model_energy_limit: float = 400
    calculation_method: str = "direct"
    well_cutoff: float = 20
    chemical_eigenvalue_max: float = 0.2
    reduction_method: str = "diagonalization"
    rate_output: str = "mess.out"
    log_output: Optional[str] = None
    eigenvalue_output: Optional[str] = None
    eigenvector_output: Optional[str] = None
    ped_output: Optional[str] = None


@dataclass
class MESSModelSettings:
    """Model settings for MESS input file."""
    energy_relaxation: Dict[str, Any] = field(default_factory=lambda: {
        "model": "Exponential",
        "factor": 350.0,
        "power": 0.85,
        "exponent_cutoff": 10
    })
    collision_frequency: Dict[str, Any] = field(default_factory=lambda: {
        "model": "LennardJones",
        "epsilons": [10.0, 595.38],
        "sigmas": [2.55, 4.89],
        "masses": [4.0, 78.05]
    })


@dataclass
class MESSSpeciesConfig:
    """Configuration for a MESS species."""
    name: str
    type: str  # "reactant", "well", "product", "bimolecular"
    quantum_data: QuantumData
    correction: Optional[CorrectionResult] = None
    stoichiometry: Optional[str] = None
    symmetry_factor: float = 1.0
    ground_energy: float = 0.0
    comment: str = ""
    method: str = "RRHO"
    core_type: str = "RigidRotor"
    electronic_levels: List[Dict[str, float]] = field(default_factory=lambda: [
        {"energy": 0.0, "degeneracy": 1.0}
    ])
    # For bimolecular species
    fragments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MESSBarrierConfig:
    """Configuration for a MESS barrier."""
    name: str
    from_species: str
    to_species: str
    quantum_data: QuantumData
    correction: Optional[CorrectionResult] = None
    stoichiometry: Optional[str] = None
    symmetry_factor: float = 1.0
    comment: str = ""
    method: str = "RRHO"
    core_type: str = "RigidRotor"
    electronic_levels: List[Dict[str, float]] = field(default_factory=lambda: [
        {"energy": 0.0, "degeneracy": 1.0}
    ])
    # For bimolecular barriers
    is_bimolecular: bool = False
    fragments: List[Dict[str, Any]] = field(default_factory=list)
    potential_prefactor: Optional[float] = None
    potential_power_exponent: Optional[int] = None
    # Barrier depths (kcal/mol)
    forward_barrier: Optional[float] = None
    reverse_barrier: Optional[float] = None


class MESSAssembler:
    """Main class for assembling MESS input files from templates."""
    
    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        """
        Initialize MESS assembler.
        
        Args:
            template_dir: Directory containing Jinja2 templates. If None,
                         uses default templates in package.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = Path(template_dir)
        
        # Setup Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Add custom filters
        self.env.filters['format_float'] = self._format_float
        
        # Template references
        self.global_template = None
        self.model_template = None
        self.species_template = None
        self.barrier_template = None
        
        # Load templates
        self._load_templates()
        
        # Expose Jinja2 Environment under the alias expected by tests
        self.template_env = self.env
        
        # Data storage
        self.global_settings = MESSGlobalSettings()
        self.model_settings = MESSModelSettings()
        self.species: Dict[str, MESSSpeciesConfig] = {}
        self.barriers: Dict[str, MESSBarrierConfig] = {}
        
        logger.info(f"Initialized MESS assembler with template dir: {template_dir}")
    
    def _load_templates(self):
        """Load Jinja2 templates."""
        try:
            self.global_template = self.env.get_template("global_section.jinja2")
            self.model_template = self.env.get_template("model_section.jinja2")
            self.species_template = self.env.get_template("species.jinja2")
            self.barrier_template = self.env.get_template("barrier.jinja2")
            logger.debug("Successfully loaded all templates")
        except jinja2.TemplateNotFound as e:
            logger.error(f"Template not found: {e}")
            raise
    
    @staticmethod
    def _format_float(value: float, precision: int = 8) -> str:
        """Format float for MESS output."""
        # For MESS, we need numbers with decimal point
        # If it looks like an integer, add .0
        formatted = f"{value:.{precision}f}"
        
        # Remove trailing zeros after decimal
        if '.' in formatted:
            # Remove trailing zeros
            formatted = formatted.rstrip('0')
            # Ensure we have at least one digit after decimal
            if formatted.endswith('.'):
                formatted = formatted + '0'
        
        return formatted
    
    def set_global_settings(self, settings: MESSGlobalSettings):
        """Set global settings for MESS input."""
        self.global_settings = settings
    
    def set_model_settings(self, settings: MESSModelSettings):
        """Set model settings for MESS input."""
        self.model_settings = settings
    
    def add_species(self, species: MESSSpeciesConfig):
        """Add a species to the reaction network."""
        self.species[species.name] = species
        logger.debug(f"Added species: {species.name} ({species.type})")
    
    def add_barrier(self, barrier: MESSBarrierConfig):
        """Add a barrier to the reaction network."""
        # Calculate barrier depths if not already set and we have the required species
        if barrier.forward_barrier is None or barrier.reverse_barrier is None:
            self._calculate_barrier_depths(barrier)
        
        self.barriers[barrier.name] = barrier
        logger.debug(f"Added barrier: {barrier.name} ({barrier.from_species} -> {barrier.to_species})")
    
    def _calculate_barrier_depths(self, barrier: MESSBarrierConfig):
        """Calculate forward and reverse barrier depths for a transition state."""
        # Get TS energy
        if barrier.quantum_data.scf_energy is None:
            logger.warning(f"Transition state {barrier.name} has no SCF energy, cannot calculate barriers")
            return
        
        ts_energy = barrier.quantum_data.scf_energy  # In Hartree
        
        # Get reactant and product energies
        reactant = self.species.get(barrier.from_species)
        product = self.species.get(barrier.to_species)
        
        if reactant is None:
            logger.warning(f"Reactant {barrier.from_species} not found for barrier {barrier.name}")
            return
        
        if product is None:
            logger.warning(f"Product {barrier.to_species} not found for barrier {barrier.name}")
            return
        
        if reactant.quantum_data.scf_energy is None:
            logger.warning(f"Reactant {barrier.from_species} has no SCF energy")
            return
        
        if product.quantum_data.scf_energy is None:
            logger.warning(f"Product {barrier.to_species} has no SCF energy")
            return
        
        reactant_energy = reactant.quantum_data.scf_energy
        product_energy = product.quantum_data.scf_energy
        
        # Calculate barriers in Hartree
        forward_barrier_hartree = ts_energy - reactant_energy
        reverse_barrier_hartree = ts_energy - product_energy
        
        # Convert to kcal/mol
        barrier.forward_barrier = forward_barrier_hartree * 627.509474  # Hartree to kcal/mol
        barrier.reverse_barrier = reverse_barrier_hartree * 627.509474
        
        logger.info(f"Calculated barrier depths for {barrier.name}: "
                   f"forward={barrier.forward_barrier:.2f}, reverse={barrier.reverse_barrier:.2f} kcal/mol")
    
    def _prepare_species_data(self, species: MESSSpeciesConfig) -> Dict[str, Any]:
        """Prepare species data for template rendering."""
        # Create molecule dictionary
        molecule = create_molecule_object(
            species.quantum_data, species.correction,
            name=species.name, species_type=species.type,
            symmetry_factor=species.symmetry_factor,
            ground_energy=species.ground_energy
        )
        
        # Add species-specific data
        molecule.update({
            "type": species.type,
            "comment": species.comment,
            "method": species.method,
            "core_type": species.core_type,
            "core_defined": True,
            "electronic_levels": species.electronic_levels,
            "stoichiometry": species.stoichiometry,
            "ground_energy": species.ground_energy,
        })
        
        # Add geometry string
        if species.correction and species.correction.success:
            molecule["geometry_string"] = species.quantum_data.get_geometry_string()
        else:
            molecule["geometry_string"] = species.quantum_data.get_geometry_string()
        
        # Add frequencies string
        if species.correction and species.correction.success:
            molecule["frequencies_string"] = species.correction.get_frequencies_string()
        else:
            molecule["frequencies_string"] = species.quantum_data.get_frequencies_string()
        
        # For bimolecular species
        if species.type == "bimolecular" and species.fragments:
            molecule["fragments"] = []
            for frag_data in species.fragments:
                frag = {
                    "name": frag_data.get("name", "FRAG"),
                    "method": frag_data.get("method", "RRHO"),
                    "geometry_string": frag_data.get("geometry_string", ""),
                    "frequencies_string": frag_data.get("frequencies_string", ""),
                    "zero_point_energy": frag_data.get("zero_point_energy"),
                    "electronic_levels": frag_data.get("electronic_levels", 
                                                      [{"energy": 0.0, "degeneracy": 1.0}]),
                    "core_type": frag_data.get("core_type", "RigidRotor"),
                    "symmetry_factor": frag_data.get("symmetry_factor", 1.0),
                    "core_defined": frag_data.get("core_defined", True),
                }
                molecule["fragments"].append(frag)
        
        return molecule
    
    def _prepare_barrier_data(self, barrier: MESSBarrierConfig) -> Dict[str, Any]:
        """Prepare barrier data for template rendering."""
        # Create molecule dictionary
        molecule = create_molecule_object(
            barrier.quantum_data, barrier.correction,
            name=barrier.name, species_type="barrier",
            symmetry_factor=barrier.symmetry_factor,
            ground_energy=0.0
        )
        
        # Add barrier-specific data
        molecule.update({
            "from_species": barrier.from_species,
            "to_species": barrier.to_species,
            "comment": barrier.comment,
            "method": barrier.method,
            "core_type": barrier.core_type,
            "core_defined": True,
            "electronic_levels": barrier.electronic_levels,
            "stoichiometry": barrier.stoichiometry,
            "is_bimolecular_barrier": barrier.is_bimolecular,
            "potential_prefactor": barrier.potential_prefactor,
            "potential_power_exponent": barrier.potential_power_exponent,
            "forward_barrier": barrier.forward_barrier,
            "reverse_barrier": barrier.reverse_barrier,
        })
        
        # Add geometry string
        molecule["geometry_string"] = barrier.quantum_data.get_geometry_string()
        
        # Add frequencies string
        if barrier.correction and barrier.correction.success:
            molecule["frequencies_string"] = barrier.correction.get_frequencies_string()
            # Add imaginary frequency if present
            if barrier.correction.imaginary_frequency is not None:
                molecule["imaginary_frequency"] = barrier.correction.imaginary_frequency
                # Ensure real frequencies count excludes the imaginary one
                if barrier.correction.real_frequencies:
                    molecule["real_frequencies"] = barrier.correction.real_frequencies
                    molecule["num_real_frequencies"] = len(barrier.correction.real_frequencies)
        else:
            molecule["frequencies_string"] = barrier.quantum_data.get_frequencies_string()
        
        # For bimolecular barriers
        if barrier.is_bimolecular and barrier.fragments:
            molecule["fragments"] = []
            for frag_data in barrier.fragments:
                frag = {
                    "name": frag_data.get("name", "FRAG"),
                    "num_atoms": frag_data.get("num_atoms", 0),
                    "atoms": frag_data.get("atoms", []),
                    "geometry_string": frag_data.get("geometry_string", ""),
                }
                molecule["fragments"].append(frag)
        
        return molecule
    
    def render_global_section(self) -> str:
        """Render the global section of MESS input."""
        if not self.global_template:
            raise ValueError("Global template not loaded")
        
        context = {
            "temperature_list": self.global_settings.temperature_list,
            "pressure_list": self.global_settings.pressure_list,
            "energy_step_over_temperature": self.global_settings.energy_step_over_temperature,
            "excess_energy_over_temperature": self.global_settings.excess_energy_over_temperature,
            "model_energy_limit": self.global_settings.model_energy_limit,
            "calculation_method": self.global_settings.calculation_method,
            "well_cutoff": self.global_settings.well_cutoff,
            "chemical_eigenvalue_max": self.global_settings.chemical_eigenvalue_max,
            "reduction_method": self.global_settings.reduction_method,
            "rate_output": self.global_settings.rate_output,
            "log_output": self.global_settings.log_output or "mess.log",
            "eigenvalue_output": self.global_settings.eigenvalue_output or "mess_eigval.out",
            "eigenvector_output": self.global_settings.eigenvector_output or "mess_eigvect.out",
            "ped_output": self.global_settings.ped_output or "mess_ped.out",
        }
        
        return self.global_template.render(**context)
    
    def render_model_section(self) -> str:
        """Render the model section of MESS input."""
        if not self.model_template:
            raise ValueError("Model template not loaded")
        
        context = {
            "energy_relaxation": self.model_settings.energy_relaxation,
            "collision_frequency": self.model_settings.collision_frequency,
        }
        
        return self.model_template.render(**context)
    
    def render_species_sections(self) -> str:
        """Render all species sections."""
        if not self.species_template:
            raise ValueError("Species template not loaded")
        
        sections = []
        
        for species_name, species in self.species.items():
            try:
                data = self._prepare_species_data(species)
                rendered = self.species_template.render(species=data)
                sections.append(rendered)
                logger.debug(f"Rendered species: {species_name}")
            except Exception as e:
                logger.error(f"Failed to render species {species_name}: {e}")
                raise
        
        return "\n".join(sections)
    
    def render_barrier_sections(self) -> str:
        """Render all barrier sections."""
        if not self.barrier_template:
            raise ValueError("Barrier template not loaded")
        
        sections = []
        
        for barrier_name, barrier in self.barriers.items():
            try:
                data = self._prepare_barrier_data(barrier)
                rendered = self.barrier_template.render(barrier=data)
                sections.append(rendered)
                logger.debug(f"Rendered barrier: {barrier_name}")
            except Exception as e:
                logger.error(f"Failed to render barrier {barrier_name}: {e}")
                raise
        
        return "\n".join(sections)
    
    def assemble(self) -> str:
        """Assemble complete MESS input file."""
        logger.info("Assembling MESS input file")
        
        parts = []
        
        # 1. Global section
        parts.append(self.render_global_section())
        
        # 2. Model section
        parts.append(self.render_model_section())
        
        # 3. Reactants section header (following vanillin-0.inp format)
        parts.append("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        parts.append("!***************************************************")
        parts.append("!  REACTANTS")
        parts.append("!***************************************************")
        parts.append("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        parts.append("!***************************************************")
        parts.append("")
        
        # 4. Species sections
        parts.append(self.render_species_sections())
        
        # 5. Barrier sections
        parts.append(self.render_barrier_sections())
        
        # 6. End markers
        parts.append("End")
        parts.append("End")
        
        # Combine all parts
        mess_input = "\n".join(parts)
        
        # Validate output
        self._validate_output(mess_input)
        
        logger.info(f"Successfully assembled MESS input ({len(mess_input.splitlines())} lines)")
        return mess_input
    
    def _validate_output(self, mess_input: str):
        """Validate generated MESS input for common issues."""
        lines = mess_input.splitlines()
        issues = []
        
        # Check for essential sections
        if not any("TemperatureList" in line for line in lines):
            issues.append("Missing TemperatureList")
        
        if not any("PressureList" in line for line in lines):
            issues.append("Missing PressureList")
        
        if not any("Model" in line for line in lines):
            issues.append("Missing Model section")
        
        # Check for unclosed blocks
        model_start = any(line.strip() == "Model" for line in lines)
        model_end_count = sum(1 for line in lines if line.strip() == "End")
        
        if model_start and model_end_count < 2:
            issues.append("Insufficient End markers")
        
        # Check species count
        species_count = sum(1 for line in lines if "Species" in line)
        if species_count == 0:
            issues.append("No species defined")
        
        if issues:
            logger.warning(f"Validation issues found: {', '.join(issues)}")
        else:
            logger.debug("MESS input validation passed")
    
    def write_to_file(self, output_file: Union[str, Path], overwrite: bool = False):
        """
        Write assembled MESS input to file.
        
        Args:
            output_file: Path to output file
            overwrite: If True, overwrite existing file
        """
        output_file = Path(output_file)
        
        if output_file.exists() and not overwrite:
            raise FileExistsError(
                f"Output file {output_file} already exists. "
                "Use overwrite=True to overwrite."
            )
        
        # Assemble the input
        mess_input = self.assemble()
        
        # Write to file
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(mess_input)
            logger.info(f"Written MESS input to {output_file}")
        except IOError as e:
            logger.error(f"Failed to write output file: {e}")
            raise

    # ------------------------------------------------------------------
    # Convenience render methods (thin wrappers used by tests)
    # ------------------------------------------------------------------

    def render_species_template(self, data: Dict[str, Any]) -> str:
        """
        Render the species Jinja2 template with *data*.

        This is a direct template render (no quantum-data lookup) used mainly
        by unit tests that supply pre-built molecule dictionaries.

        Args:
            data: Molecule data dictionary (keys: name, type, atoms, frequencies, …)

        Returns:
            Rendered template string.
        """
        if not self.species_template:
            raise ValueError("Species template not loaded")
        return self.species_template.render(species=data)

    def render_barrier_template(self, data: Dict[str, Any]) -> str:
        """
        Render the barrier Jinja2 template with *data*.

        Args:
            data: Barrier data dictionary.

        Returns:
            Rendered template string.
        """
        if not self.barrier_template:
            raise ValueError("Barrier template not loaded")
        return self.barrier_template.render(barrier=data)

    def render_global_template(self, data: Dict[str, Any]) -> str:
        """
        Render the global-section Jinja2 template with *data*.

        Args:
            data: Global settings dictionary.

        Returns:
            Rendered template string.
        """
        if not self.global_template:
            raise ValueError("Global template not loaded")
        return self.global_template.render(**data)

    def assemble_mess_input(
        self,
        global_settings: MESSGlobalSettings,
        reaction_network: "MESSReactionNetwork",
        molecule_objects: Dict[str, Dict[str, Any]],
        output_file: str,
    ) -> None:
        """
        High-level convenience method: set settings, render each species/barrier
        from *molecule_objects*, and write the result to *output_file*.

        This method is the counterpart of write_to_file() for callers that
        supply ready-made molecule dictionaries instead of MESSSpeciesConfig /
        MESSBarrierConfig objects (e.g., integration tests and main workflow).

        Args:
            global_settings: MESSGlobalSettings instance.
            reaction_network: MESSReactionNetwork describing species order and types.
            molecule_objects: Dict mapping species name → molecule data dict.
            output_file: Destination file path.
        """
        self.set_global_settings(global_settings)

        # Render global and model sections
        parts: List[str] = [
            self.render_global_section(),
            self.render_model_section(),
        ]

        # Render each species/barrier in network order
        for sp in reaction_network.species:
            mol = molecule_objects.get(sp.name)
            if mol is None:
                logger.warning(f"No molecule data for species '{sp.name}'; skipping.")
                continue

            # Add name if not already in mol dict
            mol_data = dict(mol)
            mol_data.setdefault("name", sp.name)

            if sp.species_type == "barrier":
                try:
                    parts.append(self.render_barrier_template(mol_data))
                except Exception as exc:
                    logger.error(f"Failed to render barrier '{sp.name}': {exc}")
                    raise
            else:
                try:
                    parts.append(self.render_species_template(mol_data))
                except Exception as exc:
                    logger.error(f"Failed to render species '{sp.name}': {exc}")
                    raise

        parts.extend(["End", "End"])
        content = "\n".join(parts)

        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(content)

        logger.info(f"Written MESS input via assemble_mess_input to {out}")


@dataclass
class MESSSpecies:
    """
    Lightweight species descriptor used in reaction network definitions.

    This dataclass mirrors the fields tested in test_assembler.py / test_integration.py.
    It is intentionally separate from MESSSpeciesConfig (which holds parsed quantum data)
    so that tests and user-facing code can describe a reaction network with only names
    and file paths, deferring quantum-data loading to a later stage.
    """
    name: str
    species_type: str = "well"          # "well" | "barrier" | "bimolecular"
    gaussian_file: Optional[str] = None
    symmetry_factor: float = 1.0
    ground_energy: float = 0.0
    zero_energy: Optional[float] = None
    from_species: Optional[str] = None
    to_species: Optional[str] = None
    comment: str = ""


@dataclass
class MESSReactionNetwork:
    """
    Container for a list of MESSSpecies objects that describe a reaction network.

    Provides helper methods to filter species by type, matching the test API.
    """
    species: List[MESSSpecies] = field(default_factory=list)

    def get_wells(self) -> List[MESSSpecies]:
        """Return all well (non-barrier) species."""
        return [s for s in self.species if s.species_type != "barrier"]

    def get_barriers(self) -> List[MESSSpecies]:
        """Return all barrier (transition-state) species."""
        return [s for s in self.species if s.species_type == "barrier"]

    def get_species_by_name(self, name: str) -> Optional[MESSSpecies]:
        """Return the species with *name*, or None if not found."""
        for sp in self.species:
            if sp.name == name:
                return sp
        return None


def create_mess_config_from_yaml(yaml_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Create MESS configuration from YAML file.
    
    Args:
        yaml_file: Path to YAML configuration file
        
    Returns:
        Dictionary with MESS configuration
    """
    import yaml
    
    yaml_file = Path(yaml_file)
    
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load YAML file {yaml_file}: {e}")
        raise
    
    return config


if __name__ == "__main__":
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Assemble MESS input file from templates"
    )
    parser.add_argument("--config", "-c", required=True,
                       help="Configuration YAML file")
    parser.add_argument("--output", "-o", required=True,
                       help="Output MESS input file")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite output file if it exists")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    try:
        # Create assembler
        assembler = MESSAssembler()
        
        # Load configuration (simplified example)
        config = create_mess_config_from_yaml(args.config)
        
        # Set global settings (example)
        if "mess_global" in config:
            global_cfg = config["mess_global"]
            assembler.set_global_settings(MESSGlobalSettings(
                temperature_list=global_cfg.get("temperature_list", 
                                              [200, 300, 400, 500, 600, 700, 800, 900, 1000]),
                pressure_list=global_cfg.get("pressure_list", [1.0]),
                energy_step_over_temperature=global_cfg.get("energy_step_over_temperature", 0.2),
                excess_energy_over_temperature=global_cfg.get("excess_energy_over_temperature", 50),
                model_energy_limit=global_cfg.get("model_energy_limit", 400),
                calculation_method=global_cfg.get("calculation_method", "direct"),
                well_cutoff=global_cfg.get("well_cutoff", 20),
                chemical_eigenvalue_max=global_cfg.get("chemical_eigenvalue_max", 0.2),
                reduction_method=global_cfg.get("reduction_method", "diagonalization"),
                rate_output=global_cfg.get("rate_output", "mess.out"),
            ))
        
        # Set model settings (example)
        if "mess_model" in config:
            model_cfg = config["mess_model"]
            assembler.set_model_settings(MESSModelSettings(
                energy_relaxation=model_cfg.get("energy_relaxation", {
                    "model": "Exponential",
                    "factor": 350.0,
                    "power": 0.85,
                    "exponent_cutoff": 10
                }),
                collision_frequency=model_cfg.get("collision_frequency", {
                    "model": "LennardJones",
                    "epsilons": [10.0, 595.38],
                    "sigmas": [2.55, 4.89],
                    "masses": [4.0, 78.05]
                })
            ))
        
        # Write to file
        assembler.write_to_file(args.output, overwrite=args.overwrite)
        print(f"Successfully generated MESS input: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)