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
    """Configuration for a MESS species (well/reactant/product)."""
    name: str
    type: str  # "reactant", "well", "product", "bimolecular"
    quantum_data: QuantumData
    correction: Optional[CorrectionResult] = None
    stoichiometry: Optional[str] = None
    symmetry_factor: float = 1.0
    ZeroEnergy: Optional[float] = None          # Computed ZeroEnergy [kcal/mol] – THE authoritative energy
    comment: str = ""
    method: str = "RRHO"
    core_type: str = "RigidRotor"
    electronic_levels: List[Dict[str, float]] = field(default_factory=lambda: [
        {"energy": 0.0, "degeneracy": 1.0}
    ])
    # For bimolecular species
    fragments: List[Dict[str, Any]] = field(default_factory=list)

    # Keep GroundEnergy for backwards compatibility (not used internally)
    GroundEnergy: Optional[float] = None


@dataclass
class MESSBarrierConfig:
    """Configuration for a MESS barrier (transition state)."""
    name: str
    from_species: str
    to_species: str
    quantum_data: QuantumData
    correction: Optional[CorrectionResult] = None
    stoichiometry: Optional[str] = None
    symmetry_factor: float = 1.0
    ZeroEnergy: Optional[float] = None          # Computed ZeroEnergy [kcal/mol] – THE authoritative energy
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
    # Barrier depths (kcal/mol) – will be calculated automatically if species available
    forward_barrier: Optional[float] = None
    reverse_barrier: Optional[float] = None


class MESSAssembler:
    """Main class for assembling MESS input files from templates."""

    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
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
        formatted = f"{value:.{precision}f}"
        if '.' in formatted:
            formatted = formatted.rstrip('0')
            if formatted.endswith('.'):
                formatted = formatted + '0'
        return formatted

    def set_global_settings(self, settings: MESSGlobalSettings):
        self.global_settings = settings

    def set_model_settings(self, settings: MESSModelSettings):
        self.model_settings = settings

    def add_species(self, species: MESSSpeciesConfig):
        self.species[species.name] = species
        logger.debug(f"Added species: {species.name} ({species.type})")

    def add_barrier(self, barrier: MESSBarrierConfig):
        # Calculate barrier depths if not already set and we have the required species
        if barrier.forward_barrier is None or barrier.reverse_barrier is None:
            self._calculate_barrier_depths(barrier)

        self.barriers[barrier.name] = barrier
        logger.debug(f"Added barrier: {barrier.name} ({barrier.from_species} -> {barrier.to_species})")

    def _calculate_barrier_depths(self, barrier: MESSBarrierConfig):
        """Calculate forward and reverse barrier depths using ZeroEnergy."""
        ts_energy = barrier.ZeroEnergy
        reactant = self.species.get(barrier.from_species)
        product = self.species.get(barrier.to_species)

        if ts_energy is not None and reactant is not None and reactant.ZeroEnergy is not None:
            barrier.forward_barrier = ts_energy - reactant.ZeroEnergy
            logger.info(f"Forward barrier for {barrier.name}: {barrier.forward_barrier:.2f} kcal/mol")
        elif barrier.quantum_data.scf_energy is not None and reactant is not None and reactant.quantum_data.scf_energy is not None:
            barrier.forward_barrier = (barrier.quantum_data.scf_energy - reactant.quantum_data.scf_energy) * 627.509474

        if ts_energy is not None and product is not None and product.ZeroEnergy is not None:
            barrier.reverse_barrier = ts_energy - product.ZeroEnergy
            logger.info(f"Reverse barrier for {barrier.name}: {barrier.reverse_barrier:.2f} kcal/mol")
        elif barrier.quantum_data.scf_energy is not None and product is not None and product.quantum_data.scf_energy is not None:
            barrier.reverse_barrier = (barrier.quantum_data.scf_energy - product.quantum_data.scf_energy) * 627.509474

    def _prepare_species_data(self, species: MESSSpeciesConfig) -> Dict[str, Any]:
        # Use ZeroEnergy as the authoritative energy
        mol = create_molecule_object(
            species.quantum_data, species.correction,
            name=species.name, species_type=species.type,
            symmetry_factor=species.symmetry_factor,
            ZeroEnergy=species.ZeroEnergy,
        )

        mol.update({
            "type": species.type,
            "comment": species.comment,
            "method": species.method,
            "core_type": species.core_type,
            "core_defined": True,
            "electronic_levels": species.electronic_levels,
            "stoichiometry": species.stoichiometry,
            "ZeroEnergy": species.ZeroEnergy,
        })

        # Add geometry string
        if species.correction and species.correction.success:
            mol["geometry_string"] = species.quantum_data.get_geometry_string()
        else:
            mol["geometry_string"] = species.quantum_data.get_geometry_string()

        # Add frequencies string
        if species.correction and species.correction.success:
            mol["frequencies_string"] = species.correction.get_frequencies_string()
        else:
            mol["frequencies_string"] = species.quantum_data.get_frequencies_string()

        # For bimolecular species
        if species.type == "bimolecular" and species.fragments:
            mol["fragments"] = []
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
                mol["fragments"].append(frag)

        return mol

    def _prepare_barrier_data(self, barrier: MESSBarrierConfig) -> Dict[str, Any]:
        mol = create_molecule_object(
            barrier.quantum_data, barrier.correction,
            name=barrier.name, species_type="barrier",
            symmetry_factor=barrier.symmetry_factor,
            ZeroEnergy=barrier.ZeroEnergy,
        )

        mol.update({
            "from_species": barrier.from_species,
            "to_species": barrier.to_species,
            "comment": barrier.comment,
            "method": barrier.method,
            "core_type": barrier.core_type,
            "core_defined": True,
            "electronic_levels": barrier.electronic_levels,
            "stoichiometry": barrier.stoichiometry,
            "ZeroEnergy": barrier.ZeroEnergy,
            "is_bimolecular_barrier": barrier.is_bimolecular,
            "potential_prefactor": barrier.potential_prefactor,
            "potential_power_exponent": barrier.potential_power_exponent,
            "forward_barrier": barrier.forward_barrier,
            "reverse_barrier": barrier.reverse_barrier,
        })

        mol["geometry_string"] = barrier.quantum_data.get_geometry_string()

        if barrier.correction and barrier.correction.success:
            mol["frequencies_string"] = barrier.correction.get_frequencies_string()
            if barrier.correction.imaginary_frequency is not None:
                mol["imaginary_frequency"] = barrier.correction.imaginary_frequency
                if barrier.correction.real_frequencies:
                    mol["real_frequencies"] = barrier.correction.real_frequencies
                    mol["num_real_frequencies"] = len(barrier.correction.real_frequencies)
        else:
            mol["frequencies_string"] = barrier.quantum_data.get_frequencies_string()

        if barrier.is_bimolecular and barrier.fragments:
            mol["fragments"] = []
            for frag_data in barrier.fragments:
                frag = {
                    "name": frag_data.get("name", "FRAG"),
                    "num_atoms": frag_data.get("num_atoms", 0),
                    "atoms": frag_data.get("atoms", []),
                    "geometry_string": frag_data.get("geometry_string", ""),
                }
                mol["fragments"].append(frag)

        return mol

    def render_global_section(self) -> str:
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
        if not self.model_template:
            raise ValueError("Model template not loaded")
        context = {
            "energy_relaxation": self.model_settings.energy_relaxation,
            "collision_frequency": self.model_settings.collision_frequency,
        }
        return self.model_template.render(**context)

    def render_species_sections(self) -> str:
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
        logger.info("Assembling MESS input file")
        parts = []
        parts.append(self.render_global_section())
        parts.append(self.render_model_section())
        parts.append("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        parts.append("!***************************************************")
        parts.append("!  REACTANTS")
        parts.append("!***************************************************")
        parts.append("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        parts.append("!***************************************************")
        parts.append("")
        parts.append(self.render_species_sections())
        parts.append(self.render_barrier_sections())
        parts.append("End")
        parts.append("End")
        mess_input = "\n".join(parts)
        self._validate_output(mess_input)
        logger.info(f"Successfully assembled MESS input ({len(mess_input.splitlines())} lines)")
        return mess_input

    def _validate_output(self, mess_input: str):
        lines = mess_input.splitlines()
        issues = []
        if not any("TemperatureList" in line for line in lines):
            issues.append("Missing TemperatureList")
        if not any("PressureList" in line for line in lines):
            issues.append("Missing PressureList")
        if not any("Model" in line for line in lines):
            issues.append("Missing Model section")
        model_start = any(line.strip() == "Model" for line in lines)
        model_end_count = sum(1 for line in lines if line.strip() == "End")
        if model_start and model_end_count < 2:
            issues.append("Insufficient End markers")
        species_count = sum(1 for line in lines if "Species" in line)
        if species_count == 0:
            issues.append("No species defined")
        if issues:
            logger.warning(f"Validation issues found: {', '.join(issues)}")
        else:
            logger.debug("MESS input validation passed")

    def write_to_file(self, output_file: Union[str, Path], overwrite: bool = False):
        output_file = Path(output_file)
        if output_file.exists() and not overwrite:
            raise FileExistsError(
                f"Output file {output_file} already exists. "
                "Use overwrite=True to overwrite."
            )
        mess_input = self.assemble()
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(mess_input)
            logger.info(f"Written MESS input to {output_file}")
        except IOError as e:
            logger.error(f"Failed to write output file: {e}")
            raise

    # Convenience render methods (used by tests)
    def render_species_template(self, data: Dict[str, Any]) -> str:
        if not self.species_template:
            raise ValueError("Species template not loaded")
        return self.species_template.render(species=data)

    def render_barrier_template(self, data: Dict[str, Any]) -> str:
        if not self.barrier_template:
            raise ValueError("Barrier template not loaded")
        return self.barrier_template.render(barrier=data)

    def render_global_template(self, data: Dict[str, Any]) -> str:
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
        self.set_global_settings(global_settings)
        parts: List[str] = [
            self.render_global_section(),
            self.render_model_section(),
        ]
        for sp in reaction_network.species:
            mol = molecule_objects.get(sp.name)
            if mol is None:
                logger.warning(f"No molecule data for species '{sp.name}'; skipping.")
                continue
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
    """
    name: str
    species_type: str = "well"          # "well" | "barrier" | "bimolecular"
    gaussian_file: Optional[str] = None
    symmetry_factor: float = 1.0
    GroundEnergy: float = 0.0           # Deprecated, kept for compatibility
    ZeroEnergy: Optional[float] = None  # Preferred energy field [kcal/mol]
    from_species: Optional[str] = None
    to_species: Optional[str] = None
    comment: str = ""


@dataclass
class MESSReactionNetwork:
    species: List[MESSSpecies] = field(default_factory=list)

    def get_wells(self) -> List[MESSSpecies]:
        return [s for s in self.species if s.species_type != "barrier"]

    def get_barriers(self) -> List[MESSSpecies]:
        return [s for s in self.species if s.species_type == "barrier"]

    def get_species_by_name(self, name: str) -> Optional[MESSSpecies]:
        for sp in self.species:
            if sp.name == name:
                return sp
        return None


def create_mess_config_from_yaml(yaml_file: Union[str, Path]) -> Dict[str, Any]:
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
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    try:
        assembler = MESSAssembler()
        config = create_mess_config_from_yaml(args.config)
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
        assembler.write_to_file(args.output, overwrite=args.overwrite)
        print(f"Successfully generated MESS input: {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)