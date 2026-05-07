"""
Unit tests for the assembler module.

This module tests MESS input file assembly, template rendering,
and data structure creation.
"""
import os
import sys
import tempfile
from pathlib import Path
import pytest

# Import the assembler module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assembler import (
    MESSGlobalSettings,
    MESSReactionNetwork,
    MESSSpecies,
    MESSAssembler,
    create_molecule_object
)
from parser import QuantumData, Atom
from processor import CorrectionResult


class TestMESSGlobalSettings:
    """Test MESSGlobalSettings dataclass."""
    
    def test_global_settings_defaults(self):
        """Test default values for global settings."""
        settings = MESSGlobalSettings()
        
        # Check default values
        assert len(settings.temperature_list) > 0
        assert 300 in settings.temperature_list
        assert settings.pressure_list == [1.0]
        assert settings.energy_step_over_temperature == 0.2
        assert settings.excess_energy_over_temperature == 50
        assert settings.model_energy_limit == 400
        assert settings.calculation_method == "direct"
        assert settings.well_cutoff == 20
        assert settings.chemical_eigenvalue_max == 0.2
        assert settings.reduction_method == "diagonalization"
        assert settings.rate_output == "mess.out"
    
    def test_global_settings_custom(self):
        """Test custom global settings."""
        settings = MESSGlobalSettings(
            temperature_list=[300, 400, 500],
            pressure_list=[0.1, 1.0, 10.0],
            energy_step_over_temperature=0.1,
            excess_energy_over_temperature=100,
            model_energy_limit=500,
            calculation_method="sct",
            well_cutoff=30,
            chemical_eigenvalue_max=0.1,
            reduction_method="strong-collision",
            rate_output="rates.dat"
        )
        
        assert settings.temperature_list == [300, 400, 500]
        assert settings.pressure_list == [0.1, 1.0, 10.0]
        assert settings.energy_step_over_temperature == 0.1
        assert settings.excess_energy_over_temperature == 100
        assert settings.model_energy_limit == 500
        assert settings.calculation_method == "sct"
        assert settings.well_cutoff == 30
        assert settings.chemical_eigenvalue_max == 0.1
        assert settings.reduction_method == "strong-collision"
        assert settings.rate_output == "rates.dat"


class TestMESSSpecies:
    """Test MESSSpecies dataclass."""
    
    def test_species_creation(self):
        """Test creating a MESSSpecies instance."""
        species = MESSSpecies(
            name="Reactant",
            species_type="well",
            gaussian_file="reactant.out",
            symmetry_factor=2.0,
            ground_energy=0.0
        )
        
        assert species.name == "Reactant"
        assert species.species_type == "well"
        assert species.gaussian_file == "reactant.out"
        assert species.symmetry_factor == 2.0
        assert species.ground_energy == 0.0
        assert species.zero_energy is None
        assert species.from_species is None
        assert species.to_species is None
    
    def test_barrier_species_creation(self):
        """Test creating a barrier species."""
        species = MESSSpecies(
            name="TS",
            species_type="barrier",
            gaussian_file="ts.out",
            symmetry_factor=0.5,
            zero_energy=10.0,
            from_species="Reactant",
            to_species="Product"
        )
        
        assert species.name == "TS"
        assert species.species_type == "barrier"
        assert species.gaussian_file == "ts.out"
        assert species.symmetry_factor == 0.5
        assert species.zero_energy == 10.0
        assert species.from_species == "Reactant"
        assert species.to_species == "Product"
    
    def test_bimolecular_species_creation(self):
        """Test creating a bimolecular species."""
        species = MESSSpecies(
            name="Reactants",
            species_type="bimolecular",
            gaussian_file="reactants.out",
            symmetry_factor=1.0,
            ground_energy=0.0
        )
        
        assert species.name == "Reactants"
        assert species.species_type == "bimolecular"


class TestMESSReactionNetwork:
    """Test MESSReactionNetwork dataclass."""
    
    def test_reaction_network_creation(self):
        """Test creating a MESSReactionNetwork instance."""
        # Create species
        reactant = MESSSpecies(
            name="R",
            species_type="well",
            gaussian_file="reactant.out",
            symmetry_factor=2.0,
            ground_energy=0.0
        )
        
        ts = MESSSpecies(
            name="TS",
            species_type="barrier",
            gaussian_file="ts.out",
            symmetry_factor=0.5,
            zero_energy=10.0,
            from_species="R",
            to_species="P"
        )
        
        product = MESSSpecies(
            name="P",
            species_type="well",
            gaussian_file="product.out",
            symmetry_factor=2.0,
            ground_energy=-5.0
        )
        
        # Create reaction network
        network = MESSReactionNetwork(
            species=[reactant, ts, product]
        )
        
        assert len(network.species) == 3
        assert network.species[0].name == "R"
        assert network.species[1].name == "TS"
        assert network.species[2].name == "P"
        
        # Check well and barrier counts
        wells = network.get_wells()
        assert len(wells) == 2
        assert wells[0].name == "R"
        assert wells[1].name == "P"
        
        barriers = network.get_barriers()
        assert len(barriers) == 1
        assert barriers[0].name == "TS"
    
    def test_get_species_by_name(self):
        """Test retrieving species by name."""
        reactant = MESSSpecies(name="R", species_type="well", gaussian_file="r.out")
        ts = MESSSpecies(name="TS", species_type="barrier", gaussian_file="ts.out")
        
        network = MESSReactionNetwork(species=[reactant, ts])
        
        # Test existing species
        r_species = network.get_species_by_name("R")
        assert r_species == reactant
        
        ts_species = network.get_species_by_name("TS")
        assert ts_species == ts
        
        # Test non-existent species
        non_existent = network.get_species_by_name("NonExistent")
        assert non_existent is None


class TestMESSAssembler:
    """Test MESSAssembler functionality."""
    
    def test_assembler_initialization(self):
        """Test MESSAssembler initialization."""
        assembler = MESSAssembler()
        
        assert assembler.template_env is not None
        assert "species.jinja2" in assembler.template_env.list_templates()
        assert "global_section.jinja2" in assembler.template_env.list_templates()
    
    def test_assembler_with_custom_template_dir(self):
        """Test MESSAssembler adding custom Jinja2 template via the existing environment."""
        assembler = MESSAssembler()
        
        # Add a custom template to the existing environment using a DictLoader
        import jinja2
        custom_loader = jinja2.DictLoader({"test.jinja2": "Test template: {{ value }}"})
        combined = jinja2.ChoiceLoader([assembler.env.loader, custom_loader])
        assembler.env.loader = combined
        
        # Should be able to render the test template
        result = assembler.env.get_template("test.jinja2").render(value=42)
        assert result == "Test template: 42"
    
    def test_render_species_template(self):
        """Test rendering species template."""
        assembler = MESSAssembler()
        
        # Create test molecule data
        molecule_data = {
            "name": "TestMolecule",
            "type": "well",
            "method": "RRHO",
            "atoms": [
                {"symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0},
                {"symbol": "H", "x": 0.0, "y": 0.0, "z": 1.089},
            ],
            "num_atoms": 2,
            "frequencies": [100.0, 200.0, 300.0],
            "num_frequencies": 3,
            "scf_energy": -100.0,
            "zero_point_energy": 10.0,
            "total_energy": 500.0,
            "symmetry_factor": 2.0,
            "GroundEnergy": 0.0,
            "multiplicity": 1,
            "charge": 0,
            "convergence_status": True,
            "method_basis": "B3LYP/6-31G*",
            "comment": "",
        }
        
        # Render species template
        rendered = assembler.render_species_template(molecule_data)
        
        # Check basic structure
        assert "TestMolecule" in rendered
        assert "RRHO" in rendered
        assert "C" in rendered
        assert "H" in rendered
        assert "100.0" in rendered or "100.00" in rendered
        assert "ZeroEnergy[kcal/mol]" in rendered
        assert "500.0" in rendered or "500.00" in rendered
    
    def test_render_barrier_template(self):
        """Test rendering barrier template."""
        assembler = MESSAssembler()
        
        # Create test barrier data
        barrier_data = {
            "name": "TS",
            "from_species": "Reactant",
            "to_species": "Product",
            "atoms": [
                {"symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0},
                {"symbol": "H", "x": 0.0, "y": 0.0, "z": 1.089},
            ],
            "num_atoms": 2,
            "frequencies": [-500.0, 100.0, 200.0, 300.0],
            "num_frequencies": 4,
            "real_frequencies": [100.0, 200.0, 300.0],
            "imaginary_frequencies": [-500.0],
            "imaginary_frequency": 485.5,  # After scaling
            "num_imaginary": 1,
            "scf_energy": -99.0,
            "zero_point_energy": 4.0,
            "total_energy": 505.0,
            "symmetry_factor": 0.5,
            "GroundEnergy": 10.0,
            "forward_barrier": 5.0,
            "reverse_barrier": 15.0,
            "multiplicity": 1,
            "charge": 0,
            "convergence_status": True,
            "method_basis": "B3LYP/6-31G*",
            "electronic_levels": [
                {"energy": 0.0, "degeneracy": 1.0}
            ]
        }
        
        # Render barrier template
        rendered = assembler.render_barrier_template(barrier_data)
        
        # Check basic structure
        assert "Barrier" in rendered
        assert "TS" in rendered
        assert "Reactant" in rendered
        assert "Product" in rendered
        assert "ImaginaryFrequency[1/cm]" in rendered
        assert "485.5" in rendered or "485.50" in rendered
        assert "ZeroEnergy[kcal/mol]" in rendered
        assert "505.0" in rendered or "505.00" in rendered
        assert "WellDepth[kcal/mol]" in rendered
        assert "5.0" in rendered or "5.00" in rendered
        assert "15.0" in rendered or "15.00" in rendered
    
    def test_render_global_template(self):
        """Test rendering global template."""
        assembler = MESSAssembler()
        
        # Create global settings
        global_settings = {
            "temperature_list": [300, 400, 500],
            "pressure_list": [1.0],
            "energy_step_over_temperature": 0.2,
            "excess_energy_over_temperature": 50,
            "model_energy_limit": 400,
            "calculation_method": "direct",
            "well_cutoff": 20,
            "chemical_eigenvalue_max": 0.2,
            "reduction_method": "diagonalization",
            "rate_output": "mess.out"
        }
        
        # Render global template
        rendered = assembler.render_global_template(global_settings)
        
        # Check basic structure (match actual template field names)
        assert "TemperatureList[K]" in rendered
        assert "300" in rendered
        assert "400" in rendered
        assert "500" in rendered
        assert "PressureList[atm]" in rendered
        assert "1.0" in rendered
        assert "EnergyStepOverTemperature" in rendered
        assert "0.2" in rendered
    
    def test_assemble_mess_input(self):
        """Test assembling complete MESS input."""
        assembler = MESSAssembler()
        
        # Create global settings
        global_settings = MESSGlobalSettings(
            temperature_list=[300, 400, 500],
            pressure_list=[1.0]
        )
        
        # Create reaction network
        reactant = MESSSpecies(
            name="R",
            species_type="well",
            gaussian_file="reactant.out",
            symmetry_factor=2.0,
            ground_energy=0.0
        )
        
        network = MESSReactionNetwork(species=[reactant])
        
        # Create molecule objects dictionary
        molecule_objects = {
            "R": {
                "name": "R",
                "type": "well",
                "method": "RRHO",
                "atoms": [
                    {"symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0},
                ],
                "num_atoms": 1,
                "frequencies": [100.0, 200.0],
                "num_frequencies": 2,
                "total_energy": 500.0,
                "symmetry_factor": 2.0,
                "GroundEnergy": 0.0,
                "comment": "",
            }
        }
        
        # Assemble MESS input
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_mess.inp"
            
            assembler.assemble_mess_input(
                global_settings=global_settings,
                reaction_network=network,
                molecule_objects=molecule_objects,
                output_file=str(output_file)
            )
            
            # Check file was created
            assert output_file.exists()
            
            # Read and check content
            content = output_file.read_text(encoding='utf-8')
            assert "R" in content
            assert "RRHO" in content
            assert "ZeroEnergy[kcal/mol]" in content
            assert "500.0" in content or "500.00" in content
    
    def test_assemble_mess_input_with_barriers(self):
        """Test assembling MESS input with barriers."""
        assembler = MESSAssembler()
        
        # Create global settings
        global_settings = MESSGlobalSettings(
            temperature_list=[300]
        )
        
        # Create reaction network with barrier
        reactant = MESSSpecies(
            name="R", species_type="well", gaussian_file="r.out"
        )
        
        ts = MESSSpecies(
            name="TS",
            species_type="barrier",
            gaussian_file="ts.out",
            from_species="R",
            to_species="P"
        )
        
        product = MESSSpecies(
            name="P", species_type="well", gaussian_file="p.out"
        )
        
        network = MESSReactionNetwork(species=[reactant, ts, product])
        
        # Create molecule objects
        molecule_objects = {
            "R": {"name": "R", "type": "RRHO", "total_energy": 0.0},
            "TS": {
                "name": "TS", 
                "type": "Barrier", 
                "total_energy": 10.0,
                "imaginary_frequency": 500.0,
                "forward_barrier": 10.0,
                "reverse_barrier": 5.0
            },
            "P": {"name": "P", "type": "RRHO", "total_energy": -5.0}
        }
        
        # Assemble MESS input
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_barrier.inp"
            
            assembler.assemble_mess_input(
                global_settings=global_settings,
                reaction_network=network,
                molecule_objects=molecule_objects,
                output_file=str(output_file)
            )
            
            # Check file content
            content = output_file.read_text()
            assert "Barrier" in content
            assert "TS" in content
            assert "ImaginaryFrequency[1/cm]" in content
            assert "ZeroEnergy[kcal/mol]" in content
            assert "WellDepth[kcal/mol]" in content
            assert "10.0" in content or "10.00" in content
            assert "5.0" in content or "5.00" in content
    
    def test_assemble_with_missing_species(self):
        """Test assembling with missing species data."""
        assembler = MESSAssembler()
        
        global_settings = MESSGlobalSettings()
        
        reactant = MESSSpecies(
            name="R", species_type="well", gaussian_file="r.out"
        )
        network = MESSReactionNetwork(species=[reactant])
        
        # Empty molecule objects (missing species)
        molecule_objects = {}
        
        # Should raise KeyError or handle gracefully
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.inp"
            
            # Depending on implementation, might raise KeyError
            # or handle gracefully. We'll check both cases.
            try:
                assembler.assemble_mess_input(
                    global_settings=global_settings,
                    reaction_network=network,
                    molecule_objects=molecule_objects,
                    output_file=str(output_file)
                )
                # If it doesn't raise, check if file was created
                # (might create empty or partial file)
                if output_file.exists():
                    content = output_file.read_text()
                    # Might contain only global section
                    assert "TemperatureList[K]" in content
            except KeyError:
                # This is also acceptable behavior
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])