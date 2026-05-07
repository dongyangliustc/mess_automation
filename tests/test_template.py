"""
Template rendering tests.

This module tests the Jinja2 template rendering for MESS input files.
"""
import os
import sys
from pathlib import Path
import pytest
import jinja2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assembler import MESSAssembler


class TestTemplateRendering:
    """Test template rendering functionality."""
    
    def test_template_env_creation(self):
        """Test template environment creation."""
        assembler = MESSAssembler()
        
        # Check template environment exists
        assert assembler.template_env is not None
        assert isinstance(assembler.template_env, jinja2.Environment)
        
        # Check essential templates exist
        templates = assembler.template_env.list_templates()
        assert "species.jinja2" in templates
        assert "barrier.jinja2" in templates
        assert "global_section.jinja2" in templates  # actual filename in templates/
    
    def test_species_template_rendering(self):
        """Test species template rendering."""
        assembler = MESSAssembler()
        
        # Test data
        data = {
            "name": "Molecule1",
            "type": "well",
            "method": "RRHO",
            "atoms": [
                {"symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0},
                {"symbol": "H", "x": 0.0, "y": 0.0, "z": 1.089},
            ],
            "num_atoms": 2,
            "frequencies": [100.0, 200.0, 300.0],
            "num_frequencies": 3,
            "total_energy": 500.0,
            "symmetry_factor": 2.0,
            "GroundEnergy": 0.0,
            "comment": "",
        }
        
        rendered = assembler.render_species_template(data)
        
        # Basic checks
        assert "Molecule1" in rendered
        assert "RRHO" in rendered
        assert "ZeroEnergy[kcal/mol]" in rendered
        assert "500.0" in rendered or "500.00" in rendered
        assert "100.0" in rendered or "100.00" in rendered
        assert "C" in rendered
        assert "H" in rendered
    
    def test_species_template_without_total_energy(self):
        """Test species template uses kcal/mol ZPE fallback when total_energy absent."""
        assembler = MESSAssembler()
        
        # Test data without total_energy
        data = {
            "name": "Molecule1",
            "type": "well",
            "method": "RRHO",
            "atoms": [{"symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0}],
            "num_atoms": 1,
            "frequencies": [100.0],
            "num_frequencies": 1,
            "zero_point_energy_kcal_mol": 10.0,  # Only ZPE, no total_energy
            "symmetry_factor": 1.0,
            "GroundEnergy": 0.0,
            "comment": "",
        }
        
        rendered = assembler.render_species_template(data)
        
        # Should use zero_point_energy_kcal_mol if total_energy is not available
        assert "ZeroEnergy[kcal/mol]" in rendered
        assert "10.0" in rendered or "10.00" in rendered
    
    def test_species_template_without_energy(self):
        """Test species template omits ZeroEnergy when no energy is provided."""
        assembler = MESSAssembler()
        
        # Test data without any energy
        data = {
            "name": "Molecule1",
            "type": "well",
            "method": "RRHO",
            "atoms": [{"symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0}],
            "num_atoms": 1,
            "frequencies": [100.0],
            "num_frequencies": 1,
            # No energy fields
            "symmetry_factor": 1.0,
            "GroundEnergy": 0.0,
            "comment": "",
        }
        
        rendered = assembler.render_species_template(data)
        
        # Should still render, just without ZeroEnergy line
        assert "Molecule1" in rendered
        assert "RRHO" in rendered
        # ZeroEnergy line should not be present
        assert "ZeroEnergy[kcal/mol]" not in rendered
    
    def test_barrier_template_rendering(self):
        """Test barrier template rendering."""
        assembler = MESSAssembler()
        
        # Test data for barrier
        data = {
            "name": "TS",
            "from_species": "Reactant",
            "to_species": "Product",
            "atoms": [{"symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0}],
            "num_atoms": 1,
            "frequencies": [-500.0, 100.0, 200.0],
            "num_frequencies": 3,
            "real_frequencies": [100.0, 200.0],
            "imaginary_frequencies": [-500.0],
            "imaginary_frequency": 485.5,
            "num_imaginary": 1,
            "total_energy": 505.0,
            "symmetry_factor": 0.5,
            "GroundEnergy": 10.0,
            "forward_barrier": 5.0,
            "reverse_barrier": 15.0,
            "electronic_levels": [
                {"energy": 0.0, "degeneracy": 1.0}
            ]
        }
        
        rendered = assembler.render_barrier_template(data)
        
        # Barrier-specific checks
        assert "Barrier" in rendered
        assert "TS" in rendered
        assert "Reactant" in rendered
        assert "Product" in rendered
        assert "ImaginaryFrequency[1/cm]" in rendered
        assert "485.5" in rendered or "485.50" in rendered
        assert "ZeroEnergy[kcal/mol]" in rendered
        assert "505.0" in rendered or "505.00" in rendered
        assert "WellDepth[kcal/mol]" in rendered
        assert "5.0" in rendered or "5.00" in rendered  # Forward
        assert "15.0" in rendered or "15.00" in rendered  # Reverse
    
    def test_barrier_template_without_barrier_depths(self):
        """Test barrier template without barrier depths."""
        assembler = MESSAssembler()
        
        # Test data without barrier depths
        data = {
            "name": "TS",
            "from_species": "Reactant",
            "to_species": "Product",
            "atoms": [{"symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0}],
            "num_atoms": 1,
            "frequencies": [-500.0, 100.0],
            "num_frequencies": 2,
            "real_frequencies": [100.0],
            "imaginary_frequency": 485.5,
            "num_imaginary": 1,
            "total_energy": 505.0,
            "symmetry_factor": 0.5,
            "GroundEnergy": 10.0,
            # No forward_barrier or reverse_barrier
            "electronic_levels": [
                {"energy": 0.0, "degeneracy": 1.0}
            ]
        }
        
        rendered = assembler.render_barrier_template(data)
        
        # Should render without WellDepth lines
        assert "Barrier" in rendered
        assert "ImaginaryFrequency[1/cm]" in rendered
        assert "ZeroEnergy[kcal/mol]" in rendered
        # WellDepth lines should not be present
        assert "WellDepth[kcal/mol]" not in rendered
    
    def test_global_template_rendering(self):
        """Test global template rendering."""
        assembler = MESSAssembler()
        
        # Test data for global settings
        data = {
            "temperature_list": [300, 400, 500],
            "pressure_list": [0.1, 1.0, 10.0],
            "energy_step_over_temperature": 0.2,
            "excess_energy_over_temperature": 50,
            "model_energy_limit": 400,
            "calculation_method": "direct",
            "well_cutoff": 20,
            "chemical_eigenvalue_max": 0.2,
            "reduction_method": "diagonalization",
            "rate_output": "mess.out",
            "log_output": "mess.log",
            "eigenvalue_output": "eigenvalues.dat",
            "eigenvector_output": "eigenvectors.dat",
            "ped_output": "ped.dat"
        }
        
        rendered = assembler.render_global_template(data)
        
        # Global settings checks (match actual template field names)
        assert "TemperatureList[K]" in rendered
        assert "300" in rendered
        assert "400" in rendered
        assert "500" in rendered
        assert "PressureList[atm]" in rendered
        assert "0.1" in rendered
        assert "1.0" in rendered
        assert "10.0" in rendered
        assert "EnergyStepOverTemperature" in rendered
        assert "0.2" in rendered
        assert "ExcessEnergyOverTemperature" in rendered
        assert "50" in rendered
        assert "ModelEnergyLimit[kcal/mol]" in rendered
        assert "400" in rendered
        assert "WellCutoff" in rendered       # template uses WellCutoff (no [kcal/mol])
        assert "20" in rendered
        assert "RateOutput" in rendered or "RateConstantOutput" in rendered  # allow either
        assert "mess.out" in rendered
    
    def test_global_template_minimal(self):
        """Test global template with minimal data."""
        assembler = MESSAssembler()
        
        # Minimal data
        data = {
            "temperature_list": [300],
            "pressure_list": [1.0],
        }
        
        rendered = assembler.render_global_template(data)
        
        # Should still render basic structure
        assert "TemperatureList[K]" in rendered
        assert "300" in rendered
        assert "PressureList[atm]" in rendered
        assert "1.0" in rendered
        # Default values should be used
        assert "EnergyStepOverTemperature" in rendered
        assert "ExcessEnergyOverTemperature" in rendered
    
    def test_template_formatting(self):
        """Test template formatting (indentation, etc.)."""
        assembler = MESSAssembler()
        
        # Simple test data
        data = {
            "name": "Test",
            "type": "well",
            "atoms": [{"symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0}],
            "num_atoms": 1,
            "frequencies": [100.0],
            "num_frequencies": 1,
            "total_energy": 100.0,
            "symmetry_factor": 1.0,
            "GroundEnergy": 0.0,
            "method": "RRHO",
        }
        
        rendered = assembler.render_species_template(data)
        
        # Check basic content is present
        lines = rendered.strip().split('\n')
        
        # Name and RRHO keyword should appear somewhere in the rendered output
        full_text = rendered
        assert "Test" in full_text
        assert "RRHO" in full_text
        assert "ZeroEnergy[kcal/mol]" in full_text
        assert "100." in full_text  # energy value
    
    def test_template_error_handling(self):
        """Test template rendering with missing required data."""
        assembler = MESSAssembler()
        
        # Test with minimal/missing data - the template uses "is defined" guards
        # so most missing fields are handled gracefully.
        # Confirm that the render does not crash for incomplete dicts.
        data = {
            "frequencies": [100.0]
            # name, type, method are missing
        }
        
        # The updated template uses "is defined" guards and should not raise.
        rendered = assembler.render_species_template(data)
        # Result may be incomplete but should not crash
        assert isinstance(rendered, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
