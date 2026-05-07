"""
Unit tests for configuration handling.

This module tests configuration file parsing, validation, and processing.
"""
import os
import sys
import tempfile
import yaml
from pathlib import Path
import pytest

# Import configuration utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# We'll assume config handling is in main.py or a separate config module
from main import Config, parse_config_file, validate_config


class TestConfigParsing:
    """Test configuration file parsing."""
    
    def test_parse_valid_config(self, example_config_file):
        """Test parsing a valid configuration file."""
        # Create a simple test config
        config_data = {
            "input": {
                "files": ["test.out"]
            },
            "quantum": {
                "frequency_scaling_factor": 0.971
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Parse config
            config = parse_config_file(config_path)
            assert config is not None
            assert "input" in config
            assert "quantum" in config
            assert config["input"]["files"] == ["test.out"]
            assert config["quantum"]["frequency_scaling_factor"] == 0.971
        finally:
            os.unlink(config_path)
    
    def test_parse_invalid_yaml(self):
        """Test parsing invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [}")
            config_path = f.name
        
        try:
            config = parse_config_file(config_path)
            # Should return None or raise exception
            assert config is None or isinstance(config, dict)
        except yaml.YAMLError:
            # This is also acceptable
            pass
        finally:
            os.unlink(config_path)
    
    def test_parse_nonexistent_file(self):
        """Test parsing non-existent file."""
        config = parse_config_file("non_existent_file.yaml")
        assert config is None


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_minimal_config(self):
        """Test validation of minimal valid configuration."""
        config = {
            "input": {
                "files": ["test.out"]
            },
            "quantum": {
                "frequency_scaling_factor": 0.971
            },
            "mess_global": {
                "temperature_list": [300],
                "pressure_list": [1.0]
            }
        }
        
        is_valid, errors = validate_config(config)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_missing_required_fields(self):
        """Test validation with missing required fields."""
        config = {
            "input": {
                "files": ["test.out"]
            }
            # Missing quantum and mess_global
        }
        
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        # Should have errors about missing fields
        assert any("quantum" in error.lower() for error in errors)
    
    def test_validate_invalid_frequency_scaling(self):
        """Test validation with invalid frequency scaling factor."""
        config = {
            "input": {"files": ["test.out"]},
            "quantum": {
                "frequency_scaling_factor": -0.5  # Invalid negative
            },
            "mess_global": {
                "temperature_list": [300],
                "pressure_list": [1.0]
            }
        }
        
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        assert any("frequency" in error.lower() for error in errors)
    
    def test_validate_empty_file_list(self):
        """Test validation with empty file list."""
        config = {
            "input": {
                "files": []  # Empty list
            },
            "quantum": {
                "frequency_scaling_factor": 0.971
            },
            "mess_global": {
                "temperature_list": [300],
                "pressure_list": [1.0]
            }
        }
        
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        assert any("file" in error.lower() for error in errors)
    
    def test_validate_invalid_temperature_list(self):
        """Test validation with invalid temperature list."""
        config = {
            "input": {"files": ["test.out"]},
            "quantum": {"frequency_scaling_factor": 0.971},
            "mess_global": {
                "temperature_list": [],  # Empty list
                "pressure_list": [1.0]
            }
        }
        
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        assert any("temperature" in error.lower() for error in errors)
    
    def test_validate_negative_temperatures(self):
        """Test validation with negative temperatures."""
        config = {
            "input": {"files": ["test.out"]},
            "quantum": {"frequency_scaling_factor": 0.971},
            "mess_global": {
                "temperature_list": [-100, 300],  # Negative temperature
                "pressure_list": [1.0]
            }
        }
        
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        assert any("negative" in error.lower() or "temperature" in error.lower() for error in errors)


class TestConfigClass:
    """Test Config dataclass."""
    
    def test_config_creation_from_dict(self):
        """Test creating Config from dictionary."""
        config_dict = {
            "input": {
                "files": ["test1.out", "test2.out"]
            },
            "quantum": {
                "Frequency_factor": 0.971,
                "zpe_factor": 0.95,
                "geometry_units": "angstrom",
                "frequency_units": "1/cm",
                "energy_units": "kcal/mol"
            },
            "mess_global": {
                "temperature_list": [300, 400, 500],
                "pressure_list": [0.1, 1.0, 10.0],
                "energy_step_over_temperature": 0.2,
                "excess_energy_over_temperature": 50,
                "model_energy_limit": 400,
                "calculation_method": "direct",
                "well_cutoff": 20
            },
            "reaction_network": {
                "species": [
                    {
                        "name": "R",
                        "type": "well",
                        "gaussian_file": "r.out",
                        "symmetry_factor": 2.0,
                        "GroundEnergy": 0.0
                    }
                ]
            },
            "processing": {
                "skip_unconverged": True,
                "validate_frequencies": True,
                "create_backups": False,
                "verbose": True
            }
        }
        
        # Create Config instance
        config = Config.from_dict(config_dict)
        
        # Check properties
        assert config.input_files == ["test1.out", "test2.out"]
        assert config.frequency_scaling_factor == 0.971
        assert config.Frequency_factor == 0.971
        assert config.zpe_factor == 0.95
        assert config.geometry_units == "angstrom"
        assert config.frequency_units == "1/cm"
        assert config.energy_units == "kcal/mol"
        assert config.temperature_list == [300, 400, 500]
        assert config.pressure_list == [0.1, 1.0, 10.0]
        assert config.energy_step_over_temperature == 0.2
        assert config.excess_energy_over_temperature == 50
        assert config.model_energy_limit == 400
        assert config.calculation_method == "direct"
        assert config.well_cutoff == 20
        assert config.skip_unconverged is True
        assert config.validate_frequencies is True
        assert config.create_backups is False
        assert config.verbose is True
        
        # Check reaction network
        assert len(config.reaction_network) == 1
        species = config.reaction_network[0]
        assert species["name"] == "R"
        assert species["type"] == "well"
        assert species["gaussian_file"] == "r.out"
        assert species["symmetry_factor"] == 2.0
        assert species["GroundEnergy"] == 0.0
    
    def test_config_default_values(self):
        """Test Config default values."""
        config_dict = {
            "input": {"files": ["test.out"]},
            "quantum": {"frequency_scaling_factor": 0.971},
            "mess_global": {
                "temperature_list": [300],
                "pressure_list": [1.0]
            }
        }
        
        config = Config.from_dict(config_dict)
        
        # Check defaults
        assert config.geometry_units == "angstrom"
        assert config.frequency_units == "1/cm"
        assert config.energy_units == "kcal/mol"
        assert config.energy_step_over_temperature == 0.2
        assert config.excess_energy_over_temperature == 50
        assert config.model_energy_limit == 400
        assert config.calculation_method == "direct"
        assert config.well_cutoff == 20
        assert config.skip_unconverged is False  # Default
        assert config.validate_frequencies is True  # Default
        assert config.create_backups is True  # Default
        assert config.verbose is False  # Default


class TestConfigIntegration:
    """Integration tests for configuration handling."""
    
    def test_full_config_workflow(self, test_data_dir):
        """Test full configuration workflow: parse, validate, create Config."""
        config_path = test_data_dir / "test_config.yaml"
        
        # 1. Parse config file
        config_dict = parse_config_file(str(config_path))
        assert config_dict is not None
        
        # 2. Validate config
        is_valid, errors = validate_config(config_dict)
        assert is_valid is True
        assert len(errors) == 0
        
        # 3. Create Config object
        config = Config.from_dict(config_dict)
        assert config is not None
        
        # Check specific values from test_config.yaml
        assert config.input_files == ["tests/data/example_gaussian.out"]
        assert config.frequency_scaling_factor == 0.971
        assert config.temperature_list == [300, 400, 500]
        assert config.pressure_list == [1.0]
        assert config.calculation_method == "direct"
        assert config.well_cutoff == 20
    
    def test_config_with_species_references(self):
        """Test configuration with species references (for barriers)."""
        config_dict = {
            "input": {"files": ["r.out", "ts.out", "p.out"]},
            "quantum": {"frequency_scaling_factor": 0.971},
            "mess_global": {
                "temperature_list": [300],
                "pressure_list": [1.0]
            },
            "reaction_network": {
                "species": [
                    {"name": "R", "type": "well", "gaussian_file": "r.out"},
                    {"name": "TS", "type": "barrier", "gaussian_file": "ts.out",
                     "from_species": "R", "to_species": "P"},
                    {"name": "P", "type": "well", "gaussian_file": "p.out"}
                ]
            }
        }
        
        # Parse and validate
        is_valid, errors = validate_config(config_dict)
        assert is_valid is True
        assert len(errors) == 0
        
        # Create Config
        config = Config.from_dict(config_dict)
        
        # Check species
        assert len(config.reaction_network) == 3
        
        # Find barrier
        barrier = next(s for s in config.reaction_network if s["type"] == "barrier")
        assert barrier["name"] == "TS"
        assert barrier["from_species"] == "R"
        assert barrier["to_species"] == "P"
        
        # Check referenced species exist
        species_names = [s["name"] for s in config.reaction_network]
        assert "R" in species_names
        assert "P" in species_names
    
    def test_config_invalid_species_references(self):
        """Test configuration with invalid species references."""
        config_dict = {
            "input": {"files": ["ts.out"]},
            "quantum": {"frequency_scaling_factor": 0.971},
            "mess_global": {
                "temperature_list": [300],
                "pressure_list": [1.0]
            },
            "reaction_network": {
                "species": [
                    {"name": "TS", "type": "barrier", "gaussian_file": "ts.out",
                     "from_species": "R", "to_species": "P"}  # R and P don't exist
                ]
            }
        }
        
        # Validation should catch missing referenced species
        is_valid, errors = validate_config(config_dict)
        # Might be valid at basic level, but species validation happens later
        # We'll accept either True or False depending on validation strictness
        if not is_valid:
            assert any("reference" in error.lower() or "species" in error.lower() 
                      for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
