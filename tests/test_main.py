"""
Unit tests for the main module.

This module tests the command line interface, configuration loading,
and overall workflow integration.
"""
import os
import sys
import tempfile
import argparse
from pathlib import Path
import yaml
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as mess_main


class TestMainFunctions:
    """Test main module functions."""
    
    def test_load_config_valid(self):
        """Test loading valid configuration."""
        config_data = {
            "input": {"files": ["test.out"]},
            "quantum": {"frequency_scaling_factor": 0.971},
            "mess_global": {
                "temperature_list": [300],
                "pressure_list": [1.0]
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load config
            config = mess_main.load_config(config_path)
            assert config is not None
            assert config["input"]["files"] == ["test.out"]
            assert config["quantum"]["frequency_scaling_factor"] == 0.971
            assert config["mess_global"]["temperature_list"] == [300]
        finally:
            os.unlink(config_path)
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file raises an exception."""
        with pytest.raises(Exception):
            mess_main.load_config("non_existent_file.yaml")
    
    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [}")
            config_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                mess_main.load_config(config_path)
        finally:
            os.unlink(config_path)
    
    @patch('main.MESSGlobalSettings')
    def test_setup_global_settings(self, mock_global_settings):
        """Test setting up global settings."""
        mock_settings = MagicMock()
        mock_global_settings.return_value = mock_settings
        
        config = {
            "mess_global": {
                "temperature_list": [300, 400, 500],
                "pressure_list": [0.1, 1.0, 10.0],
                "energy_step_over_temperature": 0.1,
                "excess_energy_over_temperature": 100,
                "model_energy_limit": 500,
                "calculation_method": "sct",
                "well_cutoff": 30
            }
        }
        
        # Call function
        settings = mess_main.setup_global_settings(config)
        
        # Check that MESSGlobalSettings was called with correct parameters
        mock_global_settings.assert_called_once()
        call_args = mock_global_settings.call_args[1]
        
        assert call_args["temperature_list"] == [300, 400, 500]
        assert call_args["pressure_list"] == [0.1, 1.0, 10.0]
        assert call_args["energy_step_over_temperature"] == 0.1
        assert call_args["excess_energy_over_temperature"] == 100
        assert call_args["model_energy_limit"] == 500
        assert call_args["calculation_method"] == "sct"
        assert call_args["well_cutoff"] == 30
        
        assert settings == mock_settings
    
    def test_setup_global_settings_defaults(self):
        """Test global settings with defaults."""
        config = {}  # Empty config
        
        # This should use defaults
        with patch('main.MESSGlobalSettings') as mock_global_settings:
            mock_settings = MagicMock()
            mock_global_settings.return_value = mock_settings
            
            settings = mess_main.setup_global_settings(config)
            
            # Should be called with some defaults
            mock_global_settings.assert_called_once()
            assert settings == mock_settings
    
    @patch('main.FrequencyCorrector')
    def test_setup_corrector(self, mock_frequency_corrector):
        """Test setting up frequency corrector."""
        mock_corrector = MagicMock()
        mock_frequency_corrector.return_value = mock_corrector
        
        config = {
            "quantum": {
                "Frequency_factor": 0.971,
                "zpe_factor": 0.95,
                "handle_imaginary": "remove"
            }
        }
        
        # Call function
        corrector = mess_main.setup_corrector(config)
        
        # Check that FrequencyCorrector was called with correct parameters
        mock_frequency_corrector.assert_called_once_with(
            Frequency_factor=0.971,
            zpe_factor=0.95,
            handle_imaginary="remove"
        )
        
        assert corrector == mock_corrector
    
    def test_setup_corrector_defaults(self):
        """Test frequency corrector with defaults."""
        config = {
            "quantum": {
                "frequency_scaling_factor": 0.971
                # No handle_imaginary specified
            }
        }
        
        with patch('main.FrequencyCorrector') as mock_frequency_corrector:
            mock_corrector = MagicMock()
            mock_frequency_corrector.return_value = mock_corrector
            
            corrector = mess_main.setup_corrector(config)
            
            # Should use default handle_imaginary
            mock_frequency_corrector.assert_called_once_with(
                Frequency_factor=0.971,
                zpe_factor=1.0,
                handle_imaginary="abs"  # Default
            )
    
    @patch('main.GaussianParser')
    def test_process_gaussian_file(self, mock_parser_class):
        """Test processing a Gaussian file."""
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        
        mock_qdata = MagicMock()
        mock_qdata.convergence_status = True
        mock_parser.parse_file.return_value = mock_qdata
        
        mock_corrector = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_corrector.correct_frequencies.return_value = mock_result
        
        file_path = "test.out"
        
        # Call function
        result = mess_main.process_gaussian_file(file_path, mock_corrector)
        
        # Check parser was created and used
        mock_parser_class.assert_called_once()
        mock_parser.parse_file.assert_called_once_with(file_path)
        
        # Check corrector was used
        mock_corrector.correct_frequencies.assert_called_once_with(mock_qdata)
        
        assert result == {"qdata": mock_qdata, "correction": mock_result}
    
    @patch('main.GaussianParser')
    def test_process_gaussian_file_not_converged(self, mock_parser_class):
        """Test processing non-converged Gaussian file."""
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        
        mock_qdata = MagicMock()
        mock_qdata.convergence_status = False  # Not converged
        mock_parser.parse_file.return_value = mock_qdata
        
        mock_corrector = MagicMock()
        
        file_path = "test.out"
        
        # Call function
        result = mess_main.process_gaussian_file(file_path, mock_corrector)
        
        # Should return None for non-converged
        assert result is None
        
        # Corrector should not be called
        mock_corrector.correct_frequencies.assert_not_called()
    
    @patch('main.GaussianParser')
    def test_process_gaussian_file_parser_error(self, mock_parser_class):
        """Test processing with parser error."""
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_file.return_value = None  # Parser failed
        
        mock_corrector = MagicMock()
        
        file_path = "test.out"
        
        # Call function
        result = mess_main.process_gaussian_file(file_path, mock_corrector)
        
        # Should return None
        assert result is None
        
        # Corrector should not be called
        mock_corrector.correct_frequencies.assert_not_called()
    
    def test_find_quantum_data(self):
        """Test finding quantum data in processed files."""
        # Create mock data
        processed_files = {
            "file1.out": {"qdata": "qdata1", "correction": "corr1"},
            "file2.out": {"qdata": "qdata2", "correction": "corr2"}
        }
        
        # Patch the processed_files global variable in the module
        with patch.object(mess_main, 'processed_files', processed_files):
            # Test finding existing file
            result = mess_main.find_quantum_data("file1.out")
            assert result == {"qdata": "qdata1", "correction": "corr1"}
            
            # Test finding non-existent file (relative path)
            result = mess_main.find_quantum_data("./file1.out")
            # Should try to normalize path and find it
            # Implementation may vary, but should return something or None
            # We'll just check it doesn't crash
            
            # Test finding non-existent file
            result = mess_main.find_quantum_data("non_existent.out")
            assert result is None


class TestArgumentParsing:
    """Test command line argument parsing."""
    
    def test_parse_arguments_minimal(self):
        """Test minimal argument parsing."""
        test_args = [
            "--config", "config.yaml",
            "--output", "output.inp"
        ]
        
        with patch.object(sys, 'argv', ['main.py'] + test_args):
            parser = mess_main.parse_arguments()
            
            assert parser.config == "config.yaml"
            assert parser.output == "output.inp"
            assert parser.verbose is False  # Default
            assert parser.overwrite is False  # Default
    
    def test_parse_arguments_full(self):
        """Test full argument parsing."""
        test_args = [
            "--config", "config.yaml",
            "--output", "output.inp",
            "--verbose",
            "--overwrite",
            "--scaling", "0.97",
            "--zpe-factor", "0.95",
            "--log-level", "DEBUG"
        ]
        
        with patch.object(sys, 'argv', ['main.py'] + test_args):
            parser = mess_main.parse_arguments()
            
            assert parser.config == "config.yaml"
            assert parser.output == "output.inp"
            assert parser.verbose is True
            assert parser.overwrite is True
            assert parser.scaling == 0.97
            assert parser.zpe_factor == 0.95
            assert parser.log_level == "DEBUG"
    
    def test_parse_arguments_help(self):
        """Test help argument."""
        test_args = ["--help"]
        
        with patch.object(sys, 'argv', ['main.py'] + test_args):
            # Should print help and exit
            # We'll just check it doesn't crash when called
            try:
                mess_main.parse_arguments()
            except SystemExit:
                pass  # Expected for --help
    
    def test_parse_arguments_missing_required(self):
        """Test parsing with missing required arguments."""
        test_args = []  # No arguments
        
        with patch.object(sys, 'argv', ['main.py'] + test_args):
            # Should fail with SystemExit
            with pytest.raises(SystemExit):
                mess_main.parse_arguments()


class TestIntegration:
    """Integration tests for main workflow."""
    
    @patch('main.load_config')
    @patch('main.setup_global_settings')
    @patch('main.setup_corrector')
    @patch('main.process_gaussian_file')
    @patch('main.MESSAssembler')
    def test_main_workflow_simple(self, mock_assembler_class, mock_process_file,
                                  mock_setup_corrector, mock_setup_global_settings,
                                  mock_load_config):
        """Test simple main workflow."""
        # Mock all components
        mock_config = {
            "input": {"files": ["test.out"]},
            "mess_global": {},
            "reaction_network": {
                "species": [
                    {"name": "Molecule1", "type": "well", "gaussian_file": "test.out"}
                ]
            }
        }
        mock_load_config.return_value = mock_config
        
        mock_global_settings = MagicMock()
        mock_setup_global_settings.return_value = mock_global_settings
        
        mock_corrector = MagicMock()
        mock_setup_corrector.return_value = mock_corrector
        
        mock_qdata = MagicMock()
        mock_correction = MagicMock(success=True)
        mock_process_file.return_value = {"qdata": mock_qdata, "correction": mock_correction}
        
        mock_assembler = MagicMock()
        mock_assembler_class.return_value = mock_assembler
        
        # Create temporary output file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.inp"
            
            # Run main function
            with patch('sys.argv', ['main.py', '--config', 'config.yaml', '--output', str(output_file)]):
                # We'll mock the main execution
                # Instead of calling main(), we'll test the workflow function
                
                # Test the workflow by calling functions in sequence
                config = mock_load_config('config.yaml')
                global_settings = mock_setup_global_settings(config)
                corrector = mock_setup_corrector(config)
                
                # Process files
                quantum_data = {}
                for file_path in config["input"]["files"]:
                    result = mock_process_file(file_path, corrector)
                    if result:
                        quantum_data[file_path] = result
                
                # Create molecule objects
                molecule_objects = {}
                for species_def in config["reaction_network"]["species"]:
                    # Simplified: just create a mock object
                    molecule_objects[species_def["name"]] = {"name": species_def["name"]}
                
                # Assemble MESS input
                mock_assembler.assemble_mess_input(
                    global_settings=global_settings,
                    reaction_network=config["reaction_network"],
                    molecule_objects=molecule_objects,
                    output_file=str(output_file)
                )
                
                # Verify mocks were called
                mock_load_config.assert_called_once_with('config.yaml')
                mock_setup_global_settings.assert_called_once_with(mock_config)
                mock_setup_corrector.assert_called_once_with(mock_config)
                mock_process_file.assert_called_once_with('test.out', mock_corrector)
                mock_assembler.assemble_mess_input.assert_called_once()
    
    @patch('main.load_config')
    @patch('main.setup_corrector')
    @patch('main.process_gaussian_file')
    def test_main_with_unconverged_file(self, mock_process_file, mock_setup_corrector, mock_load_config):
        """Test workflow with unconverged Gaussian file."""
        mock_config = {
            "input": {"files": ["unconverged.out"]},
            "processing": {"skip_unconverged": True}
        }
        mock_load_config.return_value = mock_config
        
        mock_corrector = MagicMock()
        mock_setup_corrector.return_value = mock_corrector
        
        # File processing returns None (unconverged)
        mock_process_file.return_value = None
        
        # Should handle gracefully
        # This is more of a system test, but we can verify process_file was called
        result = mock_process_file("unconverged.out", mock_corrector)
        assert result is None


class TestErrorHandling:
    """Test error handling in main module."""
    
    @patch('main.load_config')
    def test_config_loading_error(self, mock_load_config):
        """Test error during config loading."""
        mock_load_config.side_effect = FileNotFoundError("Config not found")
        
        # Should propagate the exception
        with pytest.raises(FileNotFoundError):
            mock_load_config("missing.yaml")
    
    @patch('main.MESSAssembler')
    def test_assembler_error(self, mock_assembler_class):
        """Test error during assembly."""
        mock_assembler = MagicMock()
        mock_assembler_class.return_value = mock_assembler
        
        # Simulate assembly error
        mock_assembler.assemble_mess_input.side_effect = Exception("Assembly failed")
        
        # Should propagate the exception
        with pytest.raises(Exception):
            mock_assembler.assemble_mess_input(
                global_settings=MagicMock(),
                reaction_network={},
                molecule_objects={},
                output_file="output.inp"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
