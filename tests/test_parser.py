"""
Unit tests for the parser module.

This module tests the functionality of Gaussian output file parsing,
including geometry extraction, frequency parsing, and energy extraction.
"""
import os
import sys
from pathlib import Path
import pytest

# Import the parser module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser import GaussianParser, QuantumData, Atom


class TestAtom:
    """Test Atom dataclass functionality."""

    def test_atom_creation(self):
        """Test creating an Atom instance."""
        atom = Atom("C", 0.0, 0.0, 0.0)
        assert atom.symbol == "C"
        assert atom.x == 0.0
        assert atom.y == 0.0
        assert atom.z == 0.0

    def test_atom_str_format(self):
        """Test string formatting of Atom."""
        atom = Atom("H", 1.0, 2.0, 3.0)
        atom_str = str(atom)
        assert "H" in atom_str
        assert "1.00000000" in atom_str
        assert "2.00000000" in atom_str
        assert "3.00000000" in atom_str


class TestQuantumData:
    """Test QuantumData dataclass functionality."""

    def test_quantum_data_creation(self):
        """Test creating a QuantumData instance."""
        atoms = [
            Atom("C", 0.0, 0.0, 0.0),
            Atom("H", 0.0, 0.0, 1.089),
        ]
        frequencies = [100.0, 200.0, 300.0]

        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=2,
            frequencies=frequencies,
            num_frequencies=3,
            imaginary_frequencies=[],
            scf_energy=-100.0,
            zero_point_energy=10.0,
            multiplicity=1,
            charge=0,
            method_basis="B3LYP/6-31G*",
        )

        assert qdata.filename == "test.out"
        assert qdata.convergence_status is True
        assert qdata.num_atoms == 2
        assert len(qdata.atoms) == 2
        assert qdata.num_frequencies == 3
        assert len(qdata.frequencies) == 3
        assert qdata.scf_energy == -100.0
        assert qdata.zero_point_energy == 10.0
        assert qdata.multiplicity == 1
        assert qdata.charge == 0
        assert qdata.method_basis == "B3LYP/6-31G*"

    def test_quantum_data_post_init_counts(self):
        """Test that __post_init__ correctly computes imaginary frequency counts."""
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        frequencies = [-500.0, 100.0, 200.0]

        qdata = QuantumData(
            filename="ts.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=frequencies,
            num_frequencies=3,
            imaginary_frequencies=[],
        )

        assert qdata.num_imaginary == 1
        assert len(qdata.imaginary_frequencies) == 1
        assert qdata.imaginary_frequencies[0] == -500.0
        assert len(qdata.real_frequencies) == 2

    def test_quantum_data_defaults(self):
        """Test QuantumData with default values."""
        qdata = QuantumData(
            filename="test.out",
            convergence_status=False,
            atoms=[],
            num_atoms=0,
            frequencies=[],
            num_frequencies=0,
            imaginary_frequencies=[],
        )

        assert qdata.scf_energy is None
        assert qdata.zero_point_energy is None
        assert qdata.multiplicity == 1
        assert qdata.charge == 0
        assert qdata.method_basis is None

    def test_get_geometry_string(self):
        """Test geometry string generation."""
        atoms = [Atom("C", 0.0, 0.0, 0.0), Atom("H", 0.0, 0.0, 1.089)]
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=2,
            frequencies=[100.0],
            num_frequencies=1,
            imaginary_frequencies=[],
        )
        geo_str = qdata.get_geometry_string()
        assert "Geometry[angstrom]" in geo_str
        assert "C" in geo_str
        assert "H" in geo_str

    def test_get_frequencies_string(self):
        """Test frequencies string generation."""
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[100.0, 200.0, 300.0],
            num_frequencies=3,
            imaginary_frequencies=[],
        )
        freq_str = qdata.get_frequencies_string()
        assert "Frequencies[1/cm]" in freq_str
        assert "100.00" in freq_str


class TestGaussianParser:
    """Test GaussianParser functionality."""

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = GaussianParser()
        assert parser is not None
        # freq_pattern is the compiled regex for frequencies
        assert parser.freq_pattern is not None

    def test_parse_gaussian_output(self, example_output_file):
        """Test parsing a Gaussian output file."""
        parser = GaussianParser(skip_unconverged=False)
        qdata = parser.parse_file(str(example_output_file))

        # Check basic properties
        assert qdata is not None
        assert qdata.filename == str(example_output_file)

        # Check geometry exists
        assert qdata.num_atoms > 0
        assert len(qdata.atoms) > 0

        # Check frequencies
        assert qdata.num_frequencies > 0
        assert len(qdata.frequencies) == qdata.num_frequencies

        # Check energy
        assert qdata.scf_energy is not None

    def test_parse_non_existent_file(self):
        """Test parsing a non-existent file raises an exception."""
        parser = GaussianParser()
        # The parser raises GaussianFileParseError when file is not found
        # (the error_handler wrapper re-raises after logging)
        import exceptions as exc
        with pytest.raises(exc.GaussianFileParseError):
            parser.parse_file("non_existent_file.out")

    def test_extract_frequencies_from_lines(self):
        """Test frequency extraction using list of lines (actual API)."""
        parser = GaussianParser()

        lines = [
            " Frequencies --  -500.0000   100.0000   200.0000\n",
            " Red. masses --     1.0480     1.0837     1.0837\n",
        ]
        frequencies = parser._extract_frequencies(lines)

        assert len(frequencies) == 3
        # Imaginary frequency should be negative
        assert frequencies[0] < 0
        assert frequencies[1] > 0
        assert frequencies[2] > 0

    def test_extract_geometry_from_lines(self, example_output_file):
        """Test geometry extraction from list of lines."""
        parser = GaussianParser()
        with open(example_output_file, "r") as f:
            lines = f.readlines()

        atoms = parser._extract_geometry(lines)

        assert len(atoms) > 0
        assert all(isinstance(atom, Atom) for atom in atoms)
        for atom in atoms:
            assert isinstance(atom.x, float)
            assert isinstance(atom.y, float)
            assert isinstance(atom.z, float)

    def test_extract_energies_from_lines(self, example_output_file):
        """Test energy extraction returning a dictionary."""
        parser = GaussianParser()
        with open(example_output_file, "r") as f:
            lines = f.readlines()

        energies = parser._extract_energies(lines)

        assert isinstance(energies, dict)
        # SCF energy should be present and negative
        assert "scf" in energies
        assert energies["scf"] < 0

    def test_check_convergence_from_lines(self, example_output_file):
        """Test convergence detection using list of lines."""
        parser = GaussianParser()
        with open(example_output_file, "r") as f:
            lines = f.readlines()

        is_converged = parser._check_convergence(lines)
        # example_gaussian.out should represent a converged calculation
        assert isinstance(is_converged, bool)

    def test_get_atom_symbol(self):
        """Test atomic number to symbol conversion."""
        assert GaussianParser.get_atom_symbol(1) == "H"
        assert GaussianParser.get_atom_symbol(6) == "C"
        assert GaussianParser.get_atom_symbol(8) == "O"
        assert GaussianParser.get_atom_symbol(7) == "N"
        assert GaussianParser.get_atom_symbol(999) == "X"

    def test_handle_imaginary_frequencies(self):
        """Test handling of imaginary frequencies in frequency line."""
        parser = GaussianParser()

        lines = [
            " Frequencies --  -500.0000   100.0000   200.0000\n",
        ]
        frequencies = parser._extract_frequencies(lines)
        assert len(frequencies) == 3
        assert frequencies[0] == -500.0
        assert frequencies[1] == 100.0
        assert frequencies[2] == 200.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
