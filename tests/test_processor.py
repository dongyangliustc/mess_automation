"""
Unit tests for the processor module.

This module tests frequency scaling, unit conversion, energy calculation,
and imaginary frequency handling.
"""
import os
import sys
from pathlib import Path
import pytest

# Import the processor module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processor import (
    FrequencyCorrector,
    UnitConverter,
    CorrectionResult,
    create_molecule_object
)
from parser import QuantumData, Atom


class TestUnitConverter:
    """Test UnitConverter functionality."""
    
    def test_convert_energy_hartree_to_kcal(self):
        """Test conversion from Hartree to kcal/mol."""
        # 1 Hartree = 627.509474 kcal/mol
        hartree_value = 1.0
        kcal_value = UnitConverter.convert_energy(hartree_value, "hartree", "kcal/mol")
        expected = 627.509474
        assert abs(kcal_value - expected) < 0.001
    
    def test_convert_energy_kcal_to_hartree(self):
        """Test conversion from kcal/mol to Hartree."""
        kcal_value = 627.509474
        hartree_value = UnitConverter.convert_energy(kcal_value, "kcal/mol", "hartree")
        expected = 1.0
        assert abs(hartree_value - expected) < 0.001
    
    def test_convert_energy_same_unit(self):
        """Test conversion with same unit."""
        value = 100.0
        result = UnitConverter.convert_energy(value, "kcal/mol", "kcal/mol")
        assert result == value
    
    def test_convert_energy_invalid_unit(self):
        """Test conversion with invalid unit."""
        with pytest.raises(ValueError):
            UnitConverter.convert_energy(100.0, "invalid_unit", "kcal/mol")
    
    def test_convert_length_angstrom_to_bohr(self):
        """Test conversion from Angstrom to Bohr (method: convert_length)."""
        # 1 Å ≈ 1.88972612456506 Bohr
        angstrom_value = 1.0
        bohr_value = UnitConverter.convert_length(angstrom_value, "angstrom", "bohr")
        expected = 1.88972612456506
        assert abs(bohr_value - expected) < 0.001

    def test_convert_length_bohr_to_angstrom(self):
        """Test conversion from Bohr to Angstrom."""
        bohr_value = 1.88972612456506
        angstrom_value = UnitConverter.convert_length(bohr_value, "bohr", "angstrom")
        expected = 1.0
        assert abs(angstrom_value - expected) < 0.001

    def test_convert_frequency_cm_to_thz(self):
        """Test conversion from cm^-1 to THz."""
        # 1 cm^-1 = 0.0299792458 THz
        # UnitConverter accepts "cm^-1", "cm-1", "cm", "wavenumber" as from_unit
        cm_value = 100.0
        thz_value = UnitConverter.convert_frequency(cm_value, "cm^-1", "thz")
        expected = 100.0 * 0.0299792458
        assert abs(thz_value - expected) < 0.001


class TestCorrectionResult:
    """Test CorrectionResult dataclass."""
    
    def test_correction_result_creation(self):
        """Test creating a CorrectionResult instance."""
        # Create test QuantumData
        atoms = [
            Atom("C", 0.0, 0.0, 0.0),
            Atom("H", 0.0, 0.0, 1.089),
        ]
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=2,
            frequencies=[100.0, 200.0, 300.0],
            num_frequencies=3,
            scf_energy=-100.0,
            zero_point_energy=10.0,
            multiplicity=1,
            charge=0,
            method_basis="B3LYP/6-31G*"
        )
        
        # Create CorrectionResult
        scaled_frequencies = [90.0, 180.0, 270.0]
        result = CorrectionResult(
            original_data=qdata,
            scaled_frequencies=scaled_frequencies,
            scaling_factor=0.9,
            corrections_applied=["frequency_scaling"]
        )
        
        # Check basic properties
        assert result.original_data == qdata
        assert result.scaled_frequencies == scaled_frequencies
        assert result.scaling_factor == 0.9
        assert "frequency_scaling" in result.corrections_applied
        assert result.success is True
    
    def test_correction_result_with_imaginary_frequencies(self):
        """Test CorrectionResult with imaginary frequencies."""
        # Create test QuantumData with imaginary frequency
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        qdata = QuantumData(
            filename="test_ts.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[-500.0, 100.0, 200.0],
            num_frequencies=3,
            scf_energy=-150.0,
            zero_point_energy=5.0,
        )
        
        # Apply scaling factor
        scaled_frequencies = [-450.0, 90.0, 180.0]  # 0.9 scaling
        result = CorrectionResult(
            original_data=qdata,
            scaled_frequencies=scaled_frequencies,
            scaling_factor=0.9,
        )
        
        # Check imaginary frequency extraction
        assert result.imaginary_frequency is not None
        # Should be absolute value of scaled imaginary frequency
        assert abs(result.imaginary_frequency - 450.0) < 0.001
        
        # Check real frequencies
        assert len(result.real_frequencies) == 2
        assert all(f > 0 for f in result.real_frequencies)
        assert 90.0 in result.real_frequencies
        assert 180.0 in result.real_frequencies
    
    def test_total_energy_calculation(self):
        """Test total energy calculation."""
        # Create test QuantumData with SCF and ZPE
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[100.0],
            num_frequencies=1,
            scf_energy=-100.0,  # Hartree
            zero_point_energy=0.01,  # Hartree
        )
        
        result = CorrectionResult(
            original_data=qdata,
            scaled_frequencies=[90.0],
            scaling_factor=0.9,
        )
        
        # Check total energy calculation
        assert result.total_energy is not None
        
        # Total energy is (SCF + ZPE) converted from Hartree to kcal/mol.
        expected_total = (-100.0 + 0.01) * 627.509474
        
        assert abs(result.total_energy - expected_total) < 0.1
    
    def test_total_energy_no_zpe(self):
        """Test total energy calculation without ZPE."""
        # Create test QuantumData without ZPE
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[100.0],
            num_frequencies=1,
            scf_energy=-100.0,  # Hartree
            zero_point_energy=None,  # No ZPE
        )
        
        result = CorrectionResult(
            original_data=qdata,
            scaled_frequencies=[90.0],
            scaling_factor=0.9,
        )
        
        # Should use only SCF energy
        assert result.total_energy is not None
        expected = -100.0 * 627.509474
        assert abs(result.total_energy - expected) < 0.1
    
    def test_total_energy_no_scf(self):
        """Test total energy calculation without SCF energy."""
        # Create test QuantumData without SCF
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[100.0],
            num_frequencies=1,
            scf_energy=None,  # No SCF
            zero_point_energy=0.01,  # Hartree
        )
        
        result = CorrectionResult(
            original_data=qdata,
            scaled_frequencies=[90.0],
            scaling_factor=0.9,
        )
        
        # Should use only ZPE converted from Hartree to kcal/mol
        assert abs(result.total_energy - (0.01 * 627.509474)) < 0.1


class TestFrequencyCorrector:
    """Test FrequencyCorrector functionality."""
    
    def test_corrector_initialization(self):
        """Test FrequencyCorrector initialization."""
        corrector = FrequencyCorrector(scaling_factor=0.971)
        assert corrector.scaling_factor == 0.971
        assert corrector.frequency_factor == 0.971
        assert corrector.zpe_factor == 1.0
        assert corrector.handle_imaginary == "abs"
        
        # Test with different imaginary handling
        corrector2 = FrequencyCorrector(scaling_factor=0.971, handle_imaginary="remove")
        assert corrector2.handle_imaginary == "remove"
    
    def test_correct_frequencies_basic(self):
        """Test basic frequency correction."""
        # Create test QuantumData
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[100.0, 200.0, 300.0],
            num_frequencies=3,
            scf_energy=-100.0,
            zero_point_energy=0.01,
        )
        
        corrector = FrequencyCorrector(scaling_factor=0.9)
        result = corrector.correct_frequencies(qdata)
        
        # Check result
        assert result is not None
        assert result.success is True
        assert result.scaling_factor == 0.9
        assert "frequency_scaling" in result.corrections_applied
        
        # Check scaled frequencies
        assert len(result.scaled_frequencies) == 3
        assert result.scaled_frequencies[0] == 90.0  # 100 * 0.9
        assert result.scaled_frequencies[1] == 180.0  # 200 * 0.9
        assert result.scaled_frequencies[2] == 270.0  # 300 * 0.9
        
        # Check other properties
        assert result.scaled_zpe == 0.01
        assert result.total_energy is not None

    def test_correct_frequencies_with_separate_zpe_factor(self):
        """Test independent frequency and ZPE scaling factors."""
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=[Atom("C", 0.0, 0.0, 0.0)],
            num_atoms=1,
            frequencies=[100.0, 200.0],
            num_frequencies=2,
            scf_energy=-100.0,
            zero_point_energy=0.01,
        )

        corrector = FrequencyCorrector(Frequency_factor=0.9, zpe_factor=0.5)
        result = corrector.correct_frequencies(qdata)

        assert result.frequency_factor == 0.9
        assert result.scaling_factor == 0.9
        assert result.zpe_factor == 0.5
        assert result.scaled_frequencies == [90.0, 180.0]
        assert result.scaled_zpe == 0.005
    
    def test_correct_frequencies_with_imaginary(self):
        """Test frequency correction with imaginary frequencies."""
        # Create test QuantumData with imaginary frequency
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        qdata = QuantumData(
            filename="test_ts.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[-500.0, 100.0, 200.0],
            num_frequencies=3,
            scf_energy=-150.0,
            zero_point_energy=5.0,
        )
        
        corrector = FrequencyCorrector(scaling_factor=0.9, handle_imaginary="abs")
        result = corrector.correct_frequencies(qdata)
        
        # Check result
        assert result is not None
        assert result.success is True
        
        # Check imaginary frequency handling
        assert result.imaginary_frequency is not None
        # Should be absolute value of scaled imaginary frequency
        expected_imag = 500.0 * 0.9  # 450.0
        assert abs(result.imaginary_frequency - expected_imag) < 0.001
        
        # Check scaled frequencies (including imaginary)
        assert len(result.scaled_frequencies) == 3
        # Imaginary frequency may be negative or positive depending on handle_imaginary
        if corrector.handle_imaginary == "abs":
            # With "abs", negative frequencies become positive
            assert result.scaled_frequencies[0] == expected_imag
        else:
            # With other handling, may remain negative
            assert result.scaled_frequencies[0] < 0
        
        # Check real frequencies list
        assert len(result.real_frequencies) == 2
        assert all(f > 0 for f in result.real_frequencies)
    
    def test_correct_frequencies_invalid_data(self):
        """Test frequency correction with invalid data."""
        # Create invalid QuantumData
        qdata = QuantumData(
            filename="test.out",
            convergence_status=False,  # Not converged
            atoms=[],
            num_atoms=0,
            frequencies=[],
            num_frequencies=0,
        )
        
        corrector = FrequencyCorrector(scaling_factor=0.9)
        result = corrector.correct_frequencies(qdata)
        
        # Should return unsuccessful result
        assert result is not None
        assert result.success is False
        assert result.error_message is not None
    
    def test_calculate_barrier_depths(self):
        """Test barrier depth calculation using FrequencyCorrector.calculate_barrier_depths."""
        atoms = [Atom("C", 0.0, 0.0, 0.0)]

        # TS has higher SCF energy than reactant (-99.0 > -100.0 in Hartree)
        ts_qdata = QuantumData(
            filename="ts.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[-500.0, 100.0],
            num_frequencies=2,
            imaginary_frequencies=[],
            scf_energy=-99.0,   # Hartree – higher than reactant
            zero_point_energy=4.0,
        )

        corrector = FrequencyCorrector(scaling_factor=0.971)
        ts_result = corrector.correct_frequencies(ts_qdata)

        # Reactant SCF: -100.0 Hartree, Product SCF: -101.0 Hartree
        updated_result = corrector.calculate_barrier_depths(
            ts_result,
            reactant_energy=-100.0,   # Hartree
            product_energy=-101.0,    # Hartree
            energy_units="hartree",
        )

        assert updated_result.forward_barrier is not None
        assert updated_result.reverse_barrier is not None

        # Forward barrier: TS(-99) - Reactant(-100) = +1 Hartree -> positive
        assert updated_result.forward_barrier > 0

        # Reverse barrier: TS(-99) - Product(-101) = +2 Hartree -> positive
        assert updated_result.reverse_barrier > 0


class TestCreateMoleculeObject:
    """Test create_molecule_object function."""
    
    def test_create_molecule_object_basic(self):
        """Test basic molecule object creation."""
        # Create test QuantumData
        atoms = [
            Atom("C", 0.0, 0.0, 0.0),
            Atom("H", 0.0, 0.0, 1.089),
        ]
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=2,
            frequencies=[100.0, 200.0, 300.0],
            num_frequencies=3,
            scf_energy=-100.0,
            zero_point_energy=10.0,
        )
        
        # Create molecule object without correction
        molecule = create_molecule_object(
            qdata=qdata,
            name="TestMolecule",
            species_type="RRHO",
            symmetry_factor=2.0,
            ground_energy=5.0
        )
        
        # Check molecule properties
        assert molecule["name"] == "TestMolecule"
        assert molecule["type"] == "RRHO"
        assert molecule["num_atoms"] == 2
        assert len(molecule["atoms"]) == 2
        assert molecule["num_frequencies"] == 3
        assert len(molecule["frequencies"]) == 3
        assert molecule["scf_energy"] == -100.0
        assert molecule["zero_point_energy"] == 10.0
        assert molecule["symmetry_factor"] == 2.0
        assert molecule["GroundEnergy"] == 5.0
        assert molecule["multiplicity"] == 1
        assert molecule["charge"] == 0
    
    def test_create_molecule_object_with_correction(self):
        """Test molecule object creation with frequency correction."""
        # Create test QuantumData
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        qdata = QuantumData(
            filename="test.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[100.0, 200.0],
            num_frequencies=2,
            scf_energy=-100.0,
            zero_point_energy=10.0,
        )
        
        # Create CorrectionResult; __post_init__ recomputes total_energy
        # from SCF and ZPE in Hartree.
        result = CorrectionResult(
            original_data=qdata,
            scaled_frequencies=[90.0, 180.0],  # 0.9 scaling
            scaling_factor=0.9,
            corrections_applied=["frequency_scaling"],
        )
        
        # Create molecule object with correction
        molecule = create_molecule_object(
            qdata=qdata,
            correction=result,
            name="TestMolecule",
            species_type="Well",
            symmetry_factor=1.0,
            ground_energy=0.0
        )
        
        # Should use corrected frequencies
        assert molecule["frequencies"] == [90.0, 180.0]
        # total_energy is (SCF + ZPE) converted from Hartree.
        assert molecule["total_energy"] is not None
        expected_total = (-100.0 + 0.01) * 627.509474
        assert abs(molecule["total_energy"] - expected_total) < 0.1
    
    def test_create_molecule_object_imaginary_frequencies(self):
        """Test molecule object creation with imaginary frequencies."""
        # Create test QuantumData with imaginary frequency
        atoms = [Atom("C", 0.0, 0.0, 0.0)]
        qdata = QuantumData(
            filename="test_ts.out",
            convergence_status=True,
            atoms=atoms,
            num_atoms=1,
            frequencies=[-500.0, 100.0, 200.0],
            num_frequencies=3,
            scf_energy=-150.0,
            zero_point_energy=5.0,
        )
        
        # Create CorrectionResult
        result = CorrectionResult(
            original_data=qdata,
            scaled_frequencies=[-450.0, 90.0, 180.0],  # 0.9 scaling
            scaling_factor=0.9,
            imaginary_frequency=450.0,
            real_frequencies=[90.0, 180.0]
        )
        
        # Create molecule object
        molecule = create_molecule_object(
            qdata=qdata,
            correction=result,
            name="TS",
            species_type="Barrier",
            symmetry_factor=0.5,
            ground_energy=10.0
        )
        
        # Check imaginary frequency properties
        assert molecule["num_imaginary"] == 1
        assert len(molecule["imaginary_frequencies"]) == 1
        assert molecule["imaginary_frequencies"][0] < 0  # Negative frequency


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
