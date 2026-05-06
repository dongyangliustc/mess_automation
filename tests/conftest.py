"""
Pytest configuration and shared fixtures for MESS automation tests.
"""
import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def example_output_file():
    """Return a sample Gaussian output file path."""
    return TEST_DATA_DIR / "example_gaussian.out"


@pytest.fixture
def example_config_file():
    """Return a sample configuration file path."""
    return TEST_DATA_DIR / "test_config.yaml"


@pytest.fixture
def scaling_factor():
    """Return a standard scaling factor for frequency correction tests."""
    return 0.971