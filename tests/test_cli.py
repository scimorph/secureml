"""
Tests for the command-line interface of SecureML.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from click.testing import CliRunner

from secureml.cli import (
    cli,
    anonymization,
    compliance,
    synthetic,
    presets
)


class TestCLI(unittest.TestCase):
    """Test cases for the SecureML CLI."""
    
    def setUp(self):
        """Set up the CLI test runner and temporary files."""
        self.runner = CliRunner()
        
        # Create a temporary directory and file for tests
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple test CSV file
        self.test_csv = os.path.join(self.temp_dir.name, "test_data.csv")
        pd.DataFrame({
            "name": ["John Doe", "Jane Smith"],
            "email": ["john@example.com", "jane@example.com"],
            "age": [35, 28],
            "income": [75000, 82000]
        }).to_csv(self.test_csv, index=False)
        
        # Create test metadata file
        self.test_metadata = os.path.join(self.temp_dir.name, "test_metadata.json")
        with open(self.test_metadata, "w") as f:
            f.write('{"contains_ca_residents": true, "ccpa_disclosure_provided": false}')
        
        # Create test model config file
        self.test_model_config = os.path.join(self.temp_dir.name, "test_model.json")
        with open(self.test_model_config, "w") as f:
            f.write('{"model_type": "random_forest", "supports_explanation": false}')
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    @patch("secureml.cli.anonymize")
    def test_anonymization_k_anonymize(self, mock_anonymize):
        """Test the k-anonymize command."""
        # Mock the k_anonymize function to return a DataFrame
        mock_anonymize.k_anonymize.return_value = pd.DataFrame({
            "name": ["ANONYMIZED", "ANONYMIZED"],
            "email": ["ANONYMIZED", "ANONYMIZED"],
            "age": [35, 28],
            "income": [75000, 82000]
        })
        
        # Create output path
        output_file = os.path.join(self.temp_dir.name, "output.csv")
        
        # Run the command
        result = self.runner.invoke(
            anonymization, 
            [
                "k-anonymize", 
                self.test_csv, 
                output_file,
                "--quasi-id", "age",
                "--sensitive", "income",
                "--k", "2"
            ]
        )
        
        # Check the command succeeded
        self.assertEqual(result.exit_code, 0)
        
        # Check that k_anonymize was called with the correct arguments
        mock_anonymize.k_anonymize.assert_called_once()
        args, kwargs = mock_anonymize.k_anonymize.call_args
        self.assertEqual(kwargs["quasi_identifiers"], ["age"])
        self.assertEqual(kwargs["sensitive_columns"], ["income"])
        self.assertEqual(kwargs["k"], 2)
    
    @patch("secureml.cli.check_compliance")
    def test_compliance_check(self, mock_check_compliance):
        """Test the compliance check command."""
        # Mock the ComplianceReport to return a report object
        mock_report = MagicMock()
        mock_report.issues = []
        mock_report.has_issues.return_value = False
        mock_check_compliance.return_value = mock_report
        
        # Run the command
        result = self.runner.invoke(
            compliance,
            [
                "check",
                self.test_csv,
                "--regulation", "GDPR",
                "--metadata", self.test_metadata,
                "--model-config", self.test_model_config
            ]
        )
        
        # Check the command succeeded
        self.assertEqual(result.exit_code, 0)
        
        # Check that check_compliance was called with the correct arguments
        mock_check_compliance.assert_called_once()
        args, kwargs = mock_check_compliance.call_args
        self.assertEqual(kwargs["regulation"], "GDPR")
    
    @patch("secureml.cli.generate_synthetic_data")
    def test_synthetic_generate(self, mock_generate_synthetic_data):
        """Test the synthetic data generation command."""
        # Mock generate_synthetic_data to return a DataFrame
        mock_generate_synthetic_data.return_value = pd.DataFrame({
            "name": ["Synthetic1", "Synthetic2", "Synthetic3"],
            "email": ["synth1@example.com", "synth2@example.com", "synth3@example.com"],
            "age": [30, 40, 50],
            "income": [60000, 70000, 80000]
        })
        
        # Create output path
        output_file = os.path.join(self.temp_dir.name, "synthetic.csv")
        
        # Run the command
        result = self.runner.invoke(
            synthetic,
            [
                "generate",
                self.test_csv,
                output_file,
                "--method", "statistical",
                "--samples", "3",
                "--sensitive", "name",
                "--sensitive", "email"
            ]
        )
        
        # Check the command succeeded
        self.assertEqual(result.exit_code, 0)
        
        # Check that generate_synthetic_data was called with the correct arguments
        mock_generate_synthetic_data.assert_called_once()
        args, kwargs = mock_generate_synthetic_data.call_args
        self.assertEqual(kwargs["method"], "statistical")
        self.assertEqual(kwargs["num_samples"], 3)
        self.assertEqual(kwargs["sensitive_columns"], ["name", "email"])
    
    @patch("secureml.cli.list_available_presets")
    def test_presets_list(self, mock_list_available_presets):
        """Test the presets list command."""
        # Mock list_available_presets to return a list of presets
        mock_list_available_presets.return_value = ["gdpr", "ccpa", "hipaa"]
        
        # Run the command
        result = self.runner.invoke(presets, ["list"])
        
        # Check the command succeeded
        self.assertEqual(result.exit_code, 0)
        
        # Check that list_available_presets was called
        mock_list_available_presets.assert_called_once()
        
        # Check output contains all presets
        for preset in ["gdpr", "ccpa", "hipaa"]:
            self.assertIn(preset, result.output)
    
    @patch("secureml.cli.load_preset")
    def test_presets_show(self, mock_load_preset):
        """Test the presets show command."""
        # Mock load_preset to return a preset dictionary
        mock_load_preset.return_value = {
            "regulation": {
                "name": "GDPR",
                "description": "General Data Protection Regulation",
                "effective_date": "2018-05-25"
            },
            "personal_data_identifiers": ["name", "email", "address"]
        }
        
        # Run the command
        result = self.runner.invoke(presets, ["show", "gdpr"])
        
        # Check the command succeeded
        self.assertEqual(result.exit_code, 0)
        
        # Check that load_preset was called with the correct arguments
        mock_load_preset.assert_called_with("gdpr")
        
        # Check output contains preset information
        self.assertIn("GDPR", result.output)
        self.assertIn("General Data Protection Regulation", result.output)


# Add pytest-style tests
@pytest.fixture
def runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    file_path = tmp_path / "test_data.csv"
    pd.DataFrame({
        "name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "email": ["john@example.com", "jane@example.com", "bob@example.com"],
        "age": [35, 28, 42],
        "income": [75000, 82000, 65000]
    }).to_csv(file_path, index=False)
    return file_path


@pytest.mark.parametrize(
    "regulation", ["GDPR", "CCPA", "HIPAA"]
)
@patch("secureml.cli.check_compliance")
def test_compliance_check_with_different_regulations(
    mock_check_compliance, runner, temp_csv, regulation
):
    """Test compliance check with different regulations."""
    # Mock the compliance report
    mock_report = MagicMock()
    mock_report.has_issues.return_value = False
    mock_check_compliance.return_value = mock_report
    
    # Run the command
    result = runner.invoke(
        compliance,
        [
            "check",
            str(temp_csv),
            "--regulation", regulation
        ]
    )
    
    # Check the command succeeded
    assert result.exit_code == 0
    
    # Check that check_compliance was called with the correct regulation
    mock_check_compliance.assert_called_once()
    args, kwargs = mock_check_compliance.call_args
    assert kwargs["regulation"] == regulation


@pytest.mark.parametrize(
    "output_format", ["csv", "json", "parquet"]
)
@patch("secureml.cli.anonymize")
def test_anonymization_output_formats(
    mock_anonymize, runner, temp_csv, tmp_path, output_format
):
    """Test anonymization with different output formats."""
    # Mock the k_anonymize function
    mock_anonymize.k_anonymize.return_value = pd.DataFrame({
        "name": ["ANON1", "ANON2", "ANON3"],
        "age": [30, 30, 40]
    })
    
    # Create output path with the appropriate extension
    output_file = tmp_path / f"output.{output_format}"
    
    # Run the command
    result = runner.invoke(
        anonymization,
        [
            "k-anonymize",
            str(temp_csv),
            str(output_file),
            "--format", output_format
        ]
    )
    
    # Check the command succeeded
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "method", ["simple", "statistical", "sdv-copula"]
)
@patch("secureml.cli.generate_synthetic_data")
def test_synthetic_different_methods(
    mock_generate_synthetic_data, runner, temp_csv, tmp_path, method
):
    """Test synthetic data generation with different methods."""
    # Mock generate_synthetic_data
    mock_generate_synthetic_data.return_value = pd.DataFrame({
        "name": ["Synthetic1", "Synthetic2"],
        "age": [30, 40]
    })
    
    # Create output path
    output_file = tmp_path / "synthetic.csv"
    
    # Run the command
    result = runner.invoke(
        synthetic,
        [
            "generate",
            str(temp_csv),
            str(output_file),
            "--method", method
        ]
    )
    
    # Check the command succeeded
    assert result.exit_code == 0
    
    # Check that generate_synthetic_data was called with the correct method
    mock_generate_synthetic_data.assert_called_once()
    args, kwargs = mock_generate_synthetic_data.call_args
    assert kwargs["method"] == method


if __name__ == "__main__":
    unittest.main() 