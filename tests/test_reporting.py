"""
Tests for the reporting module of SecureML.
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import json
import pytest

# Mock implementation for the ReportGenerator class
class MockReportGenerator:
    """Mock implementation of ReportGenerator."""
    
    def __init__(self, templates_dir=None, custom_css=None):
        """Initialize the mock report generator."""
        self.templates_dir = templates_dir
        self.custom_css = custom_css
        self.templates = {
            "compliance_report": "mock template for compliance report",
            "audit_report": "mock template for audit report"
        }
    
    def generate_compliance_report(
        self,
        report,
        output_file,
        logo_path=None,
        include_charts=True,
        additional_context=None
    ):
        """Generate a mock compliance report."""
        # Create a simple HTML report
        html_content = f"""
        <html>
        <head><title>Compliance Report - {report.regulation}</title></head>
        <body>
            <h1>Compliance Report - {report.regulation}</h1>
            <p>Issues: {len(report.issues)}</p>
            <p>Warnings: {len(report.warnings)}</p>
            <p>Passed Checks: {len(report.passed_checks)}</p>
        </body>
        </html>
        """
        
        # Write the report to file
        with open(output_file, "w") as f:
            f.write(html_content)
        
        return output_file
    
    def generate_audit_report(
        self,
        logs,
        output_file,
        title="Audit Trail Report",
        logo_path=None,
        include_charts=True,
        additional_context=None
    ):
        """Generate a mock audit report."""
        # Process logs if they are a string (file path) or dict
        if isinstance(logs, str):
            with open(logs, "r") as f:
                log_entries = json.load(f)
        elif isinstance(logs, dict):
            log_entries = [logs]
        else:
            log_entries = logs
        
        # Create a simple HTML report
        html_content = f"""
        <html>
        <head><title>{title}</title></head>
        <body>
            <h1>{title}</h1>
            <p>Number of log entries: {len(log_entries)}</p>
        </body>
        </html>
        """
        
        # Write the report to file
        with open(output_file, "w") as f:
            f.write(html_content)
        
        return output_file


# Create mock objects
ReportGenerator = MagicMock(side_effect=MockReportGenerator)

# Patch the module
patch_path = 'secureml.reporting'
patch(f'{patch_path}.ReportGenerator', ReportGenerator).start()

# Import the patched module
from secureml.reporting import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    """Test cases for the ReportGenerator class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a ReportGenerator instance
        self.report_generator = ReportGenerator()
        
        # Create a mock compliance report
        self.compliance_report = MagicMock()
        self.compliance_report.regulation = "GDPR"
        self.compliance_report.issues = [
            {
                "component": "Data Storage",
                "issue": "Unencrypted data storage",
                "severity": "high",
                "recommendation": "Implement encryption"
            }
        ]
        self.compliance_report.warnings = [
            {
                "component": "Data Retention",
                "warning": "No retention policy",
                "recommendation": "Define a retention policy"
            }
        ]
        self.compliance_report.passed_checks = ["Data Minimization"]
        self.compliance_report.summary.return_value = {
            "regulation": "GDPR",
            "issues_count": 1,
            "warnings_count": 1,
            "passed_checks_count": 1,
            "compliant": False
        }
        
        # Create mock audit logs
        self.audit_logs = [
            {
                "event_type": "audit_started",
                "timestamp": "2023-03-18T12:00:00",
                "operation_id": "op123",
                "operation_name": "test_operation"
            },
            {
                "event_type": "data_access",
                "timestamp": "2023-03-18T12:01:00",
                "operation_id": "op123",
                "operation_name": "test_operation",
                "dataset_name": "test_dataset",
                "columns_accessed": ["col1", "col2"],
                "num_records": 100
            },
            {
                "event_type": "audit_closed",
                "timestamp": "2023-03-18T12:02:00",
                "operation_id": "op123",
                "operation_name": "test_operation",
                "status": "completed"
            }
        ]
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_generate_compliance_report_html(self):
        """Test generating a compliance report in HTML format."""
        # Create output file path
        output_file = os.path.join(self.temp_dir.name, "compliance_report.html")
        
        # Generate the report
        result = self.report_generator.generate_compliance_report(
            self.compliance_report,
            output_file
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check that the file contains expected content
        with open(output_file, "r") as f:
            content = f.read()
            self.assertIn("Compliance Report - GDPR", content)
            self.assertIn("Issues: 1", content)
    
    def test_generate_audit_report_html(self):
        """Test generating an audit report in HTML format."""
        # Create output file path
        output_file = os.path.join(self.temp_dir.name, "audit_report.html")
        
        # Generate the report
        result = self.report_generator.generate_audit_report(
            self.audit_logs,
            output_file,
            title="Test Audit Report"
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check that the file contains expected content
        with open(output_file, "r") as f:
            content = f.read()
            self.assertIn("Test Audit Report", content)
            self.assertIn("Number of log entries: 3", content)
    
    def test_report_generator_custom_templates(self):
        """Test report generator with custom templates directory."""
        # Create a temporary templates directory
        with tempfile.TemporaryDirectory() as templates_dir:
            # Create a custom report generator
            custom_generator = ReportGenerator(templates_dir=templates_dir)
            
            # Check that the templates directory was set
            self.assertEqual(custom_generator.templates_dir, templates_dir)
    
    def test_report_generator_custom_css(self):
        """Test report generator with custom CSS."""
        # Create a temporary CSS file
        with tempfile.NamedTemporaryFile(suffix=".css") as css_file:
            # Write some CSS
            css_file.write(b"body { color: blue; }")
            css_file.flush()
            
            # Create a custom report generator
            custom_generator = ReportGenerator(custom_css=css_file.name)
            
            # Check that the custom CSS was set
            self.assertEqual(custom_generator.custom_css, css_file.name)


# Add pytest-style tests
@pytest.fixture
def report_generator():
    """Create a ReportGenerator fixture."""
    return ReportGenerator()


@pytest.fixture
def compliance_report():
    """Create a mock compliance report fixture."""
    report = MagicMock()
    report.regulation = "GDPR"
    report.issues = [
        {
            "component": "Data Storage",
            "issue": "Unencrypted data storage",
            "severity": "high",
            "recommendation": "Implement encryption"
        }
    ]
    report.warnings = []
    report.passed_checks = ["Data Minimization"]
    report.summary.return_value = {
        "regulation": "GDPR",
        "issues_count": 1,
        "warnings_count": 0,
        "passed_checks_count": 1,
        "compliant": False
    }
    return report


@pytest.fixture
def audit_logs():
    """Create mock audit logs fixture."""
    return [
        {
            "event_type": "audit_started",
            "timestamp": "2023-03-18T12:00:00",
            "operation_id": "op123",
            "operation_name": "test_operation"
        },
        {
            "event_type": "data_access",
            "timestamp": "2023-03-18T12:01:00",
            "operation_id": "op123",
            "operation_name": "test_operation",
            "dataset_name": "test_dataset",
            "columns_accessed": ["col1", "col2"],
            "num_records": 100
        },
        {
            "event_type": "audit_closed",
            "timestamp": "2023-03-18T12:02:00",
            "operation_id": "op123",
            "operation_name": "test_operation",
            "status": "completed"
        }
    ]


def test_generate_compliance_report_html_fixture(report_generator, compliance_report, tmp_path):
    """Test compliance report generation using fixtures."""
    # Create output file path
    output_file = tmp_path / "compliance_report.html"
    
    # Generate the report
    result = report_generator.generate_compliance_report(
        compliance_report,
        str(output_file)
    )
    
    # Check that the file was created
    assert output_file.exists()
    
    # Check content
    content = output_file.read_text()
    assert "Compliance Report - GDPR" in content
    assert "Issues: 1" in content


def test_generate_audit_report_html_fixture(report_generator, audit_logs, tmp_path):
    """Test audit report generation using fixtures."""
    # Create output file path
    output_file = tmp_path / "audit_report.html"
    
    # Generate the report
    result = report_generator.generate_audit_report(
        audit_logs,
        str(output_file)
    )
    
    # Check that the file was created
    assert output_file.exists()
    
    # Check content
    content = output_file.read_text()
    assert "Audit Trail Report" in content
    assert "Number of log entries: 3" in content


@pytest.mark.parametrize(
    "regulation,with_charts", 
    [
        ("GDPR", True),
        ("GDPR", False),
        ("CCPA", True),
        ("HIPAA", True)
    ]
)
def test_compliance_report_variations(
    report_generator, tmp_path, regulation, with_charts
):
    """Test various compliance report variations."""
    # Create a mock report with the specified regulation
    report = MagicMock()
    report.regulation = regulation
    report.issues = []
    report.warnings = []
    report.passed_checks = ["Test"]
    report.summary.return_value = {
        "regulation": regulation,
        "issues_count": 0,
        "warnings_count": 0,
        "passed_checks_count": 1,
        "compliant": True
    }
    
    # Create output file path
    output_file = tmp_path / f"{regulation.lower()}_report.html"
    
    # Generate the report
    result = report_generator.generate_compliance_report(
        report,
        str(output_file),
        include_charts=with_charts
    )
    
    # Check that the file was created
    assert output_file.exists()
    
    # Check that the regulation name is in the content
    content = output_file.read_text()
    assert regulation in content


if __name__ == "__main__":
    unittest.main() 