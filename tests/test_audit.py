"""
Tests for the audit module of SecureML.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from secureml.audit import (
    AuditTrail,
    audit_function,
    get_audit_logs,
    DEFAULT_LOG_DIR
)


class TestAuditTrail(unittest.TestCase):
    """Test cases for the AuditTrail class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for logs
        self.test_log_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_log_dir)
    
    def test_audit_trail_initialization(self):
        """Test that an AuditTrail can be initialized properly."""
        audit = AuditTrail("test_operation", log_dir=self.test_log_dir)
        
        self.assertEqual(audit.operation_name, "test_operation")
        self.assertIsNotNone(audit.operation_id)
        self.assertIsNotNone(audit.start_time)
        self.assertEqual(audit.log_dir, self.test_log_dir)
        
        # Check that a log file was created
        log_files = os.listdir(self.test_log_dir)
        self.assertEqual(len(log_files), 1)
        self.assertTrue(log_files[0].startswith("test_operation_"))
        self.assertTrue(log_files[0].endswith(".log"))
    
    def test_log_event(self):
        """Test logging an event to the audit trail."""
        audit = AuditTrail("test_operation", log_dir=self.test_log_dir)
        
        # Log an event
        audit.log_event("test_event", {"test_key": "test_value"})
        
        # Read the log file
        log_file = os.path.join(self.test_log_dir, os.listdir(self.test_log_dir)[0])
        with open(log_file, "r") as f:
            log_content = f.read()
        
        # Check that the event was logged
        self.assertIn("test_event", log_content)
        self.assertIn("test_key", log_content)
        self.assertIn("test_value", log_content)
    
    def test_log_data_access(self):
        """Test logging data access to the audit trail."""
        audit = AuditTrail("test_operation", log_dir=self.test_log_dir)
        
        # Log data access
        audit.log_data_access(
            dataset_name="test_dataset",
            columns_accessed=["col1", "col2"],
            num_records=100,
            purpose="testing",
            user="test_user"
        )
        
        # Read the log file
        log_file = os.path.join(self.test_log_dir, os.listdir(self.test_log_dir)[0])
        with open(log_file, "r") as f:
            log_content = f.read()
            log_entries = [json.loads(line.split(" - ")[-1]) for line in log_content.splitlines() if " - " in line]
        
        # Find the data_access event
        data_access_events = [entry for entry in log_entries if entry.get("event_type") == "data_access"]
        
        # Check that the event was logged correctly
        self.assertEqual(len(data_access_events), 1)
        event = data_access_events[0]
        self.assertEqual(event["dataset_name"], "test_dataset")
        self.assertEqual(event["columns_accessed"], ["col1", "col2"])
        self.assertEqual(event["num_records"], 100)
        self.assertEqual(event["purpose"], "testing")
        self.assertEqual(event["user"], "test_user")
    
    def test_close(self):
        """Test closing an audit trail."""
        audit = AuditTrail("test_operation", log_dir=self.test_log_dir)
        
        # Close the audit trail
        audit.close("success", {"detail": "test completed"})
        
        # Read the log file
        log_file = os.path.join(self.test_log_dir, os.listdir(self.test_log_dir)[0])
        with open(log_file, "r") as f:
            log_content = f.read()
            log_entries = [json.loads(line.split(" - ")[-1]) for line in log_content.splitlines() if " - " in line]
        
        # Find the audit_closed event
        closed_events = [entry for entry in log_entries if entry.get("event_type") == "audit_closed"]
        
        # Check that the event was logged correctly
        self.assertEqual(len(closed_events), 1)
        event = closed_events[0]
        self.assertEqual(event["status"], "success")
        self.assertEqual(event["details"], {"detail": "test completed"})


class TestAuditDecorator(unittest.TestCase):
    """Test cases for the audit_function decorator."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for logs
        self.test_log_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_log_dir)
    
    def test_audit_function_decorator_success(self):
        """Test the audit_function decorator on a successful function."""
        # Define a test function with the decorator
        @audit_function(log_dir=self.test_log_dir)
        def test_function(a, b):
            return a + b
        
        # Call the function
        result = test_function(2, 3)
        
        # Check the result
        self.assertEqual(result, 5)
        
        # Check that logs were created
        log_files = os.listdir(self.test_log_dir)
        self.assertEqual(len(log_files), 1)
        
        # Read the log file
        log_file = os.path.join(self.test_log_dir, log_files[0])
        with open(log_file, "r") as f:
            log_content = f.read()
            log_entries = [json.loads(line.split(" - ")[-1]) for line in log_content.splitlines() if " - " in line]
        
        # Check that function call and return were logged
        function_call_events = [entry for entry in log_entries if entry.get("event_type") == "function_call"]
        function_return_events = [entry for entry in log_entries if entry.get("event_type") == "function_return"]
        
        self.assertEqual(len(function_call_events), 1)
        self.assertEqual(len(function_return_events), 1)
        self.assertEqual(function_return_events[0]["status"], "success")
    
    def test_audit_function_decorator_error(self):
        """Test the audit_function decorator on a function that raises an error."""
        # Define a test function with the decorator
        @audit_function(log_dir=self.test_log_dir)
        def test_function_error():
            raise ValueError("Test error")
        
        # Call the function and expect an error
        with self.assertRaises(ValueError):
            test_function_error()
        
        # Check that logs were created
        log_files = os.listdir(self.test_log_dir)
        self.assertEqual(len(log_files), 1)
        
        # Read the log file
        log_file = os.path.join(self.test_log_dir, log_files[0])
        with open(log_file, "r") as f:
            log_content = f.read()
            log_entries = [json.loads(line.split(" - ")[-1]) for line in log_content.splitlines() if " - " in line]
        
        # Check that error was logged
        error_events = [entry for entry in log_entries if entry.get("event_type") == "error"]
        audit_closed_events = [entry for entry in log_entries if entry.get("event_type") == "audit_closed"]
        
        self.assertEqual(len(error_events), 1)
        self.assertEqual(error_events[0]["error_type"], "ValueError")
        self.assertEqual(error_events[0]["message"], "Test error")
        
        self.assertEqual(len(audit_closed_events), 1)
        self.assertEqual(audit_closed_events[0]["status"], "error")


class TestGetAuditLogs(unittest.TestCase):
    """Test cases for the get_audit_logs function."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for logs
        self.test_log_dir = tempfile.mkdtemp()
        
        # Create some test log files
        self.create_test_log_files()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_log_dir)
    
    def create_test_log_files(self):
        """Create test log files for different operations."""
        # Operation 1 log
        op1_id = "op1_123456"
        with open(os.path.join(self.test_log_dir, f"operation1_{op1_id}.log"), "w") as f:
            f.write('2023-03-18 12:00:00,000 - secureml.audit.op1_123456 - INFO - {"event_type": "audit_started", "timestamp": "2023-03-18T12:00:00", "operation_id": "op1_123456", "operation_name": "operation1"}\n')
            f.write('2023-03-18 12:00:01,000 - secureml.audit.op1_123456 - INFO - {"event_type": "test_event", "timestamp": "2023-03-18T12:00:01", "operation_id": "op1_123456", "operation_name": "operation1", "test_key": "test_value"}\n')
            f.write('2023-03-18 12:00:02,000 - secureml.audit.op1_123456 - INFO - {"event_type": "audit_closed", "timestamp": "2023-03-18T12:00:02", "operation_id": "op1_123456", "operation_name": "operation1", "status": "completed"}\n')
        
        # Operation 2 log
        op2_id = "op2_789012"
        with open(os.path.join(self.test_log_dir, f"operation2_{op2_id}.log"), "w") as f:
            f.write('2023-03-18 13:00:00,000 - secureml.audit.op2_789012 - INFO - {"event_type": "audit_started", "timestamp": "2023-03-18T13:00:00", "operation_id": "op2_789012", "operation_name": "operation2"}\n')
            f.write('2023-03-18 13:00:01,000 - secureml.audit.op2_789012 - INFO - {"event_type": "data_access", "timestamp": "2023-03-18T13:00:01", "operation_id": "op2_789012", "operation_name": "operation2", "dataset_name": "test_dataset"}\n')
            f.write('2023-03-18 13:00:02,000 - secureml.audit.op2_789012 - INFO - {"event_type": "audit_closed", "timestamp": "2023-03-18T13:00:02", "operation_id": "op2_789012", "operation_name": "operation2", "status": "completed"}\n')
    
    def test_get_audit_logs_by_operation_id(self):
        """Test retrieving audit logs by operation ID."""
        logs = get_audit_logs(operation_id="op1_123456", log_dir=self.test_log_dir)
        
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]["operation_id"], "op1_123456")
        self.assertEqual(logs[0]["operation_name"], "operation1")
    
    def test_get_audit_logs_by_operation_name(self):
        """Test retrieving audit logs by operation name."""
        logs = get_audit_logs(operation_name="operation2", log_dir=self.test_log_dir)
        
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]["operation_name"], "operation2")
    
    def test_get_audit_logs_by_time_range(self):
        """Test retrieving audit logs by time range."""
        logs = get_audit_logs(
            start_time="2023-03-18T12:30:00", 
            end_time="2023-03-18T13:30:00",
            log_dir=self.test_log_dir
        )
        
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]["operation_name"], "operation2")


# Add pytest-style tests
@pytest.fixture
def audit_trail(temp_log_dir):
    """Create an AuditTrail fixture."""
    audit = AuditTrail("pytest_operation", log_dir=temp_log_dir)
    yield audit
    audit.close()


def test_audit_trail_log_model_training(audit_trail):
    """Test logging model training."""
    audit_trail.log_model_training(
        model_type="random_forest",
        dataset_name="test_dataset",
        parameters={"n_estimators": 100, "max_depth": 10},
        metrics={"accuracy": 0.95, "f1": 0.94},
        privacy_parameters={"epsilon": 1.0, "delta": 1e-5}
    )
    
    # Read the log file
    log_files = os.listdir(audit_trail.log_dir)
    log_file = os.path.join(audit_trail.log_dir, log_files[0])
    
    with open(log_file, "r") as f:
        log_content = f.read()
    
    # Check that the model training was logged
    assert "model_training" in log_content
    assert "random_forest" in log_content
    assert "accuracy" in log_content
    assert "epsilon" in log_content


def test_audit_trail_log_compliance_check(audit_trail):
    """Test logging compliance checks."""
    audit_trail.log_compliance_check(
        check_type="data_minimization",
        regulation="GDPR",
        result=True,
        details={"column_count": 5, "required_columns": 5}
    )
    
    # Read the log file
    log_files = os.listdir(audit_trail.log_dir)
    log_file = os.path.join(audit_trail.log_dir, log_files[0])
    
    with open(log_file, "r") as f:
        log_content = f.read()
    
    # Check that the compliance check was logged
    assert "compliance_check" in log_content
    assert "data_minimization" in log_content
    assert "GDPR" in log_content
    assert "column_count" in log_content


@pytest.mark.parametrize(
    "log_method,args,expected_text", 
    [
        (
            "log_data_transformation",
            {
                "transformation_type": "anonymization",
                "input_data": "raw_data",
                "output_data": "anonymized_data",
                "parameters": {"method": "k-anonymity", "k": 5}
            },
            "anonymization"
        ),
        (
            "log_error",
            {
                "error_type": "ValidationError",
                "message": "Invalid data format",
                "details": {"field": "email", "issue": "wrong format"}
            },
            "ValidationError"
        ),
        (
            "log_user_request",
            {
                "request_type": "access_request",
                "user_id": "user123",
                "details": {"requested_data": "all_personal_data"},
                "status": "completed"
            },
            "access_request"
        )
    ]
)
def test_audit_trail_log_methods(audit_trail, log_method, args, expected_text):
    """Test various log methods with parameterized inputs."""
    # Call the log method with the specified arguments
    getattr(audit_trail, log_method)(**args)
    
    # Read the log file
    log_files = os.listdir(audit_trail.log_dir)
    log_file = os.path.join(audit_trail.log_dir, log_files[0])
    
    with open(log_file, "r") as f:
        log_content = f.read()
    
    # Check that the expected text is in the log
    assert expected_text in log_content


if __name__ == "__main__":
    unittest.main() 