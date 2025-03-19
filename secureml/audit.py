"""
Audit trail functionality for SecureML.

This module provides tools for creating and managing audit logs for ML operations,
helping to document data processing and model decisions for compliance purposes.
"""

import logging
import os
import json
import datetime
from typing import Any, Dict, List, Optional, Union, Callable
import uuid
import inspect
import functools
import time

# Configure the default logger
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_DIR = "secureml_audit_logs"

# Create logger
logger = logging.getLogger("secureml")
logger.setLevel(DEFAULT_LOG_LEVEL)

# Create console handler if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(DEFAULT_LOG_LEVEL)
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class AuditTrail:
    """
    Class for managing audit trails in SecureML operations.
    
    The AuditTrail class provides methods for logging operations on datasets and models,
    making it easier to track data transformations and model decisions for compliance purposes.
    """
    
    def __init__(
        self, 
        operation_name: str, 
        log_dir: Optional[str] = None,
        log_level: int = DEFAULT_LOG_LEVEL,
        context: Optional[Dict[str, Any]] = None,
        regulations: Optional[List[str]] = None
    ):
        """
        Initialize an audit trail for an operation.
        
        Args:
            operation_name: Name of the operation being audited
            log_dir: Directory to store log files (default: secureml_audit_logs)
            log_level: Logging level to use
            context: Additional context information to include in all logs
            regulations: List of regulations this audit trail is tracking compliance with
        """
        self.operation_name = operation_name
        self.operation_id = str(uuid.uuid4())
        self.start_time = datetime.datetime.now().isoformat()
        self.context = context or {}
        self.regulations = regulations or []
        
        # Setup logging to file
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.setup_file_logging()
        
        # Log the initialization
        self.log_event(
            "audit_started", 
            {
                "operation_name": operation_name,
                "operation_id": self.operation_id,
                "start_time": self.start_time,
                "regulations": self.regulations
            }
        )
    
    def setup_file_logging(self) -> None:
        """Setup file logging for the audit trail."""
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create a log file for this operation
        log_file = os.path.join(
            self.log_dir, 
            f"{self.operation_name}_{self.operation_id}.log"
        )
        
        # Create a file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(DEFAULT_LOG_LEVEL)
        
        # Create a formatter
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        audit_logger = logging.getLogger(f"secureml.audit.{self.operation_id}")
        audit_logger.setLevel(DEFAULT_LOG_LEVEL)
        audit_logger.addHandler(file_handler)
        
        self.logger = audit_logger
    
    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log an event to the audit trail.
        
        Args:
            event_type: Type of event being logged
            details: Details about the event
        """
        # Combine with context
        log_data = {
            "event_type": event_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            **details,
            **self.context
        }
        
        # Log the event
        self.logger.info(json.dumps(log_data))
    
    def log_data_access(
        self, 
        dataset_name: str, 
        columns_accessed: List[str],
        num_records: int,
        purpose: str,
        user: Optional[str] = None
    ) -> None:
        """
        Log access to a dataset.
        
        Args:
            dataset_name: Name of the dataset being accessed
            columns_accessed: List of columns accessed
            num_records: Number of records accessed
            purpose: Purpose of the access
            user: User who performed the access
        """
        self.log_event(
            "data_access",
            {
                "dataset_name": dataset_name,
                "columns_accessed": columns_accessed,
                "num_records": num_records,
                "purpose": purpose,
                "user": user
            }
        )
    
    def log_data_transformation(
        self,
        transformation_type: str,
        input_data: str,
        output_data: str,
        parameters: Dict[str, Any]
    ) -> None:
        """
        Log a data transformation.
        
        Args:
            transformation_type: Type of transformation (e.g., anonymization, encryption)
            input_data: Description of input data
            output_data: Description of output data
            parameters: Parameters used for the transformation
        """
        self.log_event(
            "data_transformation",
            {
                "transformation_type": transformation_type,
                "input_data": input_data,
                "output_data": output_data,
                "parameters": parameters
            }
        )
    
    def log_model_training(
        self,
        model_type: str,
        dataset_name: str,
        parameters: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        privacy_parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log model training.
        
        Args:
            model_type: Type of model being trained
            dataset_name: Name of the dataset used for training
            parameters: Training parameters
            metrics: Training metrics
            privacy_parameters: Privacy parameters used (e.g., epsilon for DP)
        """
        self.log_event(
            "model_training",
            {
                "model_type": model_type,
                "dataset_name": dataset_name,
                "parameters": parameters,
                "metrics": metrics or {},
                "privacy_parameters": privacy_parameters or {}
            }
        )
    
    def log_model_inference(
        self,
        model_id: str,
        input_data: str,
        output: Any,
        confidence: Optional[float] = None
    ) -> None:
        """
        Log model inference.
        
        Args:
            model_id: Identifier for the model
            input_data: Description of input data
            output: Model output
            confidence: Confidence score for the output
        """
        self.log_event(
            "model_inference",
            {
                "model_id": model_id,
                "input_data": input_data,
                "output": str(output),
                "confidence": confidence
            }
        )
    
    def log_compliance_check(
        self,
        check_type: str,
        regulation: str,
        result: bool,
        details: Dict[str, Any]
    ) -> None:
        """
        Log a compliance check.
        
        Args:
            check_type: Type of compliance check
            regulation: Regulation being checked
            result: Result of the check (True=passed, False=failed)
            details: Details about the check
        """
        self.log_event(
            "compliance_check",
            {
                "check_type": check_type,
                "regulation": regulation,
                "result": result,
                "details": details
            }
        )
    
    def log_user_request(
        self,
        request_type: str,
        user_id: str,
        details: Dict[str, Any],
        status: str
    ) -> None:
        """
        Log a user request (e.g., GDPR right to access).
        
        Args:
            request_type: Type of request
            user_id: ID of the user making the request
            details: Details about the request
            status: Status of the request
        """
        self.log_event(
            "user_request",
            {
                "request_type": request_type,
                "user_id": user_id,
                "details": details,
                "status": status
            }
        )
    
    def log_error(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error.
        
        Args:
            error_type: Type of error
            message: Error message
            details: Additional details about the error
        """
        self.log_event(
            "error",
            {
                "error_type": error_type,
                "message": message,
                "details": details or {}
            }
        )
    
    def close(self, status: str = "completed", details: Optional[Dict[str, Any]] = None) -> None:
        """
        Close the audit trail.
        
        Args:
            status: Final status of the operation
            details: Additional details about the operation's completion
        """
        end_time = datetime.datetime.now().isoformat()
        self.log_event(
            "audit_closed",
            {
                "status": status,
                "end_time": end_time,
                "details": details or {}
            }
        )
        
        # Remove the file handler from the logger
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()


def audit_function(
    operation_name: Optional[str] = None,
    log_dir: Optional[str] = None,
    regulations: Optional[List[str]] = None
) -> Callable:
    """
    Decorator for auditing function calls.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        log_dir: Directory to store audit logs
        regulations: List of regulations this function should comply with
    
    Returns:
        Decorated function with audit trail
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create an audit trail
            op_name = operation_name or func.__name__
            audit = AuditTrail(op_name, log_dir=log_dir, regulations=regulations)
            
            # Log function call
            audit.log_event(
                "function_call",
                {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
            )
            
            start_time = time.time()
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Log success
                execution_time = time.time() - start_time
                audit.log_event(
                    "function_return",
                    {
                        "status": "success",
                        "execution_time": execution_time,
                        "result": str(result)
                    }
                )
                
                # Close the audit trail
                audit.close("completed")
                
                return result
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time
                audit.log_error(
                    error_type=type(e).__name__,
                    message=str(e),
                    details={"execution_time": execution_time}
                )
                
                # Close the audit trail
                audit.close("error")
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator


def get_audit_logs(
    operation_id: Optional[str] = None,
    operation_name: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    log_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve audit logs for analysis.
    
    Args:
        operation_id: ID of the operation to retrieve logs for
        operation_name: Name of the operation to retrieve logs for
        start_time: Start time for logs (ISO format)
        end_time: End time for logs (ISO format)
        log_dir: Directory containing audit logs
    
    Returns:
        List of audit log entries matching the criteria
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    
    if not os.path.exists(log_dir):
        return []
    
    logs = []
    
    # Walk through log files
    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue
        
        # Check if operation_id matches
        if operation_id and operation_id not in filename:
            continue
        
        # Check if operation_name matches
        if operation_name and not filename.startswith(f"{operation_name}_"):
            continue
        
        # Read the log file
        with open(os.path.join(log_dir, filename), "r") as f:
            for line in f:
                try:
                    log_entry = json.loads(line.split(" - ")[-1])
                    
                    # Check if timestamp is in range
                    if start_time and log_entry.get("timestamp", "") < start_time:
                        continue
                    if end_time and log_entry.get("timestamp", "") > end_time:
                        continue
                    
                    logs.append(log_entry)
                except (json.JSONDecodeError, IndexError):
                    # Skip malformed log entries
                    continue
    
    return logs 