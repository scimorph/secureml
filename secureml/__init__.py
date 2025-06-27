"""
SecureML - A Python library for privacy-preserving machine learning.

This library provides tools for handling sensitive data in AI/ML workflows
while maintaining compliance with privacy regulations like GDPR, CCPA, HIPAA, and LGPD.
"""

__version__ = "0.3.1"

# Export core functions for easier imports
from secureml.anonymization import anonymize
from secureml.privacy import differentially_private_train
from secureml.compliance import check_compliance, ComplianceAuditor
from secureml.synthetic import generate_synthetic_data
from secureml.federated import (
    train_federated,
    start_federated_server,
    start_federated_client,
)
from secureml.presets import (
    list_available_presets,
    load_preset,
    get_preset_field,
)
# Import audit trail and reporting functionality
from secureml.audit import AuditTrail, audit_function, get_audit_logs
from secureml.reporting import ReportGenerator
# Import key management functionality directly from the .py file not the package
# to avoid circular imports
from secureml.key_management import (
    configure_default_key_manager,
    get_encryption_key,
    KeyManager,
)

# Import CLI entry point
from secureml.cli import cli

__all__ = [
    # Core functionality
    "anonymize",
    "differentially_private_train",
    "check_compliance",
    "generate_synthetic_data",
    "train_federated",
    "start_federated_server",
    "start_federated_client",
    "list_available_presets",
    "load_preset",
    "get_preset_field",
    # Audit and reporting
    "AuditTrail",
    "audit_function",
    "get_audit_logs",
    "ReportGenerator",
    "ComplianceAuditor",
    # Key management
    "configure_default_key_manager",
    "get_encryption_key",
    "KeyManager",
    # CLI entry point
    "cli",
]
