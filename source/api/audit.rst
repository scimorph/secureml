==============
Audit Trail API
==============

.. module:: secureml.audit

This module provides tools for creating and managing audit logs for ML operations, helping to document data processing and model decisions for compliance purposes.

AuditTrail Class
---------------

.. autoclass:: AuditTrail
   :members:
   :special-members: __init__

The `AuditTrail` class provides a comprehensive way to track operations in your machine learning pipeline. It records various events with timestamps and context information, creating an immutable record that can be used for compliance purposes.

Basic Usage Example:

.. code-block:: python

    from secureml.audit import AuditTrail
    
    # Create an audit trail for a model training operation
    audit = AuditTrail(
        operation_name="model_training",
        context={"model_version": "v1.0", "environment": "production"},
        regulations=["GDPR", "HIPAA"]
    )
    
    # Log events during your operation
    audit.log_data_access(
        dataset_name="patient_records",
        columns_accessed=["age", "diagnosis", "treatment"],
        num_records=1000,
        purpose="training disease prediction model",
        user="data_scientist_1"
    )
    
    # Close the audit trail when done
    audit.close()

Utility Functions
----------------

Audit Function Decorator
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: audit_function

The `audit_function` decorator provides a simple way to add auditing to any function:

.. code-block:: python

    from secureml.audit import audit_function
    
    @audit_function(regulations=["GDPR"])
    def train_model(data, params):
        # Function implementation
        return model

Log Retrieval
~~~~~~~~~~~~~

.. autofunction:: get_audit_logs

This function allows you to retrieve and analyze audit logs:

.. code-block:: python

    from secureml.audit import get_audit_logs
    
    # Get all logs for a specific operation
    logs = get_audit_logs(
        operation_name="model_training",
        start_time="2023-01-01T00:00:00",
        end_time="2023-01-31T23:59:59"
    )

Configuration
------------

The audit module uses these default configuration values:

.. code-block:: python

    DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_LOG_LEVEL = logging.INFO
    DEFAULT_LOG_DIR = "secureml_audit_logs"

You can override these by providing custom parameters when creating an `AuditTrail` instance.

Working with Regulations
-----------------------

The audit trail system is designed to support compliance with various regulations including:

- **GDPR**: General Data Protection Regulation
- **HIPAA**: Health Insurance Portability and Accountability Act
- **CCPA**: California Consumer Privacy Act

When initializing an `AuditTrail`, you can specify which regulations apply:

.. code-block:: python

    audit = AuditTrail(
        operation_name="credit_scoring",
        regulations=["GDPR", "CCPA"]
    )

    # This will be recorded in the audit logs for compliance reporting
    audit.log_compliance_check(
        check_type="data_access_permission",
        regulation="GDPR",
        result=True,
        details={"user_consent_obtained": True, "legal_basis": "legitimate_interest"}
    )

Best Practices
-------------

1. **Start early**: Begin auditing from the earliest stages of your ML project
2. **Be comprehensive**: Log all significant operations and decisions
3. **Include context**: Add relevant context to your audit logs
4. **Use consistent naming**: Maintain consistent operation names and event types
5. **Automate**: Use the `audit_function` decorator to automatically audit functions
6. **Regular review**: Periodically review audit logs for compliance 