=============
Audit Trails
=============

Audit trails provide a chronological record of all data operations and model activities, which is critical for compliance with privacy regulations and for ensuring accountability in machine learning systems. SecureML offers audit trail capabilities to track all privacy-relevant operations throughout the ML lifecycle.

Core Concepts
------------

**Audit Events**: Discrete actions or operations captured in the audit trail, such as data access, model training, or prediction requests.

**Immutability**: Ensuring audit logs cannot be altered or tampered with after they are created.

**Granularity**: Different levels of detail in audit logs, from high-level system events to fine-grained data access patterns.

**Compliance Integration**: Connecting audit trails to specific compliance requirements and regulations.

Basic Usage
----------

Creating an Audit Trail
^^^^^^^^^^^^^^^^^^^

To create an audit trail for your SecureML application:

.. code-block:: python

    from secureml.audit import AuditTrail
    
    # Initialize an audit trail
    audit = AuditTrail(
        operation_name='credit_risk_model_training',
        log_dir='audit_logs/',  # Optional: directory for storing logs
        log_level=20,  # Optional: logging level (default: INFO)
        context={'app_version': '1.0.0'},  # Optional: context to include in all logs
        regulations=['GDPR', 'CCPA']  # Optional: regulations this operation should comply with
    )
    
    # The audit trail will be automatically initialized with a unique operation_id
    # and start_time, which are included in all subsequent logs

Logging Data Access Events
^^^^^^^^^^^^^^^^^^^^^^^

Track when sensitive data is accessed:

.. code-block:: python

    # Log a data access event
    audit.log_data_access(
        dataset_name='customer_financial_data',
        columns_accessed=['income', 'credit_score', 'loan_history'],
        num_records=5000,
        purpose='model_training',
        user='analyst_123'  # Optional: user who performed the access
    )

Logging Data Transformations
^^^^^^^^^^^^^^^^^^^^^

Track data transformations:

.. code-block:: python

    # Log a data transformation event
    audit.log_data_transformation(
        transformation_type='anonymization',
        input_data='raw_customer_data',
        output_data='anonymized_customer_data',
        parameters={
            'method': 'k-anonymity',
            'k': 5,
            'quasi_identifiers': ['age', 'zipcode', 'gender']
        }
    )

Logging Model Operations
^^^^^^^^^^^^^^^^^^^^^

Track model-related activities:

.. code-block:: python

    # Log model training event
    audit.log_model_training(
        model_type='random_forest',
        dataset_name='customer_data_anonymized',
        parameters={'n_estimators': 100, 'max_depth': 10},
        metrics={'accuracy': 0.92, 'auc': 0.88},
        privacy_parameters={'epsilon': 1.0, 'delta': 1e-5}
    )
    
    # Log model inference event
    audit.log_model_inference(
        model_id='credit_risk_classifier_v1',
        input_data='customer_application_123',
        output='high_risk',
        confidence=0.85
    )

Logging Compliance Checks
^^^^^^^^^^^^^^^^^^^^^

Track compliance verification:

.. code-block:: python

    # Log a compliance check
    audit.log_compliance_check(
        check_type='data_minimization',
        regulation='GDPR',
        result=True,  # True = passed, False = failed
        details={
            'columns_before': 25,
            'columns_after': 10,
            'columns_removed': ['unnecessary_field_1', 'unnecessary_field_2']
        }
    )

Logging User Requests
^^^^^^^^^^^^^^^^^^^^^

Track GDPR/CCPA user requests:

.. code-block:: python

    # Log a user request (e.g., GDPR right to access)
    audit.log_user_request(
        request_type='data_access_request',
        user_id='user_12345',
        details={
            'request_date': '2023-06-20',
            'data_categories': ['personal_info', 'financial_data']
        },
        status='completed'
    )

Closing the Audit Trail
^^^^^^^^^^^^^^^^^^^^^

Properly close the audit trail when the operation is complete:

.. code-block:: python

    # Close the audit trail
    audit.close(
        status='completed',  # Or 'error', 'cancelled', etc.
        details={
            'execution_time': 125.7,
            'output_location': 'models/credit_risk_v1.pkl'
        }
    )

Advanced Techniques
------------------

Using the Audit Function Decorator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Automatically audit function calls:

.. code-block:: python

    from secureml.audit import audit_function
    
    # Create a decorated function
    @audit_function(
        operation_name='data_preprocessing',
        log_dir='audit_logs',
        regulations=['GDPR']
    )
    def process_sensitive_data(data, anonymize=True):
        # Function implementation...
        return processed_data
    
    # When this function is called, the audit trail will automatically:
    # 1. Log the function call with parameters
    # 2. Log the return value or any exceptions
    # 3. Close the audit trail

Retrieving Audit Logs
^^^^^^^^^^^^^^^^^^^^^

Retrieve and analyze audit logs:

.. code-block:: python

    from secureml.audit import get_audit_logs
    
    # Get logs for a specific operation
    logs = get_audit_logs(
        operation_id='12345-abcde-67890',  # Optional: specific operation ID
        operation_name='credit_risk_model_training',  # Optional: operation name
        start_time='2023-01-01T00:00:00',  # Optional: filter by start time
        end_time='2023-06-30T23:59:59',  # Optional: filter by end time
        log_dir='audit_logs'  # Optional: directory containing logs
    )
    
    # Analyze the logs
    for log in logs:
        print(f"Event: {log['event_type']} - Time: {log['timestamp']}")

Integration with Reporting
-------------------------

Using the ReportGenerator
^^^^^^^^^^^^^^^^^^^^^

Generate HTML or PDF reports from audit logs:

.. code-block:: python

    from secureml.reporting import ReportGenerator
    
    # Create a report generator
    generator = ReportGenerator()
    
    # Generate an audit report
    report_path = generator.generate_audit_report(
        logs=logs,  # Logs retrieved with get_audit_logs
        output_file='audit_report.pdf',
        title='Credit Risk Model Audit Report',
        logo_path='company_logo.png',  # Optional
        include_charts=True  # Optional: include visualizations
    )
    
    print(f"Audit report generated at: {report_path}")

Integration with Compliance Checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Audit trails can be automatically created when performing compliance checks:

.. code-block:: python

    from secureml.compliance import ComplianceAuditor
    
    # Create a compliance auditor with audit integration
    auditor = ComplianceAuditor(
        regulation='GDPR',
        log_dir='audit_logs'  # This enables automatic audit trail creation
    )
    
    # The audit trails for all operations will be stored in the log directory
    dataset_report = auditor.audit_dataset(
        dataset=df,
        dataset_name='patient_records'
    )

Best Practices
-------------

1. **Start early**: Enable audit trails from the beginning of your project, not as an afterthought

2. **Be comprehensive**: Log all privacy-relevant operations, not just the obvious ones

3. **Use proper granularity**: Balance between logging too much (performance impact) and too little (missing important events)

4. **Secure audit logs**: Implement proper access controls for log files

5. **Regular reviews**: Periodically review audit logs for anomalies or compliance issues

6. **Contextual information**: Include sufficient context in each log entry to understand the operation's purpose

7. **Automation**: Use the audit_function decorator for critical operations

8. **User attribution**: Always include user information when logging events to ensure accountability

9. **Purpose tracking**: Record the purpose for data access and processing to demonstrate compliance with purpose limitation principles

10. **Privacy by design**: Implement privacy-preserving audit logs that don't themselves become a privacy risk

Further Reading
-------------

* :doc:`/api/audit` - Complete API reference for audit trail functions
* :doc:`/examples/audit` - More examples of audit trail implementation
* :doc:`/compliance/audit_requirements` - Audit requirements for different regulations 