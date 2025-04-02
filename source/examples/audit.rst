Audit Trail Examples
==================

This section demonstrates how to use SecureML's audit trail functionality to track operations and ensure compliance with privacy regulations.

Basic Audit Trail Creation
-------------------------

Create and use a basic audit trail to track operations:

.. code-block:: python

    from secureml.audit import AuditTrail
    
    # Create an audit trail
    audit = AuditTrail(
        operation_name="data_preprocessing_example",
        log_dir="audit_logs",
        context={"project": "credit_scoring", "environment": "development"},
        regulations=["GDPR"]
    )
    
    # Load sample data
    data = pd.DataFrame({
        'name': ['Alice Smith', 'Bob Johnson'],
        'age': [32, 45],
        'income': [65000, 85000],
        'email': ['alice.s@example.com', 'bob.j@example.com']
    })
    
    # Log data access
    audit.log_data_access(
        dataset_name="customer_data",
        columns_accessed=list(data.columns),
        num_records=len(data),
        purpose="data_preparation",
        user="analyst_123"
    )
    
    # Log a data transformation
    audit.log_data_transformation(
        transformation_type="anonymization",
        input_data="raw_customer_data",
        output_data="anonymized_customer_data",
        parameters={
            "method": "k-anonymity",
            "k": 3,
            "quasi_identifiers": ["age", "income"]
        }
    )
    
    # Close the audit trail when operations are complete
    audit.close(
        status="completed",
        details={
            "execution_time": 2.5,
            "records_processed": len(data)
        }
    )

Using the Audit Function Decorator
---------------------------------

Automatically audit function calls with a decorator:

.. code-block:: python

    from secureml.audit import audit_function
    
    @audit_function(
        operation_name="model_training_example",
        log_dir="audit_logs",
        regulations=["GDPR", "CCPA"]
    )
    def train_model(data, model_type="random_forest", **params):
        """Train a machine learning model with audit logging."""
        print(f"Training {model_type} model with {len(data)} records")
        
        # Simulate model training
        accuracy = 0.92
        training_time = 35.7
        
        return {
            "model": f"{model_type}_model",
            "accuracy": accuracy,
            "training_time": training_time
        }
    
    # Call the decorated function - audit trail is created automatically
    result = train_model(
        data,
        model_type="gradient_boosting",
        n_estimators=100,
        max_depth=5
    )

Retrieving and Analyzing Audit Logs
----------------------------------

Retrieve and analyze audit logs for specific operations:

.. code-block:: python

    from secureml.audit import get_audit_logs
    
    # Get logs for a specific operation
    preprocessing_logs = get_audit_logs(
        operation_name="data_preprocessing_example",
        log_dir="audit_logs"
    )
    
    # Print summary of logs
    print(f"Retrieved {len(preprocessing_logs)} logs for data preprocessing")
    for log in preprocessing_logs:
        print(f"Event: {log.get('event_type')} - Time: {log.get('timestamp')}")
    
    # Get logs for a time period
    import datetime
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    all_logs = get_audit_logs(
        start_time=f"{today}T00:00:00",
        log_dir="audit_logs"
    )

Generating Reports from Audit Logs
---------------------------------

Generate HTML or PDF reports from audit trails:

.. code-block:: python

    from secureml.reporting import ReportGenerator
    
    # Create a report generator
    generator = ReportGenerator()
    
    # Generate an audit report
    report_path = generator.generate_audit_report(
        logs=all_logs,
        output_file="audit_report.html",
        title="SecureML Operations Audit Report",
        include_charts=True
    )
    
    print(f"Audit report generated at: {report_path}")

Integration with Other Features
-----------------------------

Audit trails integrate with other SecureML features like compliance checking:

.. code-block:: python

    from secureml.compliance import ComplianceAuditor
    
    # Create a compliance auditor with audit integration
    auditor = ComplianceAuditor(
        regulation='GDPR',
        log_dir='audit_logs'  # This enables automatic audit trail creation
    )
    
    # The audit trails for all operations will be stored in the log directory
    dataset_report = auditor.audit_dataset(
        dataset=data,
        dataset_name='customer_records'
    ) 