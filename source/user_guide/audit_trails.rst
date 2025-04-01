=============
Audit Trails
=============

Audit trails provide a chronological record of all data operations and model activities, which is critical for compliance with privacy regulations and for ensuring accountability in machine learning systems. SecureML offers comprehensive audit trail capabilities to track all privacy-relevant operations throughout the ML lifecycle.

Core Concepts
------------

**Audit Events**: Discrete actions or operations captured in the audit trail, such as data access, model training, or prediction requests.

**Immutability**: Ensuring audit logs cannot be altered or tampered with after they are created.

**Granularity**: Different levels of detail in audit logs, from high-level system events to fine-grained data access patterns.

**Compliance Integration**: Connecting audit trails to specific compliance requirements and regulations.

Basic Usage
----------

Enabling Audit Trails
^^^^^^^^^^^^^^^^^^^

To enable audit trails for your SecureML application:

.. code-block:: python

    from secureml.audit import AuditManager
    
    # Initialize the audit manager
    audit_manager = AuditManager(
        app_name='credit_risk_model',
        log_level='INFO',  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
        storage_backend='file',  # Options: 'file', 'database', 'cloud'
        storage_path='audit_logs/'
    )
    
    # Enable audit trails
    audit_manager.enable()

Logging Data Access Events
^^^^^^^^^^^^^^^^^^^^^^^

Track when sensitive data is accessed:

.. code-block:: python

    # Manually log a data access event
    audit_manager.log_data_access(
        dataset_name='customer_financial_data',
        user_id='analyst_123',
        purpose='model_training',
        access_type='read',
        columns_accessed=['income', 'credit_score', 'loan_history'],
        num_records=5000,
        query="SELECT * FROM customer_data WHERE account_status = 'active'"
    )
    
    # Automatically log data access via context manager
    with audit_manager.track_data_access(
        dataset_name='customer_financial_data',
        purpose='feature_engineering'
    ):
        # Any data operations here will be automatically logged
        df = pd.read_csv('customer_data.csv')
        features = engineer_features(df)

Logging Model Operations
^^^^^^^^^^^^^^^^^^^^^

Track model-related activities:

.. code-block:: python

    # Log model training event
    model_training_id = audit_manager.log_model_training(
        model_name='credit_risk_classifier',
        model_type='random_forest',
        training_dataset='customer_data_anonymized',
        hyperparameters={'n_estimators': 100, 'max_depth': 10},
        training_metrics={'accuracy': 0.92, 'auc': 0.88},
        privacy_parameters={'epsilon': 1.0, 'delta': 1e-5}
    )
    
    # Log model prediction event
    audit_manager.log_model_prediction(
        model_name='credit_risk_classifier',
        model_version='v1.2',
        prediction_type='batch',
        num_predictions=250,
        user_id='service_account_risk_api',
        purpose='customer_risk_assessment'
    )
    
    # Log model export event
    audit_manager.log_model_export(
        model_name='credit_risk_classifier',
        model_version='v1.2',
        export_format='pickle',
        destination='risk_api_server',
        user_id='devops_123'
    )

Automatic Audit Integration
^^^^^^^^^^^^^^^^^^^^^^^^^

Use automatic audit integration with SecureML components:

.. code-block:: python

    from secureml.anonymization import Anonymizer
    from secureml.differential_privacy import DPTrainer
    
    # Anonymization with audit trails
    anonymizer = Anonymizer(
        quasi_identifiers=['age', 'zipcode', 'gender'],
        sensitive_attributes=['income', 'disease'],
        k=5,
        audit_manager=audit_manager  # Pass the audit manager to enable automatic logging
    )
    
    # Differential privacy with audit trails
    dp_trainer = DPTrainer(
        model=model,
        epsilon=1.0,
        delta=1e-5,
        audit_manager=audit_manager  # Enable automatic logging
    )

Advanced Techniques
------------------

Tamper-Proof Audit Logs
^^^^^^^^^^^^^^^^^^^^^

Ensure audit logs cannot be modified:

.. code-block:: python

    from secureml.audit import TamperProofAuditManager
    
    # Create a tamper-proof audit manager
    tamper_proof_manager = TamperProofAuditManager(
        app_name='healthcare_prediction',
        hash_algorithm='sha256',  # Hashing algorithm for the chain
        storage_backend='blockchain',  # Options: 'blockchain', 'signed_database', 'immutable_storage'
        key_management='vault'  # Options: 'vault', 'kms', 'local'
    )
    
    # Log an event with additional integrity verification
    tamper_proof_manager.log_event(
        event_type='data_processing',
        description='PHI data anonymization',
        data_assets=['patient_records.csv'],
        verify_integrity=True
    )
    
    # Verify the audit log integrity
    is_valid = tamper_proof_manager.verify_log_integrity()
    if not is_valid:
        print("Warning: Audit logs may have been tampered with!")

Compliance-Specific Audit Trails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure audit trails for specific regulations:

.. code-block:: python

    from secureml.audit import ComplianceAuditManager
    
    # Create a GDPR-focused audit manager
    gdpr_audit = ComplianceAuditManager(
        regulation='gdpr',
        data_controller='Example Healthcare Inc.',
        data_protection_officer='jane.doe@example.com',
        legal_basis_tracking=True,
        consent_tracking=True,
        right_to_access_support=True,
        right_to_be_forgotten_support=True
    )
    
    # Log a consent event
    gdpr_audit.log_consent(
        data_subject_id='patient_12345',
        consent_given=True,
        consent_timestamp='2023-04-15T14:30:00Z',
        consent_purpose=['treatment', 'research'],
        consent_expiry='2024-04-15T14:30:00Z',
        evidence_reference='consent_form_12345.pdf'
    )
    
    # Log a data subject request
    gdpr_audit.log_data_subject_request(
        request_type='access_request',  # Options: 'access_request', 'deletion_request', 'correction_request'
        data_subject_id='patient_12345',
        request_timestamp='2023-06-20T10:15:00Z',
        request_status='completed',
        request_completion_timestamp='2023-06-22T14:30:00Z',
        operator_id='privacy_team_member_123'
    )

Querying and Analyzing Audit Logs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Search and analyze audit data:

.. code-block:: python

    from secureml.audit import AuditLogAnalyzer
    
    # Initialize the analyzer
    analyzer = AuditLogAnalyzer(audit_manager)
    
    # Search for specific events
    data_access_events = analyzer.search(
        event_type='data_access',
        user_id='analyst_123',
        time_range=('2023-01-01', '2023-06-30'),
        dataset_name='customer_financial_data'
    )
    
    # Generate compliance reports
    gdpr_report = analyzer.generate_compliance_report(
        regulation='gdpr',
        time_period='last_quarter',
        report_format='pdf'
    )
    
    # Identify unusual access patterns
    anomalies = analyzer.detect_anomalies(
        baseline_period=('2023-01-01', '2023-03-31'),
        analysis_period=('2023-04-01', '2023-06-30'),
        sensitivity=0.8
    )
    
    # Save the analysis to file
    analyzer.export_analysis('audit_analysis_q2_2023.html')

Usage Analytics and Dashboards
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate insights from audit data:

.. code-block:: python

    from secureml.audit import AuditDashboard
    
    # Create a dashboard for visualization
    dashboard = AuditDashboard(audit_manager)
    
    # Add various metrics to track
    dashboard.add_metric('data_access_by_user', timeframe='daily')
    dashboard.add_metric('model_invocations', timeframe='hourly')
    dashboard.add_metric('privacy_budget_consumption', timeframe='weekly')
    dashboard.add_metric('data_subject_requests', timeframe='monthly')
    
    # Generate and publish the dashboard
    dashboard.generate()
    dashboard.publish('audit_dashboard.html')

    # Set up alerts for specific conditions
    dashboard.add_alert(
        metric='data_access_by_user',
        condition='count > 1000',
        notification_method='email',
        notification_recipient='security@example.com'
    )

Integration with System Components
--------------------------------

Integrating Audit Trails with Data Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Track all data transformations:

.. code-block:: python

    from secureml.audit import DataPipelineAuditor
    from secureml.data import DataPipeline
    
    # Create an auditor for data pipelines
    pipeline_auditor = DataPipelineAuditor(audit_manager)
    
    # Create a data pipeline with audit integration
    pipeline = DataPipeline(
        name='feature_engineering_pipeline',
        auditor=pipeline_auditor
    )
    
    # Add pipeline steps with automatic auditing
    pipeline.add_step(
        name='load_data',
        function=load_customer_data,
        audit_metadata={'data_source': 'customer_database', 'sensitivity': 'high'}
    )
    
    pipeline.add_step(
        name='anonymize_data',
        function=anonymize_customer_data,
        audit_metadata={'privacy_technique': 'k-anonymity', 'k_value': 5}
    )
    
    # Execute the pipeline with full audit trail
    pipeline.execute()

Integrating Audit Trails with Model Serving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Track model serving and predictions:

.. code-block:: python

    from secureml.audit import ModelServingAuditor
    from secureml.serving import ModelServer
    
    # Create an auditor for model serving
    serving_auditor = ModelServingAuditor(audit_manager)
    
    # Create a model server with audit integration
    model_server = ModelServer(
        model_path='models/credit_risk_classifier.pkl',
        auditor=serving_auditor,
        audit_predictions=True,  # Enable prediction auditing
        audit_sample_rate=0.1,   # Audit 10% of predictions (for high-volume systems)
        audit_data_retention=30  # Store audit data for 30 days
    )
    
    # Start the model server
    model_server.start(port=8000)

Retention and Archiving
---------------------

Managing Audit Data Lifecycle:

.. code-block:: python

    from secureml.audit import AuditRetentionManager
    
    # Create a retention manager
    retention_manager = AuditRetentionManager(audit_manager)
    
    # Set retention policies
    retention_manager.set_retention_policy(
        event_type='data_access',
        retention_period=365,  # days
        archive_after=90,      # days
        anonymize_after=180    # days
    )
    
    retention_manager.set_retention_policy(
        event_type='model_training',
        retention_period=730,  # 2 years
        archive_after=365      # 1 year
    )
    
    # Apply retention policies
    retention_manager.apply_policies()
    
    # Archive old audit logs
    retention_manager.archive_logs(
        start_date='2022-01-01',
        end_date='2022-12-31',
        archive_format='encrypted_zip',
        archive_location='s3://audit-archives/'
    )

Best Practices
-------------

1. **Start early**: Enable audit trails from the beginning of your project, not as an afterthought

2. **Be comprehensive**: Log all privacy-relevant operations, not just the obvious ones

3. **Use proper granularity**: Balance between logging too much (performance impact) and too little (missing important events)

4. **Secure audit logs**: Implement proper access controls and ensure logs cannot be tampered with

5. **Regular reviews**: Periodically review audit logs for anomalies or compliance issues

6. **Retention policies**: Define clear retention policies that align with legal and regulatory requirements

7. **Automation**: Automate the generation of compliance reports from audit data

8. **User attribution**: Always include user information when logging events to ensure accountability

9. **Purpose tracking**: Record the purpose for data access and processing to demonstrate compliance with purpose limitation principles

10. **Privacy by design**: Implement privacy-preserving audit logs that don't themselves become a privacy risk

Further Reading
-------------

* :doc:`/api/audit` - Complete API reference for audit trail functions
* :doc:`/examples/audit` - More examples of audit trail implementation
* :doc:`/compliance/audit_requirements` - Audit requirements for different regulations 