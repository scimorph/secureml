Compliance Checking Examples
===========================

This section demonstrates how to use SecureML's compliance checking features to verify
that your ML pipelines comply with privacy regulations like GDPR, CCPA, or HIPAA.

Basic Compliance Check
---------------------

The simplest way to check compliance is using the `check_compliance` function:

.. code-block:: python

    import pandas as pd
    from secureml.compliance import check_compliance

    # Sample data with metadata
    data_list = [
        {
            'id': 1, 
            'name': 'Alice Smith', 
            'age': 30, 
            'zipcode': '12345', 
            'diagnosis': 'Flu', 
            'email': 'alice.s@example.com', 
            'income': 60000, 
            'phone': '555-1234',
            'consent_date': '2024-01-01',
            'data_storage_location': 'EU'
        }
    ]
    df = pd.DataFrame(data_list)

    # Basic GDPR compliance check
    report = check_compliance(
        data=df,
        regulation="GDPR",
        max_samples=100
    )
    print(report)

Using ComplianceAuditor
----------------------

The `ComplianceAuditor` class provides a more comprehensive way to audit your ML pipeline:

.. code-block:: python

    from secureml.compliance import ComplianceAuditor

    # Initialize auditor for GDPR
    auditor = ComplianceAuditor(
        regulation="GDPR",
        log_dir="audit_logs"  # Optional: store audit logs
    )

Dataset Audit
~~~~~~~~~~~~

Audit a dataset for compliance:

.. code-block:: python

    # Audit a dataset with metadata
    dataset_report = auditor.audit_dataset(
        dataset=df,
        dataset_name="patient_records",
        metadata={
            "description": "Patient medical records",
            "data_owner": "Hospital A",
            "data_retention_period": "5 years",
            "data_encrypted": True
        }
    )
    print(dataset_report)

Model Audit
~~~~~~~~~~

Audit a model configuration for compliance:

.. code-block:: python

    # Model configuration with compliance features
    model_config = {
        "model_type": "RandomForestClassifier",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 5
        },
        "supports_forget_request": True,
        "supports_deletion_request": True,
        "data_processing_purpose": "Medical diagnosis prediction",
        "model_storage_location": "EU"
    }

    # Audit the model
    model_report = auditor.audit_model(
        model_config=model_config,
        model_name="diagnosis_predictor",
        model_documentation={
            "version": "1.0",
            "training_date": "2024-01-01",
            "training_data_description": "Patient records from 2023",
            "model_accuracy": 0.85
        }
    )
    print(model_report)

Full Pipeline Audit
~~~~~~~~~~~~~~~~~

Audit an entire ML pipeline including preprocessing steps:

.. code-block:: python

    # Define preprocessing steps
    preprocessing_steps = [
        {
            "name": "data_cleaning",
            "type": "anonymization",
            "input": "raw_data",
            "output": "anonymized_data",
            "parameters": {
                "method": "k-anonymity",
                "k": 2,
                "sensitive_columns": ["name", "email", "phone"]
            }
        },
        {
            "name": "feature_selection",
            "type": "minimization",
            "input": "anonymized_data",
            "output": "minimized_data",
            "parameters": {
                "selected_features": ["age", "diagnosis", "income"]
            }
        }
    ]

    # Audit the entire pipeline
    pipeline_report = auditor.audit_pipeline(
        dataset=df,
        dataset_name="patient_records",
        model=model_config,
        model_name="diagnosis_predictor",
        preprocessing_steps=preprocessing_steps,
        metadata={
            "pipeline_version": "1.0",
            "last_updated": "2024-01-01",
            "data_owner": "Hospital A",
            "data_encrypted": True
        }
    )

    # Print results for each component
    for component, report in pipeline_report.items():
        print(f"\n{component.upper()} Report:")
        print(report)

Generating PDF Reports
-------------------

Generate a detailed PDF report of the compliance audit:

.. code-block:: python

    # Generate PDF report
    pdf_path = auditor.generate_pdf(
        audit_result=pipeline_report,
        output_file="compliance_report.pdf",
        title="Patient Records Pipeline Compliance Audit",
        logo_path="hospital_logo.png"  # Optional
    )
    print(f"PDF report generated at: {pdf_path}")

Supported Regulations
-------------------

SecureML supports compliance checking for multiple privacy regulations:

- GDPR (General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- HIPAA (Health Insurance Portability and Accountability Act)

Each regulation has specific requirements that are checked during the audit process:

- Data minimization
- Consent management
- Data storage location
- Right to be forgotten
- Data encryption
- Anonymization requirements
- Cross-border data transfer rules

The compliance checker will automatically apply the appropriate checks based on the specified regulation. 