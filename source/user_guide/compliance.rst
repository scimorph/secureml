===========
Compliance
===========

Compliance checking is a critical component of privacy-preserving machine learning. SecureML provides tools to verify that your datasets and models comply with relevant privacy regulations such as GDPR, CCPA, and HIPAA.

Core Concepts
------------

**Compliance Report**: A structured report containing compliance check results, including issues, warnings, and passed checks.

**Supported Regulations**: SecureML supports checks against major privacy regulations:

* **GDPR**: General Data Protection Regulation (European Union)
* **CCPA**: California Consumer Privacy Act
* **HIPAA**: Health Insurance Portability and Accountability Act
* **LGPD**: Brazilian General Data Protection Law (Brazil)

**Compliance Levels**:

* **Dataset-level compliance**: Detecting personal data and PHI in datasets
* **Model-level compliance**: Verifying that models support privacy requirements
* **Pipeline-level compliance**: Checking the entire machine learning pipeline

Basic Usage
----------

Basic Compliance Check
^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to check compliance is using the `check_compliance` function:

.. code-block:: python

    from secureml.compliance import check_compliance
    import pandas as pd
    
    # Sample dataset
    data = pd.DataFrame({
        'name': ['Alice Smith', 'Bob Johnson'],
        'age': [30, 45],
        'email': ['alice@example.com', 'bob@example.com'],
        'diagnosis': ['Flu', 'Diabetes']
    })
    
    # Check compliance with GDPR
    report = check_compliance(
        data=data,
        regulation="GDPR",
        max_samples=100  # Maximum number of samples to analyze
    )
    
    # Print the report
    print(report)
    
    # Check compliance status
    if report.has_issues():
        print("Compliance issues found!")
        for issue in report.issues:
            print(f"- {issue['severity']}: {issue['issue']}")
            print(f"  Recommendation: {issue['recommendation']}")

Using ComplianceAuditor
^^^^^^^^^^^^^^^^^^^^^^^

For more comprehensive audits, use the `ComplianceAuditor` class:

.. code-block:: python

    from secureml.compliance import ComplianceAuditor
    
    # Create an auditor for GDPR
    auditor = ComplianceAuditor(
        regulation="GDPR",
        log_dir="audit_logs"  # Optional: directory to store audit logs
    )

Dataset Audit
~~~~~~~~~~~~

Audit a dataset for compliance:

.. code-block:: python

    # Audit a dataset with metadata
    dataset_report = auditor.audit_dataset(
        dataset=data,
        dataset_name="patient_records",
        metadata={
            "description": "Patient medical records",
            "data_owner": "Hospital A", 
            "data_retention_period": "5 years",
            "data_encrypted": True,
            "data_storage_location": "EU"
        }
    )
    
    # Print the report
    print(dataset_report)

Model Audit
~~~~~~~~~~

Audit a model for compliance:

.. code-block:: python

    # Model configuration
    model_config = {
        "model_type": "RandomForestClassifier",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 5
        },
        "supports_forget_request": True,  # Supports GDPR right to be forgotten
        "data_processing_purpose": "Medical diagnosis prediction"
    }
    
    # Audit the model
    model_report = auditor.audit_model(
        model_config=model_config,
        model_name="diagnosis_predictor",
        model_documentation={
            "version": "1.0",
            "training_date": "2024-01-01",
            "training_data_description": "Patient records from 2023"
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
        dataset=data,
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
    
    # The pipeline audit returns a dictionary with individual component reports
    for component, report in pipeline_report.items():
        print(f"\n{component.upper()} Report:")
        print(report)

Generating PDF Reports
-------------------

Generate a detailed PDF report of the compliance audit:

.. code-block:: python

    # Generate PDF report from pipeline audit
    pdf_path = auditor.generate_pdf(
        audit_result=pipeline_report,
        output_file="compliance_report.pdf",
        title="Patient Records Pipeline Compliance Audit",
        logo_path="company_logo.png"  # Optional
    )

How Compliance Checks Work
-------------------------

Identifying Sensitive Data
^^^^^^^^^^^^^^^^^^^^^^^^^

SecureML uses several approaches to identify sensitive data:

1. **Column name analysis**: Checks column names against known patterns of sensitive data
2. **Content analysis**: Uses NLP techniques to identify patterns in text data
3. **Automated detection**: The `_identify_sensitive_columns` function can automatically detect potentially sensitive columns

.. code-block:: python

    from secureml.anonymization import _identify_sensitive_columns
    
    # Automatically identify sensitive columns
    sensitive_cols = _identify_sensitive_columns(data)
    print(f"Automatically identified sensitive columns: {sensitive_cols}")

Regulation-Specific Checks
^^^^^^^^^^^^^^^^^^^^^^^^

Each regulation has specific checks based on its requirements:

**GDPR Checks**:
- Personal data identification
- Special category data identification
- Data minimization
- Explicit consent
- Right to be forgotten capability
- Cross-border data transfer

**CCPA Checks**:
- Personal information identification
- California residents' data handling
- Sale of personal information
- Deletion capability

**HIPAA Checks**:
- Protected Health Information (PHI) identification
- De-identification method verification
- Data security and encryption

**LGPD Checks**:
- Personal data identification
- Sensitive data identification
- Data minimization
- Explicit consent
- Right to be forgotten capability
- Cross-border data transfer

Regulation Presets
----------------

SecureML uses presets for each regulation stored in YAML files. You can access preset information programmatically:

.. code-block:: python

    from secureml.presets import list_available_presets, load_preset, get_preset_field
    
    # List available regulation presets
    available_presets = list_available_presets()
    print(f"Available regulations: {available_presets}")
    
    # Load a specific preset
    gdpr_preset = load_preset("gdpr")
    
    # Get specific field from a preset
    personal_data_identifiers = get_preset_field("gdpr", "personal_data_identifiers")
    special_categories = get_preset_field("gdpr", "special_categories")
    
    print(f"GDPR personal data identifiers: {personal_data_identifiers}")

Best Practices
-------------

1. **Start early**: Build compliance into your ML workflows from the beginning, not as an afterthought

2. **Be comprehensive**: Check compliance across all phases of the ML lifecycle, from data collection to model deployment

3. **Document everything**: Maintain detailed records of compliance checks and actions taken to address issues

4. **Add appropriate metadata**: Include information about data sources, consent, processing purpose, etc.

5. **Regular audits**: Schedule regular compliance audits of your ML systems 

6. **Integrate with audit trails**: Use audit trails to document compliance activities

7. **Remediate issues**: Address identified compliance issues promptly

8. **Stay updated**: Keep abreast of changes in regulations that may affect compliance requirements

Further Reading
-------------

* :doc:`/api/compliance` - Complete API reference for compliance functions
* :doc:`/examples/compliance` - More examples of compliance checking techniques 
* :doc:`/regulations/gdpr` - Detailed guide on GDPR compliance
* :doc:`/regulations/ccpa` - Detailed guide on CCPA compliance
* :doc:`/regulations/hipaa` - Detailed guide on HIPAA compliance 