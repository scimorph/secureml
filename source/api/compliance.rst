=================
Compliance API
=================

.. module:: secureml.compliance

This module provides tools to verify that datasets and models comply with privacy regulations like GDPR, CCPA, and HIPAA.

Main Functions
-------------

.. autofunction:: check_compliance

This is the main function for checking compliance with privacy regulations:

.. code-block:: python

    from secureml.compliance import check_compliance
    
    # Check a dataset for GDPR compliance
    report = check_compliance(
        data=my_dataframe,
        regulation="GDPR"
    )
    
    # Check if any issues were found
    if report.has_issues():
        print(report)

Compliance Reports
-----------------

.. autoclass:: ComplianceReport
   :members:
   :special-members: __init__

The `ComplianceReport` class contains the results of a compliance check and provides methods for accessing and displaying those results:

.. code-block:: python

    # Access report summary
    summary = report.summary()
    
    # Get detailed information
    if report.has_issues():
        for issue in report.issues:
            print(f"Issue: {issue['issue']}")
            print(f"Severity: {issue['severity']}")
            print(f"Recommendation: {issue['recommendation']}")

Compliance Auditor
------------------

.. autoclass:: ComplianceAuditor
   :members:
   :special-members: __init__

The `ComplianceAuditor` class provides a higher-level interface for conducting compliance audits of ML pipelines, generating comprehensive audit trails, and producing detailed reports:

.. code-block:: python

    from secureml.compliance import ComplianceAuditor
    
    # Create an auditor for GDPR compliance
    auditor = ComplianceAuditor(regulation="GDPR")
    
    # Audit a dataset
    dataset_report = auditor.audit_dataset(
        dataset=my_dataframe,
        dataset_name="customer_data"
    )
    
    # Audit a model
    model_report = auditor.audit_model(
        model_config=model_params,
        model_name="credit_scoring_model"
    )
    
    # Audit an entire ML pipeline
    pipeline_report = auditor.audit_pipeline(
        dataset=my_dataframe,
        dataset_name="customer_data",
        model=my_model,
        model_name="credit_scoring_model",
        preprocessing_steps=preprocessing_config
    )
    
    # Generate a PDF report
    auditor.generate_pdf(
        pipeline_report,
        output_file="compliance_report.pdf",
        title="ML Pipeline Compliance Audit"
    )

Data Identification Functions
----------------------------

.. autofunction:: identify_personal_data

This function identifies personal data in a dataset:

.. code-block:: python

    from secureml.compliance import identify_personal_data
    
    # Identify personal data in a dataframe
    personal_data_info = identify_personal_data(
        data=my_dataframe,
        max_samples=200  # Analyze up to 200 samples for text content
    )
    
    # Check which columns contain personal data
    personal_columns = personal_data_info["columns"]
    
    # Check what personal data was found in text content
    content_findings = personal_data_info["content_findings"]

.. autofunction:: identify_phi

This function identifies Protected Health Information (PHI) in a dataset:

.. code-block:: python

    from secureml.compliance import identify_phi
    
    # Identify PHI in a healthcare dataset
    phi_info = identify_phi(
        data=healthcare_data,
        max_samples=100
    )
    
    # Check which columns contain PHI
    phi_columns = phi_info["columns"]

NLP Utilities
------------

.. autofunction:: get_nlp_model

This function loads and caches a SpaCy NLP model for text analysis:

.. code-block:: python

    from secureml.compliance import get_nlp_model
    
    # Get the default SpaCy model
    nlp = get_nlp_model()
    
    # Analyze text for entities
    doc = nlp("Patient John Doe was diagnosed with hypertension.")
    entities = [(ent.text, ent.label_) for ent in doc.ents]

Working with Regulation Presets
------------------------------

The compliance module uses regulation-specific presets that define rules and checks for each regulation. These presets are loaded from the `secureml.presets` module:

.. code-block:: python

    from secureml.presets import list_available_presets, load_preset, get_preset_field
    
    # List available regulations
    regulations = list_available_presets()  # Returns ['gdpr', 'ccpa', 'hipaa', ...]
    
    # Load GDPR preset
    gdpr_preset = load_preset("gdpr")
    
    # Get specific field from a preset
    personal_data_identifiers = get_preset_field("gdpr", "personal_data_identifiers")

Supported Regulations
--------------------

The module currently supports compliance checks for:

1. **GDPR** (General Data Protection Regulation)
   - Checks for personal data and special categories
   - Verifies data minimization
   - Checks for consent metadata
   - Verifies right-to-be-forgotten support

2. **CCPA** (California Consumer Privacy Act)
   - Checks for personal information disclosure
   - Verifies opt-out options for data sharing
   - Checks deletion request support

3. **HIPAA** (Health Insurance Portability and Accountability Act)
   - Identifies Protected Health Information (PHI)
   - Checks for proper de-identification
   - Verifies data encryption

Best Practices
-------------

1. **Regular audits**: Run compliance checks regularly, especially before training models
2. **Document remediation**: Document how compliance issues were addressed
3. **Multi-regulation**: Check against all regulations applicable to your jurisdiction
4. **Full pipeline**: Audit the entire ML pipeline, not just individual components
5. **Update checks**: Keep regulation presets updated as laws and interpretations change
