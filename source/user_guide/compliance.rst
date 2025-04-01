===========
Compliance
===========

Compliance checking is a critical component of privacy-preserving machine learning. SecureML provides tools to verify that your datasets and models comply with relevant privacy regulations such as GDPR, CCPA, and HIPAA.

Core Concepts
------------

**Regulation Frameworks**: SecureML supports checks against major privacy regulations including:

* **GDPR**: General Data Protection Regulation (European Union)
* **CCPA**: California Consumer Privacy Act
* **HIPAA**: Health Insurance Portability and Accountability Act
* **PCI DSS**: Payment Card Industry Data Security Standard
* **Custom frameworks**: Define your own compliance requirements

**Compliance Levels**:

* **Dataset-level compliance**: Verifying data handling practices
* **Model-level compliance**: Ensuring models don't leak sensitive information
* **System-level compliance**: Checking the entire machine learning pipeline

Basic Usage
----------

Checking Dataset Compliance
^^^^^^^^^^^^^^^^^^^^^^^^^

To check if a dataset complies with privacy regulations:

.. code-block:: python

    from secureml.compliance import check_dataset_compliance
    
    # Check dataset compliance
    compliance_report = check_dataset_compliance(
        data=df,
        regulations=['gdpr', 'ccpa'],
        sensitive_attributes=['ssn', 'name', 'address', 'dob'],
        quasi_identifiers=['zipcode', 'age', 'gender'],
        data_purpose='marketing_analytics',
        data_origin='customer_database'
    )
    
    # Print compliance status
    print(f"Overall compliance: {compliance_report.is_compliant}")
    
    # Get detailed findings
    if not compliance_report.is_compliant:
        for issue in compliance_report.issues:
            print(f"- {issue.severity}: {issue.description}")
    
    # Generate compliance report in various formats
    compliance_report.to_pdf('compliance_report.pdf')
    compliance_report.to_json('compliance_report.json')
    compliance_report.to_html('compliance_report.html')

Checking Model Compliance
^^^^^^^^^^^^^^^^^^^^^^^

To check if a model complies with privacy regulations:

.. code-block:: python

    from secureml.compliance import check_model_compliance
    
    # Check model compliance
    model_compliance = check_model_compliance(
        model=trained_model,
        training_data=training_df,
        regulations=['gdpr', 'hipaa'],
        model_purpose='patient_diagnosis',
        model_type='classification'
    )
    
    # Print compliance status
    print(f"Model compliance: {model_compliance.is_compliant}")
    
    # Get detailed issues
    for issue in model_compliance.issues:
        print(f"- {issue.severity}: {issue.description}")
        print(f"  Recommendation: {issue.recommendation}")

Combined Compliance Check
^^^^^^^^^^^^^^^^^^^^^^

For a comprehensive check of both data and model:

.. code-block:: python

    from secureml.compliance import check_compliance
    
    # Comprehensive compliance check
    compliance_report = check_compliance(
        data=df,
        model=trained_model,
        regulations=['gdpr', 'ccpa', 'hipaa'],
        sensitive_attributes=['ssn', 'patient_id', 'name'],
        quasi_identifiers=['zipcode', 'age', 'gender'],
        data_purpose='healthcare_analytics',
        model_purpose='patient_risk_assessment'
    )
    
    # Get compliance summary
    print(compliance_report.summary())

Advanced Techniques
------------------

Creating Custom Compliance Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define custom compliance requirements:

.. code-block:: python

    from secureml.compliance import ComplianceFramework, ComplianceRule, Severity
    
    # Create custom rules
    custom_rules = [
        ComplianceRule(
            id='custom-rule-001',
            name='No email addresses in dataset',
            description='Dataset should not contain email addresses',
            check_function=lambda data: 'email' not in data.columns,
            severity=Severity.HIGH,
            recommendation='Remove or anonymize email column'
        ),
        ComplianceRule(
            id='custom-rule-002',
            name='Age binning required',
            description='Age must be binned in groups of at least 5 years',
            check_function=lambda data: not ('age' in data.columns and data['age'].nunique() > 20),
            severity=Severity.MEDIUM,
            recommendation='Bin age into 5-year groups'
        )
    ]
    
    # Create custom framework
    internal_framework = ComplianceFramework(
        id='internal-privacy-policy',
        name='Internal Privacy Policy',
        version='1.0',
        rules=custom_rules
    )
    
    # Register the framework
    from secureml.compliance import register_framework
    register_framework(internal_framework)
    
    # Use the custom framework in compliance checks
    compliance_report = check_dataset_compliance(
        data=df,
        regulations=['internal-privacy-policy', 'gdpr'],
        sensitive_attributes=['ssn', 'name', 'address']
    )

Continuous Compliance Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set up automated compliance checks:

.. code-block:: python

    from secureml.compliance import ComplianceMonitor
    
    # Create a compliance monitor
    monitor = ComplianceMonitor(
        regulations=['gdpr', 'ccpa'],
        schedule='daily',  # Options: 'hourly', 'daily', 'weekly'
        notification_email='privacy@example.com',
        fail_on_non_compliance=True
    )
    
    # Register assets to monitor
    monitor.register_dataset(
        data=customer_data,
        name='customer_database',
        sensitive_attributes=['ssn', 'credit_card']
    )
    
    monitor.register_model(
        model=recommendation_model,
        name='recommendation_engine',
        training_data=training_data
    )
    
    # Start monitoring
    monitor.start()
    
    # Stop monitoring
    monitor.stop()

Specific Regulation Checks
^^^^^^^^^^^^^^^^^^^^^^^

Perform checks focused on specific regulations:

**GDPR-specific checks:**

.. code-block:: python

    from secureml.compliance.gdpr import check_gdpr_compliance
    
    gdpr_report = check_gdpr_compliance(
        data=df,
        data_purpose='customer_analytics',
        has_consent=True,
        retention_period_days=90,
        data_subject_access_mechanism='api',
        right_to_be_forgotten_implemented=True,
        cross_border_transfers=['eu', 'usa']
    )
    
    # Check specific GDPR article compliance
    article_5_compliance = gdpr_report.get_article_compliance('article_5')
    print(f"Article 5 compliance: {article_5_compliance.is_compliant}")

**HIPAA-specific checks:**

.. code-block:: python

    from secureml.compliance.hipaa import check_hipaa_compliance
    
    hipaa_report = check_hipaa_compliance(
        data=patient_data,
        phi_attributes=['patient_name', 'medical_record_number', 'treatment_codes'],
        has_authorization=True,
        minimum_necessary_applied=True,
        security_measures={
            'encryption_at_rest': True,
            'encryption_in_transit': True,
            'access_controls': True,
            'audit_trails': True
        }
    )
    
    print(f"HIPAA compliance: {hipaa_report.is_compliant}")

Privacy Impact Assessment
^^^^^^^^^^^^^^^^^^^^^^

Conduct a full privacy impact assessment:

.. code-block:: python

    from secureml.compliance import privacy_impact_assessment
    
    pia_results = privacy_impact_assessment(
        data=df,
        model=trained_model,
        application_name='Customer Churn Prediction',
        data_flows=[
            {'source': 'CRM System', 'destination': 'Analytics Platform', 'data_elements': ['customer_id', 'purchase_history']},
            {'source': 'Analytics Platform', 'destination': 'Marketing System', 'data_elements': ['churn_risk_score']}
        ],
        data_retention_policy='90 days',
        data_protection_measures=['encryption', 'access_control', 'anonymization'],
        risk_mitigation_measures=['staff_training', 'regular_audits']
    )
    
    # Generate formal PIA report
    pia_results.generate_report('privacy_impact_assessment.docx')

Data Protection by Design
^^^^^^^^^^^^^^^^^^^^^

Assess compliance with Data Protection by Design principles:

.. code-block:: python

    from secureml.compliance import assess_data_protection_by_design
    
    dpd_assessment = assess_data_protection_by_design(
        ml_pipeline=pipeline,
        principles_implemented={
            'data_minimization': True,
            'purpose_limitation': True,
            'storage_limitation': True,
            'privacy_by_default': True
        },
        documentation={
            'privacy_notice': True,
            'dpia_conducted': True,
            'processing_records': True
        }
    )
    
    print(dpd_assessment.summary())

Compliance Documentation Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate necessary compliance documentation:

.. code-block:: python

    from secureml.compliance.documentation import generate_compliance_documentation
    
    # Generate documentation suite
    docs = generate_compliance_documentation(
        data=df,
        model=trained_model,
        regulations=['gdpr', 'ccpa'],
        required_documents=['privacy_notice', 'processing_records', 'dpia', 'consent_form']
    )
    
    # Save documents
    for doc_name, doc_content in docs.items():
        with open(f"{doc_name}.md", "w") as f:
            f.write(doc_content)

Best Practices
-------------

1. **Start early**: Build compliance into your ML workflows from the beginning, not as an afterthought

2. **Be comprehensive**: Check compliance across all phases of the ML lifecycle, from data collection to model deployment

3. **Document everything**: Maintain detailed records of compliance checks and actions taken to address issues

4. **Stay updated**: Regularly update compliance checks as regulations and internal policies evolve

5. **Automate checks**: Implement continuous compliance monitoring in ML pipelines

6. **Involve experts**: Consult with legal and privacy experts when designing compliance checks

7. **Balance with utility**: Find the right balance between compliance requirements and model utility

Further Reading
-------------

* :doc:`/api/compliance` - Complete API reference for compliance functions
* :doc:`/examples/compliance` - More examples of compliance checking techniques 
* :doc:`/regulations/gdpr` - Detailed guide on GDPR compliance
* :doc:`/regulations/ccpa` - Detailed guide on CCPA compliance
* :doc:`/regulations/hipaa` - Detailed guide on HIPAA compliance 