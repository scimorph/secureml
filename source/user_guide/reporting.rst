==========
Reporting
==========

SecureML provides powerful reporting tools to document privacy measures, generate compliance documentation, and create audit trail reports. These tools help organizations demonstrate compliance with privacy regulations and maintain transparency in their AI systems.

Core Concepts
------------

**Report Generation**: Creating structured documentation of privacy measures, compliance checks, and audit trails.

**Report Formats**: Output reports in different formats including HTML and PDF.

**Visual Elements**: Including charts, graphs, and other visual elements to aid in understanding complex privacy data.

**Compliance Documentation**: Generating reports specifically designed to demonstrate regulatory compliance.

Basic Usage
----------

Generating Compliance Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate comprehensive reports from compliance checks:

.. code-block:: python

    from secureml import check_compliance
    import pandas as pd

    # Load your data
    data = pd.read_csv("sensitive_data.csv")

    # Check compliance
    report = check_compliance(data, regulation="GDPR")

    # Generate an HTML report
    report.generate_report("gdpr_compliance.html")

    # Generate a PDF report (requires WeasyPrint)
    report.generate_report("gdpr_compliance.pdf", format="pdf")

Compliance reports include:

* Overall compliance status
* Detailed issues and warnings
* Recommendations for addressing compliance gaps
* Visual breakdown of compliance issues by severity

Generating Audit Trail Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create reports from audit trails to document operations on sensitive data:

.. code-block:: python

    from secureml import get_audit_logs, ReportGenerator

    # Retrieve audit logs for a specific operation
    logs = get_audit_logs(
        operation_name="data_anonymization",
        start_time="2023-01-01T00:00:00",
        end_time="2023-01-31T23:59:59"
    )

    # Create a report generator
    generator = ReportGenerator()

    # Generate an HTML report
    generator.generate_audit_report(
        logs=logs,
        output_file="audit_report.html",
        title="Data Anonymization Audit"
    )

    # Generate a PDF report
    generator.generate_audit_report(
        logs=logs,
        output_file="audit_report.pdf",
        title="Data Anonymization Audit"
    )

Advanced Techniques
------------------

Customizing Report Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create custom report templates to match organizational branding:

.. code-block:: python

    from secureml import ReportGenerator

    # Create a report generator with custom templates
    generator = ReportGenerator(
        templates_dir="my_custom_templates/",
        custom_css="my_custom_style.css"
    )

    # Generate a report using custom templates
    generator.generate_compliance_report(
        report=compliance_report,
        output_file="custom_compliance_report.html",
        logo_path="company_logo.png"
    )

Adding Charts and Visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enhance reports with visual elements:

.. code-block:: python

    from secureml import ReportGenerator
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    # Create a custom chart
    plt.figure(figsize=(8, 5))
    plt.bar(['High', 'Medium', 'Low'], [5, 8, 3])
    plt.title('Privacy Risks by Severity')
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    chart_data = base64.b64encode(buffer.read()).decode('utf-8')

    # Create a report generator
    generator = ReportGenerator()

    # Generate a report with the custom chart
    generator.generate_compliance_report(
        report=compliance_report,
        output_file="report_with_charts.html",
        additional_context={"custom_chart": chart_data}
    )

Comprehensive ML Pipeline Reporting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate reports for an entire ML pipeline:

.. code-block:: python

    from secureml import ComplianceAuditor
    import pandas as pd

    # Create a compliance auditor for HIPAA
    auditor = ComplianceAuditor(regulation="HIPAA")

    # Load dataset
    data = pd.read_csv("patient_data.csv")

    # Define model configuration
    model_config = {
        "model_type": "RandomForest",
        "supports_forget_request": True,
        "access_controls": True,
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10
        }
    }

    # Define preprocessing steps
    preprocessing_steps = [
        {
            "name": "remove_identifiers",
            "type": "anonymization",
            "input": "raw_data",
            "output": "deidentified_data",
            "parameters": {"columns_to_remove": ["name", "ssn", "address"]}
        },
        {
            "name": "feature_selection",
            "type": "data_minimization",
            "input": "deidentified_data",
            "output": "minimal_data",
            "parameters": {"selected_features": ["age", "lab_results", "diagnosis"]}
        }
    ]

    # Audit the entire pipeline
    audit_result = auditor.audit_pipeline(
        dataset=data,
        dataset_name="patient_records",
        model=model_config,
        model_name="diagnosis_predictor",
        preprocessing_steps=preprocessing_steps,
        metadata={"data_storage_location": "US-East", "data_encrypted": True}
    )

    # Generate a comprehensive PDF report
    auditor.generate_pdf(
        audit_result=audit_result,
        output_file="compliance_report.pdf",
        title="HIPAA Compliance Audit"
    )

Integrating with Audit Trails
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the reporting tools with audit trails to create comprehensive documentation:

.. code-block:: python

    from secureml import AuditTrail, ReportGenerator

    # Create an audit trail for an operation
    audit = AuditTrail(
        operation_name="data_anonymization", 
        regulations=["GDPR"]
    )

    # Log dataset access
    audit.log_data_access(
        dataset_name="patient_records",
        columns_accessed=["age", "gender", "zipcode", "disease"],
        num_records=5000,
        purpose="Anonymization for research"
    )

    # Log data transformation
    audit.log_data_transformation(
        transformation_type="k_anonymity",
        input_data="Raw patient data",
        output_data="Anonymized patient data",
        parameters={"k": 5, "quasi_identifiers": ["age", "gender", "zipcode"]}
    )

    # Close the audit trail when done
    audit.close(status="completed")

    # Get the audit logs
    logs = get_audit_logs(operation_name="data_anonymization")

    # Generate a report from the audit logs
    generator = ReportGenerator()
    generator.generate_audit_report(
        logs=logs,
        output_file="anonymization_audit.pdf",
        title="Data Anonymization Audit Report"
    )

Report Formats
------------

HTML Reports
^^^^^^^^^^

HTML reports are interactive, easy to view in browsers, and can include rich formatting:

.. code-block:: python

    # Generate an HTML report
    report.generate_report("compliance_report.html", format="html")

PDF Reports
^^^^^^^^^

PDF reports are ideal for formal documentation and sharing with stakeholders:

.. code-block:: python

    # Install WeasyPrint for PDF support
    # pip install secureml[pdf]
    # On Windows, you'll also need to install GTK libraries. See:
    # https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows

    # Generate a PDF report
    report.generate_report("compliance_report.pdf", format="pdf")

Report Components
--------------

Compliance Reports
^^^^^^^^^^^^^^^

Compliance reports typically include:

* **Summary Section**: Overall compliance status
* **Issues Section**: Detailed list of compliance issues with severity ratings
* **Warnings Section**: Potential compliance risks that need attention
* **Passed Checks Section**: Successfully passed compliance requirements
* **Visual Elements**: Charts showing issues by severity
* **Recommendations**: Actionable steps to address compliance gaps

Audit Reports
^^^^^^^^^^^

Audit reports typically include:

* **Timeline of Events**: Chronological record of operations
* **Operation Details**: Information about each logged event
* **Visual Breakdown**: Charts showing events by type
* **Metadata Section**: Information about the audited operation
* **Regulatory Context**: Applicable regulations and requirements

Integration with Command Line Interface
-------------------------------------

Generate reports using the SecureML CLI:

.. code-block:: bash

    # Generate a compliance report from a dataset
    secureml compliance check data.csv --regulation GDPR --output report.html --format html

    # Generate a PDF compliance report
    secureml compliance check data.csv --regulation HIPAA --output report.pdf --format pdf

    # Generate an audit report from logs
    secureml audit report --operation-name "data_anonymization" --output audit_report.html

Best Practices
-------------

1. **Regular Reporting**: Generate compliance and audit reports regularly, not just during audits

2. **Version Control**: Maintain report history to track compliance improvements over time

3. **Comprehensive Documentation**: Include all relevant details in reports to provide full context

4. **Consistent Branding**: Use custom templates that match organizational branding

5. **Include Visualizations**: Use charts and graphs to make complex data more accessible

6. **Multiple Formats**: Generate both HTML (for internal review) and PDF (for formal documentation)

7. **Secure Storage**: Store reports securely and implement appropriate access controls

8. **Actionable Insights**: Focus on providing clear recommendations for addressing issues

9. **Integration with Workflows**: Automate report generation as part of regular workflows

10. **Stakeholder Focus**: Tailor reports to the needs of different stakeholders (legal, technical, executive)

Further Reading
-------------

* :doc:`/api/reporting` - Complete API reference for reporting functions
* :doc:`/api/audit` - API reference for audit trail functions
* :doc:`/api/compliance` - API reference for compliance checking functions
* :doc:`/examples/reporting` - More examples of report generation techniques 