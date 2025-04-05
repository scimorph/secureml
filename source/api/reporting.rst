=================
Reporting API
=================

.. module:: secureml.reporting

This module provides tools for generating compliance and audit reports from SecureML operations, helping to document and visualize privacy compliance for various regulations.

ReportGenerator Class
--------------------

.. autoclass:: ReportGenerator
   :members:
   :special-members: __init__

The ``ReportGenerator`` class provides methods for creating HTML and PDF reports from audit logs and compliance checks.

Basic Usage Example:

.. code-block:: python

    from secureml.reporting import ReportGenerator
    
    # Create a report generator
    generator = ReportGenerator()
    
    # Generate a compliance report
    report_path = generator.generate_compliance_report(
        report=compliance_report,  # A ComplianceReport instance
        output_file="compliance_report.html",
        logo_path="company_logo.png",  # Optional
        include_charts=True  # Include visualizations
    )
    
    print(f"Compliance report generated at: {report_path}")

Generating Audit Reports
-----------------------

.. code-block:: python

    # Generate an audit report from audit logs
    audit_report_path = generator.generate_audit_report(
        logs=audit_logs,  # List of audit log entries
        output_file="audit_report.pdf",
        title="GDPR Compliance Audit Report",
        logo_path="company_logo.png",
        include_charts=True
    )

You can also generate audit reports directly from operation IDs or query parameters:

.. code-block:: python

    # Generate an audit report from an operation ID
    report_path = generator.generate_audit_report(
        logs="12345-abcde-67890",  # Operation ID
        output_file="model_training_audit.html",
        title="Model Training Audit Report"
    )
    
    # Generate an audit report using query parameters
    report_path = generator.generate_audit_report(
        logs={
            "operation_name": "model_training",
            "start_time": "2023-01-01T00:00:00",
            "end_time": "2023-01-31T23:59:59"
        },
        output_file="january_training_audit.pdf"
    )

Custom Report Templates
---------------------

The ``ReportGenerator`` uses Jinja2 templates to render reports. By default, templates are stored in the ``templates`` directory within the SecureML package. You can customize these templates:

.. code-block:: python

    # Create a report generator with custom templates
    generator = ReportGenerator(
        templates_dir="/path/to/custom/templates",
        custom_css="path/to/custom/style.css"
    )

The default templates include:
- ``compliance_report.html``: Template for compliance reports
- ``audit_report.html``: Template for audit reports
- ``report_style.css``: CSS styling for reports

ComplianceReport Extension
------------------------

The module extends the ``ComplianceReport`` class with a ``generate_report`` method:

.. code-block:: python

    from secureml.compliance import check_compliance
    
    # Run a compliance check
    compliance_report = check_compliance(
        data=my_dataset,
        regulation="GDPR"
    )
    
    # Generate a report directly from the compliance report
    report_path = compliance_report.generate_report(
        output_file="compliance_report.pdf",
        format="pdf",  # or "html"
        logo_path="logo.png",
        include_charts=True
    )

PDF Generation
------------

The ``ReportGenerator`` can generate both HTML and PDF reports. PDF generation requires WeasyPrint:

.. code-block:: python

    # Generate a PDF report
    pdf_path = generator.generate_compliance_report(
        report=compliance_report,
        output_file="report.pdf"  # .pdf extension triggers PDF generation
    )

Charts and Visualizations
-----------------------

The ``ReportGenerator`` can include visualizations in reports:

- For compliance reports: Charts showing issues by severity
- For audit reports: Charts showing events by type

These visualizations help to quickly understand the compliance status and audit activity.

Best Practices
------------

1. **Use descriptive titles**: Provide clear titles for reports to make them easier to identify later
2. **Include logos**: Add organizational logos for official reports
3. **Enable charts**: Include visualizations for easier interpretation
4. **Use PDF format**: For official reports that need to be shared externally
5. **Customize templates**: Adapt templates to match organizational branding and requirements
6. **Include context**: Add additional context information to enhance report usefulness
