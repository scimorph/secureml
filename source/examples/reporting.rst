Report Generation Examples
=======================

This section demonstrates how to create compliance and audit reports with SecureML. Reports are critical for documenting privacy and compliance measures for regulators, auditors, and stakeholders.

Basic Compliance Reports
----------------------

The simplest way to generate a compliance report is from an existing ComplianceReport object:

.. code-block:: python

    from secureml.reporting import ReportGenerator
    from secureml.compliance import ComplianceReport
    
    # Create a compliance report
    report = ComplianceReport("GDPR")
    
    # Add passed checks
    report.add_passed_check("Data minimization principle")
    report.add_passed_check("Explicit consent obtained")
    
    # Add warnings
    report.add_warning(
        component="Data Storage",
        warning="Data retention period not specified",
        recommendation="Define explicit data retention periods"
    )
    
    # Add issues
    report.add_issue(
        component="Sensitive Data",
        issue="Email addresses not encrypted",
        severity="medium",
        recommendation="Apply encryption to email fields"
    )
    
    # Create a report generator
    generator = ReportGenerator()
    
    # Generate HTML report
    output_file = "compliance_report.html"
    report_path = generator.generate_compliance_report(
        report=report,
        output_file=output_file,
        include_charts=True
    )
    
    print(f"HTML report generated and saved to: {output_file}")

The generated HTML report includes:

- A summary of the compliance status
- Charts showing issues by severity
- Lists of passed checks, warnings, and issues
- Recommendations for addressing each issue

Reports From Compliance Checks
----------------------------

You can generate reports directly from compliance checks on datasets:

.. code-block:: python

    import pandas as pd
    from secureml.compliance import check_compliance
    from secureml.reporting import ReportGenerator
    
    # Sample data with sensitive information
    data = pd.DataFrame({
        'name': ['John Smith', 'Jane Doe'],
        'age': [34, 29],
        'email': ['john.smith@example.com', 'jane.doe@example.com'],
        'phone': ['555-123-4567', '555-234-5678'],
        'ssn': ['123-45-6789', '234-56-7890'],
        'medical_condition': ['Diabetes', 'None'],
        'income': [65000, 72000]
    })
    
    # Check compliance with GDPR
    report = check_compliance(
        data=data,
        regulation="GDPR",
        max_samples=10
    )
    
    # Create a report generator
    generator = ReportGenerator()
    
    # Generate HTML report
    output_file = "dataset_compliance_report.html"
    report_path = generator.generate_compliance_report(
        report=report,
        output_file=output_file,
        include_charts=True
    )

This workflow is particularly useful for:
- Documenting dataset compliance before ML model training
- Regular compliance audits of data processing systems
- Demonstrating compliance to privacy officers and regulators

Audit Reports
-----------

You can generate reports from audit logs to track data processing operations:

.. code-block:: python

    from secureml.reporting import ReportGenerator
    from secureml.audit import get_audit_logs
    
    # Retrieve audit logs
    logs = get_audit_logs(
        operation_name="model_training",
        start_time="2023-01-01T00:00:00",
        end_time="2023-01-31T23:59:59"
    )
    
    # Create a report generator
    generator = ReportGenerator()
    
    # Generate HTML report
    output_file = "audit_report.html"
    report_path = generator.generate_audit_report(
        logs=logs,
        output_file=output_file,
        title="ML Model Training Audit Report",
        include_charts=True,
        additional_context={
            "regulations": ["GDPR", "HIPAA"],
            "data_owner": "Research Department",
            "report_purpose": "Regulatory compliance verification"
        }
    )

For a complete audit trail workflow, you can create an audit trail, log events, and then generate a report:

.. code-block:: python

    from secureml.audit import AuditTrail, get_audit_logs
    from secureml.reporting import ReportGenerator
    
    # Create an audit trail
    audit = AuditTrail(
        operation_name="model_training",
        regulations=["GDPR", "HIPAA"]
    )
    
    # Log events
    audit.log_data_access(
        dataset_name="patient_data",
        columns_accessed=["age", "gender", "blood_pressure"],
        num_records=1000,
        purpose="Training disease prediction model"
    )
    
    audit.log_model_training(
        model_type="RandomForest",
        dataset_name="patient_data_anonymized",
        parameters={"n_estimators": 100, "max_depth": 10},
        metrics={"accuracy": 0.85, "auc": 0.91},
        privacy_parameters={"anonymization": "k_anonymity_5"}
    )
    
    # Close the audit trail
    audit.close()
    
    # Retrieve the audit logs
    logs = get_audit_logs(operation_name="model_training")
    
    # Generate a report
    generator = ReportGenerator()
    generator.generate_audit_report(
        logs=logs,
        output_file="model_training_audit.html",
        title="Model Training Audit Report",
        include_charts=True
    )

Customizing Reports
-----------------

You can customize reports with logos and custom CSS:

.. code-block:: python

    # Define custom CSS
    custom_css = """
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f9f9f9;
    }
    
    .report-header {
        background-color: #3498db;
        color: white;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 30px;
    }
    
    .high {
        color: #e74c3c;
        font-weight: bold;
    }
    
    .medium {
        color: #f39c12;
        font-weight: bold;
    }
    
    .low {
        color: #3498db;
        font-weight: bold;
    }
    """
    
    # Create a report generator with custom CSS
    generator = ReportGenerator(custom_css=custom_css)
    
    # Generate report with a logo
    report_path = generator.generate_compliance_report(
        report=compliance_report,
        output_file="custom_report.html",
        logo_path="company_logo.png",
        include_charts=True,
        additional_context={
            "organization": "Example Corporation",
            "department": "Data Science Team",
            "project": "Customer Behavior Analysis"
        }
    )

Combined Reports
-------------

You can create combined reports that include both compliance and audit information:

.. code-block:: python

    import os
    from datetime import datetime
    from secureml.reporting import ReportGenerator
    
    # First, create separate reports
    generator = ReportGenerator()
    
    # Generate compliance report to a temporary file
    compliance_file = "temp_compliance.html"
    generator.generate_compliance_report(
        report=compliance_report,
        output_file=compliance_file,
        include_charts=True
    )
    
    # Generate audit report to a temporary file
    audit_file = "temp_audit.html"
    generator.generate_audit_report(
        logs=audit_logs,
        output_file=audit_file,
        title="ML Operation Audit Trail",
        include_charts=True
    )
    
    # Read the contents of both files
    with open(compliance_file, 'r') as f:
        compliance_content = f.read()
    
    with open(audit_file, 'r') as f:
        audit_content = f.read()
    
    # Extract the main content from each
    compliance_body = compliance_content.split('<body>')[1].split('</body>')[0]
    audit_body = audit_content.split('<body>')[1].split('</body>')[0]
    
    # Create a combined HTML file
    combined_html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Combined Compliance and Audit Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .report-section {{
                margin-bottom: 30px;
                border: 1px solid #ddd;
                padding: 20px;
                border-radius: 5px;
            }}
            
            /* Preserve styling from original reports */
            {generator._get_css()}
        </style>
    </head>
    <body>
        <h1>Combined Compliance and Audit Report</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="report-section">
            <h2>Compliance Report</h2>
            {compliance_body}
        </div>
        
        <div class="report-section">
            <h2>Audit Report</h2>
            {audit_body}
        </div>
    </body>
    </html>
    """
    
    # Write the combined file
    combined_file = "combined_report.html"
    with open(combined_file, 'w') as f:
        f.write(combined_html)
    
    # Clean up temporary files
    os.remove(compliance_file)
    os.remove(audit_file)

Scheduled Reports
--------------

For regular reporting, you can set up automated report generation:

.. code-block:: python

    from datetime import datetime, timedelta
    import os
    from secureml.audit import get_audit_logs
    from secureml.reporting import ReportGenerator
    
    def generate_weekly_report(output_dir="reports"):
        """Generate a weekly report from logs."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate date range for the previous week
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Format for log retrieval
        start_time = start_date.isoformat()
        end_time = end_date.isoformat()
        
        # Retrieve logs
        logs = get_audit_logs(
            start_time=start_time,
            end_time=end_time
        )
        
        if not logs:
            print(f"No logs found for period {start_date} to {end_date}")
            return None
        
        # Generate report
        generator = ReportGenerator()
        output_file = f"{output_dir}/weekly_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.html"
        
        report_path = generator.generate_audit_report(
            logs=logs,
            output_file=output_file,
            title=f"Weekly Audit Report: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            include_charts=True,
            additional_context={
                "report_type": "Weekly",
                "generated_by": "Automated System"
            }
        )
        
        return report_path
    
    # This function can be called by a scheduler (cron, Windows Task Scheduler, Airflow, etc.)
    # Example cron entry (Linux/Mac) for running every Sunday at midnight:
    # 0 0 * * 0 python /path/to/generate_weekly_report.py

Complete Example
--------------

Here's a complete example that generates a comprehensive report for a privacy-preserving ML pipeline:

.. code-block:: python

    import pandas as pd
    import os
    from datetime import datetime
    
    from secureml.reporting import ReportGenerator
    from secureml.compliance import check_compliance, ComplianceReport
    from secureml.audit import AuditTrail, get_audit_logs
    
    # Create output directory
    report_dir = "privacy_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Create an audit trail for the entire process
    audit = AuditTrail(
        operation_name="ml_pipeline_execution",
        regulations=["GDPR"],
        context={"project": "Customer Churn Prediction"}
    )
    
    # 2. Load and check the dataset
    try:
        # Log the data access
        audit.log_data_access(
            dataset_name="customer_data",
            columns_accessed=["id", "age", "account_balance", "transaction_history", "email"],
            num_records=10000,
            purpose="Churn prediction model training",
            user="data_scientist_1"
        )
        
        # Simulate loading data
        data = pd.DataFrame({
            'id': range(1, 5),
            'age': [34, 29, 42, 35],
            'email': ['john@example.com', 'jane@example.com', 'robert@example.com', 'emily@example.com'],
            'account_balance': [5000, 12000, 3000, 8000],
            'churn_risk': [0.2, 0.1, 0.7, 0.3]
        })
        
        # Check compliance
        audit.log_event(
            "compliance_checking",
            {"dataset": "customer_data", "regulation": "GDPR"}
        )
        
        compliance_report = check_compliance(
            data=data,
            regulation="GDPR",
            max_samples=100
        )
        
        audit.log_compliance_check(
            check_type="dataset_compliance",
            regulation="GDPR",
            result=not compliance_report.has_issues(),
            details={
                "issues_count": len(compliance_report.issues),
                "warnings_count": len(compliance_report.warnings)
            }
        )
        
        # 3. Apply anonymization (simulated)
        audit.log_data_transformation(
            transformation_type="anonymization",
            input_data="customer_data",
            output_data="customer_data_anonymized",
            parameters={"method": "k_anonymity", "k": 5}
        )
        
        # 4. Train model (simulated)
        audit.log_model_training(
            model_type="RandomForest",
            dataset_name="customer_data_anonymized",
            parameters={"n_estimators": 100, "max_depth": 10},
            metrics={"accuracy": 0.85, "auc": 0.91, "f1": 0.87},
            privacy_parameters={"anonymization": "k_anonymity_5"}
        )
        
        # 5. Close the audit trail
        audit.close("completed")
        
        # 6. Generate reports
        generator = ReportGenerator()
        
        # Compliance report
        compliance_file = f"{report_dir}/compliance_report.html"
        generator.generate_compliance_report(
            report=compliance_report,
            output_file=compliance_file,
            include_charts=True
        )
        
        # Audit report
        logs = get_audit_logs(operation_name="ml_pipeline_execution")
        audit_file = f"{report_dir}/audit_report.html"
        generator.generate_audit_report(
            logs=logs,
            output_file=audit_file,
            title="ML Pipeline Execution Audit",
            include_charts=True
        )
        
        print(f"Report generation completed. Reports saved to {report_dir}")
        
    except Exception as e:
        # Log the error
        audit.log_error(
            error_type=type(e).__name__,
            message=str(e)
        )
        audit.close("error")
        raise
    
Best Practices
------------

1. **Be consistent with reporting**: Generate reports at regular intervals and after significant ML operations.

2. **Include context**: Add metadata like project name, department, and purpose to make reports more meaningful.

3. **Customize reports for different audiences**:
   - Technical teams need detailed error messages and code references
   - Management needs high-level summaries and risk assessments
   - Regulators need compliance status and evidence of controls
   
4. **Store reports securely**: Reports often contain sensitive information about vulnerabilities.

5. **Automate report generation**: Set up scheduled tasks for regular reporting.

6. **Include visual elements**: Charts and graphs make reports more understandable.

7. **Provide actionable recommendations**: Every issue should have a clear recommendation.

8. **Establish a reporting workflow**: Define who receives reports and how issues are addressed. 