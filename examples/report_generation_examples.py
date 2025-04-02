import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import SecureML reporting functionality
from secureml.reporting import ReportGenerator
from secureml.compliance import ComplianceReport, check_compliance
from secureml.audit import AuditTrail, get_audit_logs

# Define a function to print example headers
def print_header(title):
    """Print a section header with formatting."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

# Example 1: Basic Compliance Report Generation
def example_basic_compliance_report():
    """
    Generate a basic compliance report from a ComplianceReport object.
    """
    print_header("Example 1: Basic Compliance Report Generation")
    
    # Create a compliance report
    print("Creating a sample compliance report...")
    report = ComplianceReport("GDPR")
    
    # Add some passed checks
    report.add_passed_check("Data minimization principle")
    report.add_passed_check("Explicit consent obtained")
    report.add_passed_check("Data retention policy")
    
    # Add some warnings
    report.add_warning(
        component="Data Storage",
        warning="Data retention period not specified",
        recommendation="Define explicit data retention periods"
    )
    
    # Add some issues
    report.add_issue(
        component="Sensitive Data",
        issue="Email addresses not encrypted",
        severity="medium",
        recommendation="Apply encryption to email fields"
    )
    report.add_issue(
        component="Processing",
        issue="No audit trail for data access",
        severity="high",
        recommendation="Implement audit logging for all data access operations"
    )
    
    print("Sample ComplianceReport created with:")
    print(f"- {len(report.passed_checks)} passed checks")
    print(f"- {len(report.warnings)} warnings")
    print(f"- {len(report.issues)} issues")
    
    # Create a report generator
    print("\nGenerating HTML report...")
    generator = ReportGenerator()
    
    # Create output directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Generate HTML report
    output_file = "reports/basic_compliance_report.html"
    report_path = generator.generate_compliance_report(
        report=report,
        output_file=output_file,
        include_charts=True
    )
    
    print(f"HTML report generated and saved to: {output_file}")
    
    return report, generator

# Example 2: Compliance Report from Dataset
def example_compliance_report_from_dataset():
    """
    Generate a compliance report by checking a dataset.
    """
    print_header("Example 2: Compliance Report from Dataset")
    
    # Create a sample dataset with sensitive information
    print("Creating a sample dataset with personal information...")
    data = pd.DataFrame({
        'name': ['John Smith', 'Jane Doe', 'Robert Johnson', 'Emily Williams'],
        'age': [34, 29, 42, 35],
        'email': ['john.smith@example.com', 'jane.doe@example.com', 
                 'robert.j@example.com', 'e.williams@example.com'],
        'phone': ['555-123-4567', '555-234-5678', '555-345-6789', '555-456-7890'],
        'ssn': ['123-45-6789', '234-56-7890', '345-67-8901', '456-78-9012'],
        'medical_condition': ['Diabetes', 'None', 'Hypertension', 'Asthma'],
        'income': [65000, 72000, 58000, 93000]
    })
    
    print("Sample data:")
    print(data.head())
    
    # Check compliance with GDPR
    print("\nChecking GDPR compliance...")
    report = check_compliance(
        data=data,
        regulation="GDPR",
        max_samples=4  # Using all rows in our small sample
    )
    
    print(f"Compliance check completed with:")
    print(f"- {len(report.passed_checks)} passed checks")
    print(f"- {len(report.warnings)} warnings")
    print(f"- {len(report.issues)} issues")
    
    # Create a report generator
    print("\nGenerating HTML report...")
    generator = ReportGenerator()
    
    # Generate HTML report
    output_file = "reports/dataset_compliance_report.html"
    report_path = generator.generate_compliance_report(
        report=report,
        output_file=output_file,
        include_charts=True
    )
    
    print(f"HTML report generated and saved to: {output_file}")
    
    return report, data, generator

# Example 3: Audit Report Generation
def example_audit_report():
    """
    Generate an audit report from audit logs.
    """
    print_header("Example 3: Audit Report Generation")
    
    # First, create some audit logs
    print("Creating audit logs for an example ML operation...")
    
    # Create an audit trail
    audit = AuditTrail(
        operation_name="model_training",
        regulations=["GDPR", "HIPAA"],
        context={"model_type": "RandomForest", "purpose": "disease_prediction"}
    )
    
    # Log various events
    audit.log_data_access(
        dataset_name="patient_data",
        columns_accessed=["age", "gender", "blood_pressure", "cholesterol"],
        num_records=1000,
        purpose="Training disease prediction model",
        user="data_scientist_1"
    )
    
    audit.log_data_transformation(
        transformation_type="anonymization",
        input_data="patient_data",
        output_data="patient_data_anonymized",
        parameters={"method": "k_anonymity", "k": 5}
    )
    
    audit.log_model_training(
        model_type="RandomForest",
        dataset_name="patient_data_anonymized",
        parameters={"n_estimators": 100, "max_depth": 10},
        metrics={"accuracy": 0.85, "auc": 0.91, "f1": 0.87},
        privacy_parameters={"anonymization": "k_anonymity_5"}
    )
    
    audit.log_compliance_check(
        check_type="data_protection",
        regulation="GDPR",
        result=True,
        details={"check": "Data minimization", "passed": True}
    )
    
    # Close the audit trail
    audit.close()
    
    print("Audit logs created")
    
    # Retrieve the audit logs
    print("\nRetrieving audit logs...")
    logs = get_audit_logs(
        operation_name="model_training"
    )
    
    print(f"Retrieved {len(logs)} audit log entries")
    
    # Create a report generator
    print("\nGenerating audit report...")
    generator = ReportGenerator()
    
    # Generate HTML report
    output_file = "reports/audit_report.html"
    report_path = generator.generate_audit_report(
        logs=logs,
        output_file=output_file,
        title="Model Training Audit Report",
        include_charts=True,
        additional_context={
            "regulations": ["GDPR", "HIPAA"],
            "data_owner": "Medical Research Department",
            "report_purpose": "Regulatory compliance verification"
        }
    )
    
    print(f"Audit report generated and saved to: {output_file}")
    
    return logs, generator

# Example 4: Customized Report with Logo and CSS
def example_customized_report(compliance_report):
    """
    Generate a customized report with a logo and custom CSS.
    """
    print_header("Example 4: Customized Report with Logo and CSS")
    
    # Define custom CSS
    print("Creating a customized report with custom styling...")
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
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .report-header {
        background-color: #3498db;
        color: white;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 30px;
    }
    
    .report-header h1 {
        color: white;
        margin: 0;
    }
    
    .section {
        background-color: white;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    
    th, td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    
    th {
        background-color: #3498db;
        color: white;
    }
    
    .chart-container {
        background-color: white;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    """
    
    # Create a report generator with custom CSS
    generator = ReportGenerator(custom_css=custom_css)
    
    # Create output directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Generate HTML report
    output_file = "reports/customized_compliance_report.html"
    
    # Try to find a logo or create a simple one for the example
    logo_path = None
    try:
        # Create a simple logo for the example
        plt.figure(figsize=(4, 1))
        plt.text(0.5, 0.5, 'SecureML', fontsize=24, ha='center', va='center', color='#3498db')
        plt.axis('off')
        logo_file = "reports/secureml_logo.png"
        plt.savefig(logo_file, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        logo_path = logo_file
        print(f"Created a simple logo at {logo_file}")
    except Exception as e:
        print(f"Could not create logo: {str(e)}")
    
    report_path = generator.generate_compliance_report(
        report=compliance_report,
        output_file=output_file,
        logo_path=logo_path,
        include_charts=True,
        additional_context={
            "organization": "Example Corporation",
            "department": "Data Science Team",
            "project": "Customer Behavior Analysis",
            "date": datetime.now().strftime("%Y-%m-%d")
        }
    )
    
    print(f"Customized report generated and saved to: {output_file}")
    
    return generator

# Example 5: Combined Compliance and Audit Report
def example_combined_report(compliance_report, audit_logs):
    """
    Generate a combined report with both compliance and audit information.
    """
    print_header("Example 5: Combined Compliance and Audit Report")
    
    print("Creating a combined compliance and audit report...")
    
    # First, create separate reports
    generator = ReportGenerator()
    
    # Create output directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Generate compliance report to a temporary file
    compliance_file = "reports/temp_compliance.html"
    generator.generate_compliance_report(
        report=compliance_report,
        output_file=compliance_file,
        include_charts=True
    )
    
    # Generate audit report to a temporary file
    audit_file = "reports/temp_audit.html"
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
    
    # Extract the main content from each (simple approach)
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
        
        h1, h2, h3 {{
            color: #2c3e50;
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
    combined_file = "reports/combined_report.html"
    with open(combined_file, 'w') as f:
        f.write(combined_html)
    
    print(f"Combined report generated and saved to: {combined_file}")
    
    # Clean up temporary files
    try:
        os.remove(compliance_file)
        os.remove(audit_file)
    except:
        pass
    
    return combined_file

# Example 6: Scheduled Report Generation
def example_scheduled_reporting():
    """
    Demonstrate how to set up scheduled report generation.
    """
    print_header("Example 6: Scheduled Report Generation")
    
    print("This example demonstrates a pattern for scheduled/automated report generation.")
    print("In a real application, you would use a scheduler like cron or Airflow.")
    
    # Simulate a week's worth of audit logs
    print("Creating simulated audit logs for the past week...")
    
    # Get the current date and time
    now = datetime.now()
    
    # Create some simulated logs for each day of the past week
    all_logs = []
    event_types = ["data_access", "model_training", "model_inference", "data_transformation"]
    
    for days_ago in range(7):
        # Calculate the date
        log_date = now - timedelta(days=days_ago)
        
        # Generate 3-10 logs for this day
        num_logs = np.random.randint(3, 11)
        
        for i in range(num_logs):
            # Create a log entry
            event_type = np.random.choice(event_types)
            log_entry = {
                "event_type": event_type,
                "timestamp": log_date.isoformat(),
                "operation_id": f"op_{log_date.strftime('%Y%m%d')}_{i}",
                "operation_name": "daily_ml_operations",
                "user": f"user_{np.random.randint(1, 6)}"
            }
            
            # Add event-specific details
            if event_type == "data_access":
                log_entry.update({
                    "dataset_name": "customer_data",
                    "columns_accessed": ["id", "age", "purchase_history"],
                    "num_records": np.random.randint(100, 1000),
                    "purpose": "Daily model retraining"
                })
            elif event_type == "model_training":
                log_entry.update({
                    "model_type": "GradientBoosting",
                    "dataset_name": "customer_data",
                    "metrics": {
                        "accuracy": round(0.8 + np.random.random() * 0.15, 3),
                        "f1": round(0.75 + np.random.random() * 0.2, 3)
                    }
                })
            elif event_type == "model_inference":
                log_entry.update({
                    "model_id": "daily_prediction_model",
                    "input_data": "new_customers",
                    "predictions_count": np.random.randint(10, 100)
                })
            elif event_type == "data_transformation":
                log_entry.update({
                    "transformation_type": np.random.choice(["normalization", "tokenization", "anonymization"]),
                    "input_data": "raw_customer_data",
                    "output_data": "processed_customer_data"
                })
            
            all_logs.append(log_entry)
    
    print(f"Created {len(all_logs)} simulated audit logs across 7 days")
    
    # Function to generate a weekly report
    def generate_weekly_report(logs, output_dir):
        """Generate a weekly report from logs."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get date range
        timestamps = [datetime.fromisoformat(log["timestamp"]) for log in logs]
        min_date = min(timestamps).strftime("%Y-%m-%d")
        max_date = max(timestamps).strftime("%Y-%m-%d")
        
        # Count events by type
        event_counts = {}
        for log in logs:
            event_type = log["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Count events by user
        user_counts = {}
        for log in logs:
            user = log.get("user", "unknown")
            user_counts[user] = user_counts.get(user, 0) + 1
        
        # Generate report
        generator = ReportGenerator()
        output_file = f"{output_dir}/weekly_report_{min_date}_to_{max_date}.html"
        
        # Create visualizations for the report
        charts = {}
        
        # Event type chart
        plt.figure(figsize=(10, 6))
        plt.bar(event_counts.keys(), event_counts.values(), color='skyblue')
        plt.title('Events by Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to buffer
        from io import BytesIO
        import base64
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        charts["events_by_type_base64"] = base64.b64encode(buffer.read()).decode('utf-8')
        
        # User activity chart
        plt.figure(figsize=(10, 6))
        plt.bar(user_counts.keys(), user_counts.values(), color='lightgreen')
        plt.title('Activity by User')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        charts["activity_by_user_base64"] = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Create a custom HTML template for the weekly report
        custom_template = """<!DOCTYPE html>
<html>
<head>
    <title>Weekly Activity Report</title>
    <style>
        {{ css }}
        .weekly-summary {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="report-header">
        <h1>Weekly Activity Report</h1>
        <p>{{ min_date }} to {{ max_date }}</p>
    </div>
    
    <div class="weekly-summary">
        <h2>Summary</h2>
        <p>Total events: {{ total_events }}</p>
        <p>Date range: {{ min_date }} to {{ max_date }}</p>
        
        <h3>Events by Type:</h3>
        <ul>
            {% for event_type, count in event_counts.items() %}
            <li>{{ event_type }}: {{ count }}</li>
            {% endfor %}
        </ul>
        
        <h3>Activity by User:</h3>
        <ul>
            {% for user, count in user_counts.items() %}
            <li>{{ user }}: {{ count }}</li>
            {% endfor %}
        </ul>
    </div>
    
    <div class="chart-container">
        <h2>Events by Type</h2>
        <img src="data:image/png;base64,{{ charts.events_by_type_base64 }}" alt="Events by Type">
    </div>
    
    <div class="chart-container">
        <h2>Activity by User</h2>
        <img src="data:image/png;base64,{{ charts.activity_by_user_base64 }}" alt="Activity by User">
    </div>
    
    <div class="events-list">
        <h2>Recent Events</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Event Type</th>
                <th>User</th>
                <th>Details</th>
            </tr>
            {% for log in recent_logs %}
            <tr>
                <td>{{ log.timestamp }}</td>
                <td>{{ log.event_type }}</td>
                <td>{{ log.user }}</td>
                <td>
                    <ul>
                    {% for key, value in log.items() %}
                        {% if key not in ['timestamp', 'event_type', 'user'] %}
                        <li><strong>{{ key }}:</strong> {{ value }}</li>
                        {% endif %}
                    {% endfor %}
                    </ul>
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""
        
        # Write the template to a temporary file
        template_file = f"{output_dir}/weekly_report_template.html"
        with open(template_file, 'w') as f:
            f.write(custom_template)
        
        # Set up the template environment
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(output_dir))
        template = env.get_template("weekly_report_template.html")
        
        # Render the template
        html_content = template.render(
            css=generator._get_css(),
            min_date=min_date,
            max_date=max_date,
            total_events=len(logs),
            event_counts=event_counts,
            user_counts=user_counts,
            charts=charts,
            recent_logs=sorted(logs, key=lambda x: x["timestamp"], reverse=True)[:10]
        )
        
        # Write the output file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        # Clean up template file
        try:
            os.remove(template_file)
        except:
            pass
        
        return output_file
    
    # Generate the weekly report
    weekly_report_file = generate_weekly_report(all_logs, "reports")
    print(f"Weekly report generated and saved to: {weekly_report_file}")
    
    # Explain how to schedule this
    print("\nTo automatically generate reports on a schedule:")
    print("1. In Linux/Mac, you can use cron jobs:")
    print("   0 0 * * 0 python /path/to/generate_weekly_report.py")
    print("2. In Windows, you can use Task Scheduler.")
    print("3. For more complex workflows, consider Apache Airflow or similar tools.")
    
    return weekly_report_file

# Main function to run all examples
def main():
    print("SecureML Report Generation Examples")
    print("-----------------------------------")
    
    # Run the examples
    compliance_report, generator = example_basic_compliance_report()
    dataset_report, dataset, _ = example_compliance_report_from_dataset()
    audit_logs, _ = example_audit_report()
    _ = example_customized_report(compliance_report)
    _ = example_combined_report(compliance_report, audit_logs)
    _ = example_scheduled_reporting()
    
    print("\nAll report generation examples completed.")
    print("Reports are saved in the 'reports' directory.")

if __name__ == "__main__":
    main() 