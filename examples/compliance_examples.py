import pandas as pd
from secureml.compliance import ComplianceAuditor, check_compliance

# Sample Data
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
    },
    {
        'id': 2, 
        'name': 'Bob Johnson', 
        'age': 45, 
        'zipcode': '12345', 
        'diagnosis': 'Diabetes', 
        'email': 'b.johnson@email.net', 
        'income': 85000, 
        'phone': '555-5678',
        'consent_date': '2024-01-02',
        'data_storage_location': 'EU'
    }
]

df = pd.DataFrame(data_list)

# Example 1: Basic Compliance Check
print("Example 1: Basic GDPR Compliance Check")
report = check_compliance(
    data=df,
    regulation="GDPR",
    max_samples=100
)
print("\nCompliance Report:")
print(report)
print("\n" + "="*50 + "\n")

# Example 2: Using ComplianceAuditor for Dataset Audit
print("Example 2: Dataset Audit with ComplianceAuditor")
auditor = ComplianceAuditor(regulation="GDPR", log_dir="audit_logs")
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
print("\nDataset Audit Report:")
print(dataset_report)
print("\n" + "="*50 + "\n")

# Example 3: Model Compliance Audit
print("Example 3: Model Compliance Audit")
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
print("\nModel Audit Report:")
print(model_report)
print("\n" + "="*50 + "\n")

# Example 4: Full Pipeline Audit
print("Example 4: Full Pipeline Audit")
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

print("\nPipeline Audit Results:")
for component, report in pipeline_report.items():
    print(f"\n{component.upper()} Report:")
    print(report)
print("\n" + "="*50 + "\n")

# Example 5: Generate PDF Report
print("Example 5: Generate PDF Report")
try:
    pdf_path = auditor.generate_pdf(
        audit_result=pipeline_report,
        output_file="compliance_report.pdf",
        title="Patient Records Pipeline Compliance Audit",
        logo_path="hospital_logo.png"  # Optional
    )
    print(f"\nPDF report generated at: {pdf_path}")
except ImportError as e:
    print("\nCouldn't generate PDF report: WeasyPrint dependency missing.")
    print("To use PDF generation, install WeasyPrint with 'pip install secureml[pdf]'")
    print("On Windows, you'll also need to install GTK libraries. See:")
    print("https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows")
except OSError as e:
    print(f"\nCouldn't generate PDF report: Required system libraries missing.")
    print("On Windows, WeasyPrint requires GTK libraries. See installation guide:")
    print("https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows")
    print(f"Error details: {str(e)}") 