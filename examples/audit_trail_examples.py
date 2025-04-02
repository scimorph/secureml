import pandas as pd
from secureml.audit import AuditTrail, audit_function, get_audit_logs
from secureml.reporting import ReportGenerator
import datetime

# Example 1: Basic Audit Trail Creation and Usage
print("Example 1: Basic Audit Trail Creation and Usage")

# Create an audit trail
audit = AuditTrail(
    operation_name="data_preprocessing_example",
    log_dir="audit_logs",
    context={"project": "credit_scoring", "environment": "development"},
    regulations=["GDPR"]
)

# Log sample operations
print("Logging sample operations...")

# Load sample data
data = pd.DataFrame({
    'name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown'],
    'age': [32, 45, 27],
    'income': [65000, 85000, 52000],
    'email': ['alice.s@example.com', 'bob.j@example.com', 'charlie.b@example.com']
})

# Log data access
audit.log_data_access(
    dataset_name="customer_data",
    columns_accessed=list(data.columns),
    num_records=len(data),
    purpose="data_preparation",
    user="analyst_123"
)

# Log data transformation
audit.log_data_transformation(
    transformation_type="anonymization",
    input_data="raw_customer_data",
    output_data="anonymized_customer_data",
    parameters={
        "method": "k-anonymity",
        "k": 3,
        "quasi_identifiers": ["age", "income"]
    }
)

# Log compliance check
audit.log_compliance_check(
    check_type="data_minimization",
    regulation="GDPR",
    result=True,
    details={
        "columns_before": 4,
        "columns_after": 3,
        "columns_removed": ["email"]
    }
)

# Close the audit trail
audit.close(
    status="completed",
    details={
        "execution_time": 2.5,
        "records_processed": len(data)
    }
)

print("Basic audit trail created and closed.")
print("\n" + "="*50 + "\n")

# Example 2: Using the Audit Function Decorator
print("Example 2: Using the Audit Function Decorator")

@audit_function(
    operation_name="model_training_example",
    log_dir="audit_logs",
    regulations=["GDPR", "CCPA"]
)
def train_model(data, model_type="random_forest", **params):
    """Train a machine learning model with audit logging."""
    print(f"Training {model_type} model with {len(data)} records")
    
    # Simulate model training
    accuracy = 0.92
    training_time = 35.7
    
    return {
        "model": f"{model_type}_model",
        "accuracy": accuracy,
        "training_time": training_time
    }

# Call the decorated function
try:
    result = train_model(
        data,
        model_type="gradient_boosting",
        n_estimators=100,
        max_depth=5
    )
    print(f"Model training result: {result}")
except Exception as e:
    print(f"Error during model training: {e}")

print("\n" + "="*50 + "\n")

# Example 3: Retrieving and Analyzing Audit Logs
print("Example 3: Retrieving and Analyzing Audit Logs")

# Get logs for the data preprocessing operation
preprocessing_logs = get_audit_logs(
    operation_name="data_preprocessing_example",
    log_dir="audit_logs"
)

# Print summary of logs
print(f"Retrieved {len(preprocessing_logs)} logs for data preprocessing")
for log in preprocessing_logs:
    print(f"Event: {log.get('event_type')} - Time: {log.get('timestamp')}")

# Get logs for the model training operation
training_logs = get_audit_logs(
    operation_name="model_training_example",
    log_dir="audit_logs"
)

# Print summary of logs
print(f"\nRetrieved {len(training_logs)} logs for model training")
for log in training_logs:
    print(f"Event: {log.get('event_type')} - Time: {log.get('timestamp')}")

print("\n" + "="*50 + "\n")

# Example 4: Generating Reports from Audit Logs
print("Example 4: Generating Reports from Audit Logs")

# Create a report generator
generator = ReportGenerator()

# Get all logs from today
today = datetime.datetime.now().strftime("%Y-%m-%d")
all_logs = get_audit_logs(
    start_time=f"{today}T00:00:00",
    log_dir="audit_logs"
)

# Generate an audit report
try:
    report_path = generator.generate_audit_report(
        logs=all_logs,
        output_file="audit_report.html",
        title="SecureML Operations Audit Report",
        include_charts=True
    )
    print(f"Audit report generated at: {report_path}")
except Exception as e:
    print(f"Error generating report: {e}")

print("\n" + "="*50 + "\n")

print("Audit trail examples complete.") 