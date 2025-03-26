<p align="center">
  <img src="https://github.com/scimorph/secureml/blob/master/secureml_logo.png" alt="SecureML Logo" width="500">
</p>

# SecureML

SecureML is an open-source Python library that integrates with popular machine learning frameworks like TensorFlow and PyTorch. It provides developers with easy-to-use utilities to ensure that AI agents handle sensitive data in compliance with data protection regulations.

## Key Features

- **Data Anonymization Utilities**:
  - K-anonymity implementation with adaptive generalization
  - Pseudonymization with format-preserving encryption
  - Configurable data masking with statistical property preservation
  - Hierarchical data generalization with taxonomy support
  - Automatic sensitive data detection
- **Privacy-Preserving Training Methods**: 
  - Differential privacy integration with PyTorch (via Opacus) and TensorFlow (via TF Privacy)
  - Federated learning with Flower, allowing training on distributed data without centralization
  - Support for secure aggregation and privacy-preserving federated learning
- **Compliance Checkers**: Tools to analyze datasets and model configurations for potential privacy risks
- **Synthetic Data Generation**: Utilities to create synthetic datasets that mimic real data
- **Regulation-Specific Presets**: 
  - Pre-configured YAML settings aligned with major regulations (GDPR, CCPA, HIPAA)
  - Detailed compliance requirements for each regulation
  - Customizable identifiers for personal data and sensitive information
  - Integration with compliance checking functionality
- **Audit Trails and Reporting**: Automatic logging of privacy measures and model decisions

## Installation
> **Disclaimer**: Due to Tensorflow-privacy compatibility issues, SecureML is only available up to Python 3.11. We will update as soon as Tensorflow-privacy releases a version compatible to Python 3.12+

With pip (Python 3.9 - 3.11):
```bash
pip install secureml
```
### Optional Dependencies

SecureML can generate PDF compliance reports if WeasyPrint is installed:

```bash
pip install secureml[pdf]
```

## Quick Start

### Basic Usage

```python
from secureml import anonymize, privacy

# Load your dataset
import pandas as pd
data = pd.read_csv('path/to/your/dataset.csv')

# Anonymize sensitive data
anonymized_data = anonymize.k_anonymize(data,
sensitive_columns=["medical_condition"],
quasi_identifiers=["age", "zipcode", "gender"],
k=5)

# Check if your dataset meets privacy requirements
compliance_report = privacy.check_compliance(data, regulation="GDPR")
print(compliance_report)
```

### Command Line Interface

SecureML includes a command-line interface (CLI) that provides access to its core functionality directly from your terminal. After installing the package, you can use the `secureml` command to perform common privacy and compliance tasks.

#### Basic Usage

```bash
# Check the installed version
secureml --version

# See available commands
secureml --help
```

#### Data Anonymization

Apply k-anonymity to a dataset:

```bash
secureml anonymization k-anonymize data.csv anonymized.csv \
  --quasi-id age --quasi-id zipcode \
  --sensitive medical_condition \
  --k 5
```

#### Compliance Checking

Check a dataset for compliance with privacy regulations:

```bash
# Basic compliance check
secureml compliance check data.csv --regulation GDPR

# Advanced compliance check with metadata and model configuration
secureml compliance check data.csv \
  --regulation HIPAA \
  --metadata metadata.json \
  --model-config model_config.json \
  --output report.html \
  --format html
```

#### Synthetic Data Generation

Generate synthetic data based on real data patterns:

```bash
# Generate synthetic data using statistical modeling
secureml synthetic generate real_data.csv synthetic_data.csv \
  --method statistical \
  --samples 5000

# Generate synthetic data using advanced SDV Copula model
secureml synthetic generate real_data.csv synthetic_data.csv \
  --method sdv-copula \
  --samples 5000 \
  --sensitive name --sensitive email

# Use automatic sensitive data detection with custom parameters
secureml synthetic generate real_data.csv synthetic_data.csv \
  --method sdv-copula \
  --auto-detect-sensitive \
  --sensitivity-confidence 0.7 \
  --sensitivity-sample-size 200
```

#### Working with Regulation Presets

List and explore built-in regulation presets:

```bash
# List available presets
secureml presets list

# View a specific preset
secureml presets show gdpr

# Extract a specific field from a preset
secureml presets show gdpr --field personal_data_identifiers

# Save a preset to a file
secureml presets show hipaa --output hipaa_preset.json
```
### Isolated Environments

SecureML uses isolated virtual environments to manage dependencies with conflicts. In particular, tensorflow-privacy requires packaging ~= 22.0, while other dependencies need packaging 24.0.

When you use TensorFlow Privacy functionality through SecureML, the library automatically creates and manages a separate virtual environment for this purpose. The first time you use TensorFlow Privacy, there might be a delay as SecureML sets up this environment.

#### Managing Isolated Environments

SecureML provides CLI commands to manage isolated environments:

```bash
# Set up the TensorFlow Privacy environment in advance
secureml environments setup-tf-privacy

# Force recreation of the environment (useful for troubleshooting)
secureml environments setup-tf-privacy --force

# Check the status of isolated environments
secureml environments info
```

#### How Isolated Environments Work

1. **Automatic Management**: When you use functionality that requires TensorFlow Privacy, SecureML automatically creates and manages an isolated Python virtual environment.

2. **Location**: The environment is created at `~/.secureml/tf_privacy_venv` by default.

3. **Communication**: SecureML uses a secure, serialized JSON-based communication protocol to transfer data and model information between environments.

4. **Dependencies**: The isolated environment includes all necessary dependencies like TensorFlow, TensorFlow Privacy, NumPy, and Pandas.

#### Using TensorFlow Privacy in Your Code

When you use the `differentially_private_train` function with the "tensorflow" framework, SecureML automatically handles the transition to the isolated environment:

```python
from secureml import privacy
import tensorflow as tf

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train with differential privacy
private_model = privacy.differentially_private_train(
    model=model,
    data=training_data,
    epsilon=1.0,
    delta=1e-5,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    framework="tensorflow"
)

# The model is trained with differential privacy guarantees
predictions = private_model.predict(test_data)
```

The training happens in the isolated environment while seamlessly returning the trained model to your main environment.

### Compliance Checking with Regulation Presets

SecureML includes built-in presets for major regulations (GDPR, CCPA, HIPAA) that define the compliance requirements specific to each regulation:

```python
import pandas as pd
from secureml import check_compliance
from secureml.presets import list_available_presets, load_preset, get_preset_field

# List available regulation presets
print(list_available_presets())  # ['ccpa', 'gdpr', 'hipaa']

# Load and examine a preset
gdpr_preset = load_preset('gdpr')
print(gdpr_preset['regulation']['name'])  # 'GDPR'
print(gdpr_preset['regulation']['description'])  # 'European Union General Data Protection Regulation'

# Access specific fields using dot notation
personal_identifiers = get_preset_field('gdpr', 'personal_data_identifiers')
print(personal_identifiers)  # ['name', 'email', 'phone', ...]

# Check a dataset for compliance with a specific regulation
df = pd.DataFrame({
    'name': ['John Doe', 'Jane Smith'],
    'email': ['john@example.com', 'jane@example.com'],
    'medical_condition': ['Asthma', 'Diabetes']
})

# Add metadata about the dataset
metadata = {
    'data_storage_location': 'US-East',
    'consent_obtained': True,
    'data_encrypted': False
}

# Model configuration (if available)
model_config = {
    'supports_forget_request': False,
    'access_controls': True
}

# Perform the compliance check
report = check_compliance(
    {'data': df, **metadata},
    model_config=model_config,
    regulation='GDPR'
)

# Check the results
print(report)
if report.has_issues():
    print("Compliance issues found!")
    for issue in report.issues:
        print(f"{issue['severity'].upper()}: {issue['issue']}")
        print(f"Recommendation: {issue['recommendation']}")
```

### Privacy-Preserving Machine Learning

#### With TensorFlow

```python
import tensorflow as tf
from secureml.tensorflow import PrivacyPreservingModel

# Create a privacy-preserving model with differential privacy
model = PrivacyPreservingModel(
tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
]),
    epsilon=3.0, # Privacy budget
    delta=1e-5 # Privacy relaxation parameter
)

# Train with privacy guarantees
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

#### With PyTorch

```python
import torch
import torch.nn as nn
from secureml.torch import private_training

# Define your model
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Enable privacy-preserving training
private_model = private_training.make_private(
    model,
    epsilon=3.0,
    delta=1e-5,
    max_grad_norm=1.0
)

# Train with privacy guarantees
private_training.train(private_model, train_loader, optimizer, epochs=5)
```

### Federated Learning

#### Simulation Mode (for Development)

```python
import torch.nn as nn
from secureml import train_federated
from secureml.federated import FederatedConfig

# Define model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# Function that returns client datasets
def get_client_data():
    return {
        "client1": client1_data,
        "client2": client2_data,
        "client3": client3_data
    }

# Configure federated learning with privacy and security
config = FederatedConfig(
    num_rounds=5,
    use_secure_aggregation=True,
    apply_differential_privacy=True,
    epsilon=1.0
)

# Train with federated learning (simulation mode)
trained_model = train_federated(
    model=model,
    client_data_fn=get_client_data,
    config=config,
    framework="pytorch"  # or "tensorflow", or "auto"
)
```

#### Deployment Mode (Server and Clients)

```python
# On the server
from secureml import start_federated_server
from secureml.federated import FederatedConfig

# Initialize model and configuration
model = create_initial_model()
config = FederatedConfig(
    num_rounds=10,
    min_fit_clients=3,
    use_secure_aggregation=True,
    server_address="0.0.0.0:8080"
)

# Start the server
start_federated_server(model, config)

# -------------------------------------------------------
# On each client
from secureml import start_federated_client

# Initialize model with same architecture as server
model = create_model_architecture()
client_data = load_local_data()

# Start the client
start_federated_client(
    model=model,
    data=client_data,
    server_address="server_ip:8080",
    apply_differential_privacy=True,
    epsilon=2.0
)
```

#### Advanced Weight Update Strategies

SecureML provides sophisticated weight update mechanisms for federated learning to improve convergence and stability. These strategies can be especially valuable in challenging federated scenarios with heterogeneous data distributions or when training complex models.

```python
from secureml import train_federated
from secureml.federated import FederatedConfig

# Configure federated learning with Exponential Moving Average (EMA) weight updates
ema_config = FederatedConfig(
    num_rounds=10,
    # Weight update configuration
    weight_update_strategy="ema",  # Use exponential moving average
    weight_mixing_rate=0.5,  # 50% mix of new weights, 50% of old weights
    warmup_rounds=2  # Gradually increase mixing rate over first 2 rounds
)

# Configure federated learning with Momentum-based weight updates
momentum_config = FederatedConfig(
    num_rounds=10,
    # Weight update configuration
    weight_update_strategy="momentum",  # Use momentum-based updates
    weight_mixing_rate=0.1,  # Small update step size
    weight_momentum=0.9,  # High momentum coefficient
    # Constrain updates to prevent instability
    apply_weight_constraints=True,
    max_weight_change=0.3  # Maximum 30% change in any weight
)

# Train with preferred weight update strategy
trained_model = train_federated(
    model=model,
    client_data_fn=get_client_data,
    config=momentum_config  # Use the momentum configuration
)
```

##### Weight Update Strategies

SecureML supports three different strategies for updating model weights in federated learning:

1. **Direct Updates** (`weight_update_strategy="direct"`): The simplest strategy, where client models directly adopt the weights received from the server. This is the classic federated learning approach.

2. **Exponential Moving Average (EMA)** (`weight_update_strategy="ema"`): A weighted average between old and new weights. This creates smoother updates and can improve training stability:
   ```
   updated_weight = (1 - mixing_rate) * old_weight + mixing_rate * new_weight
   ```

3. **Momentum-Based Updates** (`weight_update_strategy="momentum"`): Uses a momentum term to accelerate training and avoid local minima:
   ```
   momentum_update = momentum * previous_update + mixing_rate * (new_weight - old_weight)
   updated_weight = old_weight + momentum_update
   ```

##### Key Configuration Parameters

- **`weight_mixing_rate`**: Controls how much of the new weights to incorporate (0.0 to 1.0). Lower values make smaller, more conservative updates.

- **`weight_momentum`**: For momentum strategy, determines how much previous updates influence current ones (typically 0.9 to 0.99).

- **`warmup_rounds`**: Number of initial rounds with gradually increasing mixing rates. Useful for stabilizing early training.

- **`apply_weight_constraints`**: When `True`, prevents any weight from changing too dramatically in a single update.

- **`max_weight_change`**: Maximum relative change allowed in any weight when constraints are enabled (e.g., 0.2 = 20% maximum change).

##### Choosing a Strategy

- Use **Direct** for simpler models and homogeneous data distributions.
- Use **EMA** for improved stability and when working with sensitive data that might create noisy updates.
- Use **Momentum** for faster convergence on complex problems and when clients have heterogeneous data distributions.

For maximum stability, especially with differential privacy enabled, combine momentum with weight constraints:

```python
config = FederatedConfig(
    num_rounds=20,
    apply_differential_privacy=True,
    epsilon=1.0,
    weight_update_strategy="momentum", 
    weight_momentum=0.95,
    apply_weight_constraints=True,
    max_weight_change=0.25
)
```

### Synthetic Data Generation

SecureML provides multiple approaches to synthetic data generation, from simple Faker-based methods to advanced statistical modeling with the Synthetic Data Vault (SDV).

#### Basic Usage

```python
from secureml.synthetic import generate_synthetic_data
import pandas as pd

# Load your real data
real_data = pd.read_csv('path/to/your/dataset.csv')

# Simple synthetic data generation (using Faker for sensitive columns)
synthetic_data = generate_synthetic_data(
    template=real_data,
    num_samples=1000,
    method="simple"
)

# Or use statistical modeling to preserve relationships between variables
statistical_synthetic = generate_synthetic_data(
    template=real_data,
    num_samples=1000,
    method="statistical"
)
```

#### Advanced Statistical Modeling with SDV

For more complex use cases where preserving statistical relationships and distributions is critical:

```python
from secureml.synthetic import generate_synthetic_data
import pandas as pd

# Load your real data
real_data = pd.read_csv('path/to/your/dataset.csv')

# Generate synthetic data using SDV's GaussianCopula model
# This preserves statistical relationships between variables
synthetic_data = generate_synthetic_data(
    template=real_data,
    num_samples=1000,
    method="sdv-copula",
    sensitive_columns=["name", "email", "phone_number"]
)

# For more complex patterns, use GAN-based approaches
gan_synthetic = generate_synthetic_data(
    template=real_data,
    num_samples=1000,
    method="sdv-ctgan"
)

# Add constraints to ensure synthetic data follows business rules
constraints = [
    {"type": "unique", "columns": ["id"]},
    {"type": "inequality", "low_column": "start_date", "high_column": "end_date"}
]

constrained_synthetic = generate_synthetic_data(
    template=real_data,
    num_samples=1000,
    method="sdv-copula",
    constraints=constraints
)

# Use automatic sensitive data detection with custom parameters
auto_detected_synthetic = generate_synthetic_data(
    template=real_data,
    num_samples=1000,
    method="sdv-copula",
    sensitivity_detection={
        "auto_detect": True,
        "confidence_threshold": 0.7,  # Higher confidence threshold for stricter detection
        "sample_size": 200  # Use more samples for better detection accuracy
    }
)

# You can check which columns were detected as sensitive
from secureml.synthetic import _identify_sensitive_columns

detected_sensitive_columns = _identify_sensitive_columns(
    real_data,
    sample_size=200,
    confidence_threshold=0.7
)
print("Detected sensitive columns:", detected_sensitive_columns)
```

#### Legacy Synthesizer (For backward compatibility)

```python
from secureml.synthetic import TabularSynthesizer

# Initialize the synthesizer
synthesizer = TabularSynthesizer(method="gan")

# Fit to your real data
synthesizer.fit(data)

# Generate synthetic data that preserves statistical properties
synthetic_data = synthesizer.generate(n_samples=1000)
```

### Automated Compliance Reporting

SecureML provides comprehensive audit trails and reporting capabilities to help document privacy measures and model decisions for compliance purposes.

#### Basic Audit Trail Usage

```python
from secureml import AuditTrail

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

# Log a compliance check
audit.log_compliance_check(
    check_type="data_minimization",
    regulation="GDPR",
    result=True,
    details={"fields_removed": ["patient_name", "ssn", "address"]}
)

# Close the audit trail when done
audit.close(status="completed")
```

#### Auditing Functions

SecureML provides a decorator to automatically create audit trails for functions:

```python
from secureml import audit_function

@audit_function(regulations=["GDPR", "HIPAA"])
def process_patient_data(data, anonymize=True):
    # Process the data
    if anonymize:
        # Anonymize the data
        return anonymized_data
    return processed_data
```

#### Comprehensive ML Pipeline Auditing

The `ComplianceAuditor` class provides an integrated approach to auditing an entire ML pipeline:

```python
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
```

#### Generating Reports from Audit Logs

You can also generate reports from audit logs after they've been created:

```python
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
```

#### Enhancing ComplianceReport with Report Generation

The basic `ComplianceReport` returned by `check_compliance()` now includes report generation capabilities:

```python
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
```

## Documentation

For detailed documentation, examples, and API reference, visit [our documentation](https://secureml.readthedocs.io).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or Issue.
Our focus is expanding supported legislations beyond GDPR, CCPA, and HIPAA. You can help us with that!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.