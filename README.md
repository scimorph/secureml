<p align="center">
  <img src="https://github.com/scimorph/secureml/blob/master/secureml_logo_2-.png" alt="SecureML Logo" width="500">
</p>

<p align="center">
  <a href="https://github.com/scimorph/secureml/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/scimorph/secureml/ci.yml?branch=master&label=CI/CD&logo=github" alt="CI/CD Status"></a>
  <a href="https://github.com/scimorph/secureml/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/scimorph/secureml/ci.yml?branch=master&label=tests&logo=pytest" alt="Tests Status"></a>
  <a href="https://pypi.org/project/secureml/"><img src="https://img.shields.io/pypi/v/secureml.svg" alt="PyPI Version"></a>
  <a href="https://github.com/scimorph/secureml/blob/master/LICENSE"><img src="https://img.shields.io/github/license/scimorph/secureml" alt="License"></a>
  <img src="https://img.shields.io/pypi/pyversions/secureml.svg" alt="Python Versions">
</p>

<h3 align="center">
  <a href="https://secureml.readthedocs.io/en/latest/index.html">Documentation</a>
</h3>

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
- **Synthetic Data Generation**: 
  - Multiple generation methods including statistical modeling, GANs, and copulas
  - SDV integration with Gaussian Copula, CTGAN, and TVAE synthesizers
  - Automatic sensitive data detection and special handling
  - Preservation of statistical properties and correlations between variables
  - Support for mixed data types (numeric, categorical, datetime)
  - Configurable privacy-utility tradeoff controls
  - Tabular data synthesis with relation preservation
- **Regulation-Specific Presets**: 
  - Pre-configured YAML settings aligned with major regulations (GDPR, CCPA, HIPAA, LGPD)
  - Detailed compliance requirements for each regulation
  - Customizable identifiers for personal data and sensitive information
  - Integration with compliance checking functionality
- **Audit Trails and Reporting**: 
  - Comprehensive audit logging of data access, transformations, and model operations
  - Detailed event tracking for privacy-related operations with timestamps and contexts
  - Function-level auditing through decorators
  - Automated compliance reports in HTML and PDF formats
  - Visual dashboards with charts showing privacy metrics and event distributions
  - Integration with compliance checkers for continuous monitoring

## Installation

With pip (Python 3.11-3.12):
```bash
pip install secureml
```
### Optional Dependencies

```bash
# For generating PDF reports for compliance and audit trails
pip install secureml[pdf]

# For secure key management with HashiCorp Vault
pip install secureml[vault]

# For all optional components
pip install secureml[pdf,vault]
```

## Quick Start

### Data Anonymization

Anonymizing a dataset to comply with privacy regulations:

```python
import pandas as pd
from secureml import anonymize

# Load your dataset
data = pd.DataFrame({
    "name": ["John Doe", "Jane Smith", "Bob Johnson"],
    "age": [32, 45, 28],
    "email": ["john.doe@example.com", "jane.smith@example.com", "bob.j@example.com"],
    "ssn": ["123-45-6789", "987-65-4321", "456-78-9012"],
    "zip_code": ["10001", "94107", "60601"],
    "income": [75000, 82000, 65000]
})
    
# Anonymize using k-anonymity
anonymized_data = anonymize(
    data,
    method="k-anonymity",
    k=2,
        sensitive_columns=["name", "email", "ssn"]
    )
    
    print(anonymized_data)
```

### Compliance Checking with Regulation Presets

SecureML includes built-in presets for major regulations (GDPR, CCPA, HIPAA, LGPD) that define the compliance requirements specific to each regulation:

```python
import pandas as pd
from secureml import check_compliance
    
# Load your dataset
data = pd.read_csv("your_dataset.csv")
    
# Model configuration
model_config = {
    "model_type": "neural_network",
    "input_features": ["age", "income", "zip_code"],
    "output": "purchase_likelihood",
    "training_method": "standard_backprop"
}
    
# Check compliance with GDPR
report = check_compliance(   
    data=data,
    model_config=model_config,
    regulation="GDPR"
)
    
# View compliance issues
if report.has_issues():
    print("Compliance issues found:")
    for issue in report.issues:
        print(f"- {issue['component']}: {issue['issue']} ({issue['severity']})")
        print(f"  Recommendation: {issue['recommendation']}")

```

### Privacy-Preserving Machine Learning

Train a model with differential privacy guarantees:

```python
import torch.nn as nn
import pandas as pd
from secureml import differentially_private_train
    
# Create a simple PyTorch model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
    nn.Softmax(dim=1)
)
    
# Load your dataset
data = pd.read_csv("your_dataset.csv")
    
# Train with differential privacy
private_model = differentially_private_train(
    model=model,
    data=data,
    epsilon=1.0,  # Privacy budget
    delta=1e-5,   # Privacy delta parameter
    epochs=10,
    batch_size=64
)
```

### Synthetic Data Generation

Generate synthetic data that maintains the statistical properties of the original data:

```python
import pandas as pd
from secureml import generate_synthetic_data
    
# Load your dataset
data = pd.read_csv("your_dataset.csv")
    
# Generate synthetic data
synthetic_data = generate_synthetic_data(
    template=data,
    num_samples=1000,
    method="statistical",  # Options: simple, statistical, sdv-copula, gan
    sensitive_columns=["name", "email", "ssn"]
)
    
print(synthetic_data.head())
```

## Documentation

For detailed documentation, examples, and API reference, visit [our documentation](https://secureml.readthedocs.io).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or Issue.
Our focus is expanding supported legislations beyond GDPR, CCPA, HIPAA, and LGPD. You can help us with that!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.