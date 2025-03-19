# SecureML Examples

This directory contains usage examples for the SecureML library. These examples demonstrate the core features of SecureML, including data anonymization, privacy-preserving training, compliance checking, and synthetic data generation.

## Basic Usage Example

The `basic_usage.py` script demonstrates the core functionality of SecureML:

```bash
python basic_usage.py
```

This script:
1. Creates a sample dataset with sensitive information
2. Anonymizes the dataset using k-anonymity
3. Trains a neural network with differential privacy
4. Checks GDPR compliance
5. Generates synthetic data

## API Server Example

The `api_server.py` script shows how to create a REST API that provides SecureML functionality:

```bash
python api_server.py
```

This starts a FastAPI server with the following endpoints:
- `/anonymize` - Anonymize a dataset
- `/synthetic` - Generate synthetic data
- `/compliance` - Check compliance with privacy regulations

You can then interact with the API using tools like curl or Postman, or visit [http://localhost:8000/docs](http://localhost:8000/docs) to view the interactive API documentation.

### Example API Request

Here's an example request to the anonymization endpoint:

```bash
curl -X POST "http://localhost:8000/anonymize" \
  -H "Content-Type: application/json" \
  -d '{
    "columns": [
      {"name": "name", "data": ["John Doe", "Jane Smith", "Bob Johnson"]},
      {"name": "email", "data": ["john@example.com", "jane@example.com", "bob@example.com"]},
      {"name": "age", "data": [30, 25, 45]},
      {"name": "income", "data": [50000, 60000, 70000]}
    ],
    "sensitive_columns": ["name", "email", "income"],
    "method": "k-anonymity",
    "k": 2
  }'
```

## Running the Examples

To run these examples, you'll need to install the SecureML library and its dependencies:

```bash
# From the root directory of the project
pip install -e .
```

Additionally, for the API server example, you'll need to install FastAPI and Uvicorn:

```bash
pip install fastapi uvicorn
``` 