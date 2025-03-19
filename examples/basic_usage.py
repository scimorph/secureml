"""
Basic example demonstrating the core functionality of SecureML.

This example shows how to:
1. Anonymize a dataset
2. Train a model with differential privacy
3. Check compliance with GDPR
4. Generate synthetic data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from secureml import (
    anonymize, 
    differentially_private_train, 
    check_compliance,
    generate_synthetic_data
)


def create_sample_data():
    """Create a sample dataset with some sensitive information."""
    np.random.seed(42)
    
    # Create a dataframe with some sensitive columns
    data = pd.DataFrame({
        "user_id": range(1000),
        "name": [f"User {i}" for i in range(1000)],
        "email": [f"user{i}@example.com" for i in range(1000)],
        "age": np.random.randint(18, 80, 1000),
        "income": np.random.normal(50000, 15000, 1000),
        "credit_score": np.random.randint(300, 850, 1000),
        "purchase_amount": np.random.exponential(100, 1000),
        "is_premium": np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
    })
    
    print("Original data sample:")
    print(data.head())
    print(f"Number of records: {len(data)}")
    return data


def define_model():
    """Define a simple neural network model for binary classification."""
    model = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )
    return model


def main():
    # Step 1: Create and prepare sample data
    data = create_sample_data()
    
    # Step 2: Anonymize the dataset
    print("\n--- Anonymizing data ---")
    anonymized_data = anonymize(
        data, 
        method="k-anonymity",
        k=5,
        sensitive_columns=["name", "email", "income", "credit_score"]
    )
    print("\nAnonymized data sample:")
    print(anonymized_data.head())
    
    # Step 3: Prepare features and target for training
    features = anonymized_data[["age", "income", "credit_score", "purchase_amount", "is_premium"]]
    target = (anonymized_data["purchase_amount"] > anonymized_data["purchase_amount"].median()).astype(int)
    
    # Convert to PyTorch tensors
    X = torch.tensor(features.values, dtype=torch.float32)
    y = torch.tensor(target.values, dtype=torch.float32).reshape(-1, 1)
    
    # Create dataset and model
    dataset = TensorDataset(X, y)
    model = define_model()
    
    # Step 4: Train the model with differential privacy
    print("\n--- Training model with differential privacy ---")
    dp_model = differentially_private_train(
        model,
        dataset,
        epsilon=1.0,
        delta=1e-5,
        framework="pytorch",
        batch_size=64,
        epochs=5,
        learning_rate=0.001
    )
    print("Model trained with differential privacy")
    
    # Step 5: Check compliance with GDPR
    print("\n--- Checking GDPR compliance ---")
    model_config = {
        "supports_forget_request": True,
        "architecture": "neural_network",
        "training_method": "differential_privacy"
    }
    
    metadata = {
        "consent_obtained": True,
        "data_storage_location": "EU",
        "data_encrypted": True
    }
    
    # Combine data and metadata for compliance check
    data_with_metadata = {
        "data": anonymized_data,
        **metadata
    }
    
    report = check_compliance(
        data_with_metadata,
        model_config=model_config,
        regulation="GDPR"
    )
    
    print(report)
    
    # Step 6: Generate synthetic data
    print("\n--- Generating synthetic data ---")
    synthetic_data = generate_synthetic_data(
        anonymized_data,
        num_samples=500,
        method="statistical",
        seed=42
    )
    
    print("\nSynthetic data sample:")
    print(synthetic_data.head())
    print(f"Number of synthetic records: {len(synthetic_data)}")


if __name__ == "__main__":
    main() 