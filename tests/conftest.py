"""
Shared pytest fixtures for SecureML tests.

This module contains fixtures that can be used across all test files.
"""

import os
import tempfile
from typing import Dict, List, Any

import pandas as pd
import pytest
import numpy as np


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame with various data types for testing."""
    return pd.DataFrame({
        "name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Davis"],
        "email": ["john@example.com", "jane@example.com", "bob@example.com", 
                 "alice@example.com", "charlie@example.com"],
        "age": [35, 28, 42, 31, 25],
        "zip_code": ["12345", "23456", "34567", "45678", "56789"],
        "income": [85000, 72000, 95000, 63000, 78000],
        "purchase": [120.50, 85.75, 210.25, 55.30, 150.00],
        "medical_condition": ["Hypertension", "Diabetes", "Asthma", "Migraine", "None"],
        "credit_card": ["4111-1111-1111-1111", "5500-0000-0000-0004", "3400-0000-0000-009", 
                        "6011-0000-0000-0004", "3088-0000-0000-0009"]
    })


@pytest.fixture
def sample_data_dict(sample_data) -> Dict[str, Any]:
    """Convert sample DataFrame to a dictionary format."""
    return sample_data.to_dict("records")


@pytest.fixture
def temp_log_dir() -> str:
    """Create a temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def medical_data() -> pd.DataFrame:
    """Create a sample medical dataset for HIPAA compliance testing."""
    return pd.DataFrame({
        "patient_id": ["P001", "P002", "P003", "P004", "P005"],
        "name": ["John Doe", "Jane Smith", "Robert Brown", "Emily Wilson", "Michael Lee"],
        "dob": ["1980-05-15", "1975-10-23", "1990-03-08", "1988-07-12", "1965-11-30"],
        "ssn": ["123-45-6789", "234-56-7890", "345-67-8901", "456-78-9012", "567-89-0123"],
        "address": [
            "123 Main St, Boston, MA 02115",
            "456 Oak Ave, Chicago, IL 60611",
            "789 Pine Rd, Seattle, WA 98101",
            "101 Elm Ln, Austin, TX 78701",
            "202 Cedar Ct, Denver, CO 80202"
        ],
        "phone": ["(617) 555-1234", "(312) 555-2345", "(206) 555-3456", 
                  "(512) 555-4567", "(303) 555-5678"],
        "diagnosis": ["Hypertension", "Type 2 Diabetes", "Asthma", "Migraine", "Arthritis"],
        "medication": ["Lisinopril", "Metformin", "Albuterol", "Sumatriptan", "Ibuprofen"],
        "doctor": ["Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Jones", "Dr. Brown"]
    })


@pytest.fixture
def synthetic_template() -> pd.DataFrame:
    """Create a simple template for synthetic data generation."""
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame({
        "numeric_normal": np.random.normal(50, 10, 100),
        "numeric_uniform": np.random.uniform(0, 100, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "binary": np.random.choice([0, 1], 100),
        "date": pd.date_range(start="2020-01-01", periods=100).astype(str)
    })


@pytest.fixture
def simple_model():
    """Create a simple model for privacy testing."""
    try:
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 1)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        return SimpleModel()
    except ImportError:
        # Fall back to a dictionary representation if PyTorch is not available
        return {
            "type": "SimpleModel",
            "layers": [
                {"type": "Linear", "in_features": 10, "out_features": 20},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": 20, "out_features": 1}
            ]
        } 