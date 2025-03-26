"""
Shared pytest fixtures for SecureML tests.

This module contains fixtures that can be used across all test files.
"""

import os
import sys
import tempfile
from typing import Dict, List, Any

import pandas as pd
import pytest
import numpy as np


# Define a minimal mock for Flower (fl)
class MockFlower:
    class client:
        class Client:
            pass  # Minimal implementation for type annotation

    class server:
        class Server:
            pass  # For -> fl.server.Server    


# Apply the mock at session start
def pytest_sessionstart(session):
    fl_mock = MockFlower()
    sys.modules["flwr"] = fl_mock
    import secureml.federated

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


@pytest.fixture
def sample_data():
    """
    Create a sample DataFrame for testing anonymization functionality.
    
    Returns:
        pd.DataFrame: A DataFrame with various data types including potentially
                     sensitive information.
    """
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Create a sample dataset
    n_rows = 50
    data = pd.DataFrame({
        "name": [f"Person {i}" for i in range(n_rows)],
        "email": [f"person{i}@example.com" for i in range(n_rows)],
        "age": np.random.randint(18, 80, n_rows),
        "zip_code": np.random.choice(["12345", "23456", "34567", "45678", "56789"], n_rows),
        "income": np.random.normal(70000, 15000, n_rows).astype(int),
        "credit_card": [f"**** **** **** {i:04d}" for i in range(n_rows)],
        "medical_condition": np.random.choice(
            ["None", "Diabetes", "Hypertension", "Asthma", "Allergies"], n_rows
        ),
        "purchase_amount": np.random.exponential(100, n_rows),
        "timestamp": pd.date_range(start="2020-01-01", periods=n_rows),
    })
    
    return data


@pytest.fixture
def sample_data_dict(sample_data):
    """
    Convert the sample DataFrame to a list of dictionaries for testing.
    
    Args:
        sample_data: The sample DataFrame fixture
        
    Returns:
        list: A list of dictionaries representing the rows of the DataFrame
    """
    return sample_data.to_dict("records")


@pytest.fixture
def time_series_data():
    """
    Create a sample time series DataFrame for testing.
    
    Returns:
        pd.DataFrame: A DataFrame representing time series data for multiple entities
    """
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=30)
    data = pd.DataFrame({
        "date": np.repeat(dates, 5),
        "user_id": np.tile(range(5), 30),
        "value": np.random.normal(100, 20, 30*5)
    })
    return data


@pytest.fixture
def hierarchical_data():
    """
    Create a sample DataFrame with hierarchical categorical data.
    
    Returns:
        pd.DataFrame: A DataFrame with hierarchical categorical columns
    """
    np.random.seed(42)
    n_rows = 100
    
    # Define hierarchies
    city_to_state = {
        "New York": "NY", "Buffalo": "NY", "Albany": "NY",
        "Los Angeles": "CA", "San Francisco": "CA", "San Diego": "CA",
        "Chicago": "IL", "Springfield": "IL", "Peoria": "IL",
        "Miami": "FL", "Orlando": "FL", "Tampa": "FL"
    }
    
    state_to_region = {
        "NY": "Northeast", "MA": "Northeast", "CT": "Northeast",
        "CA": "West", "OR": "West", "WA": "West",
        "IL": "Midwest", "OH": "Midwest", "MI": "Midwest",
        "FL": "South", "GA": "South", "TX": "South"
    }
    
    # Generate city data and create corresponding state and region
    cities = list(city_to_state.keys())
    city_data = np.random.choice(cities, n_rows)
    state_data = [city_to_state[city] for city in city_data]
    region_data = [state_to_region[state] for state in state_data]
    
    # Create the DataFrame
    data = pd.DataFrame({
        "id": range(n_rows),
        "city": city_data,
        "state": state_data,
        "region": region_data,
        "income": np.random.normal(70000, 15000, n_rows).astype(int),
        "age_group": np.random.choice(["18-25", "26-35", "36-45", "46-55", "56+"], n_rows)
    })
    
    return data


@pytest.fixture
def taxonomy_hierarchies():
    """
    Provide hierarchical taxonomies for generalization.
    
    Returns:
        dict: A dictionary of hierarchical taxonomies
    """
    # Location hierarchy
    location_hierarchy = {
        "city_to_state": {
            "New York": "NY", "Buffalo": "NY", "Albany": "NY",
            "Los Angeles": "CA", "San Francisco": "CA", "San Diego": "CA",
            "Chicago": "IL", "Springfield": "IL", "Peoria": "IL",
            "Miami": "FL", "Orlando": "FL", "Tampa": "FL"
        },
        "state_to_region": {
            "NY": "Northeast", "MA": "Northeast", "CT": "Northeast",
            "CA": "West", "OR": "West", "WA": "West",
            "IL": "Midwest", "OH": "Midwest", "MI": "Midwest",
            "FL": "South", "GA": "South", "TX": "South"
        }
    }
    
    # Age group hierarchy
    age_hierarchy = {
        "18-25": "Young Adult",
        "26-35": "Young Adult",
        "36-45": "Middle Aged",
        "46-55": "Middle Aged",
        "56+": "Senior"
    }
    
    # Medical condition hierarchy
    medical_hierarchy = {
        "Type 1 Diabetes": "Diabetes",
        "Type 2 Diabetes": "Diabetes",
        "Gestational Diabetes": "Diabetes",
        "Diabetes": "Endocrine Disorder",
        "Hypertension": "Cardiovascular Disorder",
        "Arrhythmia": "Cardiovascular Disorder",
        "Asthma": "Respiratory Disorder",
        "COPD": "Respiratory Disorder",
        "Allergies": "Immune Disorder",
        "Eczema": "Immune Disorder"
    }
    
    return {
        "location": location_hierarchy,
        "age": age_hierarchy,
        "medical": medical_hierarchy
    }


@pytest.fixture
def empty_data():
    """
    Create an empty DataFrame for edge case testing.
    
    Returns:
        pd.DataFrame: An empty DataFrame
    """
    return pd.DataFrame()


@pytest.fixture
def single_row_data():
    """
    Create a DataFrame with a single row for edge case testing.
    
    Returns:
        pd.DataFrame: A DataFrame with a single row
    """
    return pd.DataFrame({
        "name": ["John Doe"],
        "email": ["john@example.com"],
        "age": [35],
        "income": [85000]
    }) 