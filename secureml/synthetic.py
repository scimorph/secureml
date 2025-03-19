"""
Synthetic data generation functionality for SecureML.

This module provides tools to create synthetic datasets that mimic
the statistical properties of real data without containing sensitive information.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from faker import Faker

# Add SDV imports
try:
    from sdv.single_table import (
        GaussianCopulaSynthesizer,
        CTGANSynthesizer,
        TVAESynthesizer
    )
    from sdv.metadata import SingleTableMetadata
    from sdv.constraints import FixedCombinations, Inequality, Unique
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False


def generate_synthetic_data(
    template: Union[pd.DataFrame, Dict[str, Any]],
    num_samples: int = 100,
    method: str = "simple",
    sensitive_columns: Optional[List[str]] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate synthetic data based on a template dataset.

    Args:
        template: Original dataset to mimic (DataFrame) or schema specification (Dict)
        num_samples: Number of synthetic samples to generate
        method: Generation method: 'simple', 'statistical', 'sdv-copula', 
               'sdv-ctgan', 'sdv-tvae', 'gan', or 'copula'
        sensitive_columns: Columns that contain sensitive data and need special handling
        seed: Random seed for reproducibility
        **kwargs: Additional parameters for specific generation methods

    Returns:
        DataFrame containing synthetic data

    Raises:
        ValueError: If an unsupported generation method is specified
        ImportError: If an SDV method is requested but SDV is not installed
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        
    # Handle different types of templates
    if isinstance(template, dict):
        # Template is a schema specification
        schema = template
        # Create empty DataFrame with the specified columns
        if "columns" in schema:
            columns = schema["columns"]
            template_df = pd.DataFrame(columns=columns.keys())
            # Set data types based on schema
            for col, col_type in columns.items():
                if col_type == "int":
                    template_df[col] = template_df[col].astype("int64")
                elif col_type == "float":
                    template_df[col] = template_df[col].astype("float64")
                elif col_type == "bool":
                    template_df[col] = template_df[col].astype("bool")
                elif col_type == "category":
                    template_df[col] = template_df[col].astype("category")
                # Add other types as needed
        else:
            raise ValueError("Schema must include 'columns' specification")
    else:
        # Template is a DataFrame
        template_df = template.copy()
        
    # If no sensitive columns specified, try to identify them
    if sensitive_columns is None:
        sensitive_columns = _identify_sensitive_columns(template_df)
        
    # Check if SDV methods are requested but SDV is not available
    sdv_methods = ["sdv-copula", "sdv-ctgan", "sdv-tvae"]
    if method in sdv_methods and not SDV_AVAILABLE:
        raise ImportError(
            f"Method '{method}' requires the SDV package. "
            f"Please install it with 'pip install sdv'."
        )
        
    # Generate synthetic data using the specified method
    if method == "simple":
        return _generate_simple_synthetic(
            template_df, num_samples, sensitive_columns, **kwargs
        )
    elif method == "statistical":
        return _generate_statistical_synthetic(
            template_df, num_samples, sensitive_columns, **kwargs
        )
    elif method == "sdv-copula":
        return _generate_sdv_synthetic(
            template_df, 
            num_samples, 
            sensitive_columns, 
            synthesizer_type="copula",
            **kwargs
        )
    elif method == "sdv-ctgan":
        return _generate_sdv_synthetic(
            template_df, 
            num_samples, 
            sensitive_columns, 
            synthesizer_type="ctgan",
            **kwargs
        )
    elif method == "sdv-tvae":
        return _generate_sdv_synthetic(
            template_df, 
            num_samples, 
            sensitive_columns, 
            synthesizer_type="tvae",
            **kwargs
        )
    elif method == "gan":
        return _generate_gan_synthetic(
            template_df, num_samples, sensitive_columns, **kwargs
        )
    elif method == "copula":
        return _generate_copula_synthetic(
            template_df, num_samples, sensitive_columns, **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported generation method: {method}. "
            f"Supported methods are 'simple', 'statistical', 'sdv-copula', "
            f"'sdv-ctgan', 'sdv-tvae', 'gan', and 'copula'."
        )


def _identify_sensitive_columns(data: pd.DataFrame) -> List[str]:
    """
    Identify columns that likely contain sensitive data.
    
    This is a simplified version - in a real implementation, this would be 
    more sophisticated.
    
    Args:
        data: The dataset to analyze
        
    Returns:
        A list of column names that likely contain sensitive data
    """
    sensitive_patterns = [
        "name", "email", "address", "phone", "social", "ssn", "birth", "gender",
        "race", "ethnicity", "religion", "politics", "income", "salary", "credit",
        "account", "password", "health", "medical", "diagnosis", "treatment"
    ]
    
    sensitive_columns = []
    for col in data.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in sensitive_patterns):
            sensitive_columns.append(col)
            
    return sensitive_columns


def _generate_simple_synthetic(
    template: pd.DataFrame,
    num_samples: int,
    sensitive_columns: List[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate synthetic data using simple random sampling from observed distributions.
    
    Args:
        template: Template dataset
        num_samples: Number of samples to generate
        sensitive_columns: Columns that contain sensitive data
        **kwargs: Additional parameters
        
    Returns:
        DataFrame containing synthetic data
    """
    synthetic_data = pd.DataFrame(columns=template.columns)
    faker = Faker()
    
    # Generate synthetic data for each column
    for col in template.columns:
        if col in sensitive_columns:
            # For sensitive columns, use faker to generate realistic but fake data
            dtype = template[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                # For numeric columns, generate values within the same range
                min_val = template[col].min()
                max_val = template[col].max()
                # Apply some jitter to avoid exact matches
                jitter = (max_val - min_val) * 0.05
                synthetic_data[col] = np.random.uniform(
                    min_val - jitter, max_val + jitter, num_samples
                )
                if pd.api.types.is_integer_dtype(dtype):
                    synthetic_data[col] = synthetic_data[col].round().astype(int)
            else:
                # For string columns, use appropriate faker providers
                if "name" in col.lower():
                    synthetic_data[col] = [
                        faker.name() for _ in range(num_samples)
                    ]
                elif "email" in col.lower():
                    synthetic_data[col] = [
                        faker.email() for _ in range(num_samples)
                    ]
                elif "address" in col.lower():
                    synthetic_data[col] = [
                        faker.address() for _ in range(num_samples)
                    ]
                elif "phone" in col.lower():
                    synthetic_data[col] = [
                        faker.phone_number() for _ in range(num_samples)
                    ]
                elif "date" in col.lower() or "birth" in col.lower():
                    synthetic_data[col] = [
                        faker.date() for _ in range(num_samples)
                    ]
                else:
                    # Default to random sampling of observed values with replacement
                    synthetic_data[col] = np.random.choice(
                        template[col].dropna(), num_samples, replace=True
                    )
        else:
            # For non-sensitive columns, sample from observed distribution
            if (template[col].nunique() < 10 
                    or pd.api.types.is_categorical_dtype(template[col])):
                # For categorical columns, preserve distribution
                value_counts = template[col].value_counts(normalize=True)
                synthetic_data[col] = np.random.choice(
                    value_counts.index, num_samples, p=value_counts.values
                )
            elif pd.api.types.is_numeric_dtype(template[col]):
                # For numeric columns, sample from normal distribution with same mean/std
                mean = template[col].mean()
                std = template[col].std() if template[col].std() > 0 else 0.1
                synthetic_data[col] = np.random.normal(mean, std, num_samples)
                if pd.api.types.is_integer_dtype(template[col]):
                    synthetic_data[col] = synthetic_data[col].round().astype(int)
            else:
                # For other types, sample from observed values
                synthetic_data[col] = np.random.choice(
                    template[col].dropna(), num_samples, replace=True
                )
    
    return synthetic_data


def _generate_statistical_synthetic(
    template: pd.DataFrame,
    num_samples: int,
    sensitive_columns: List[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate synthetic data using statistical models of the data distribution.
    
    Args:
        template: Template dataset
        num_samples: Number of samples to generate
        sensitive_columns: Columns that contain sensitive data
        **kwargs: Additional parameters
        
    Returns:
        DataFrame containing synthetic data
    """
    # Placeholder - in a real implementation, this would use statistical modeling
    # For now, we'll use a more sophisticated version of the simple approach
    
    # 1. Compute correlation matrix
    numeric_columns = template.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) >= 2:
        corr_matrix = template[numeric_columns].corr()
        
        # 2. Generate correlated numeric data
        means = template[numeric_columns].mean().values
        stds = template[numeric_columns].std().values
        stds = np.where(stds > 0, stds, 0.1)  # Replace zeros with small value
        
        # Generate correlated random variables
        L = np.linalg.cholesky(np.clip(corr_matrix, -0.99, 0.99))
        uncorrelated = np.random.normal(
            size=(num_samples, len(numeric_columns))
        )
        correlated = uncorrelated @ L.T
        
        # Scale to match original mean and standard deviation
        synthetic_numeric = pd.DataFrame(
            correlated * stds + means,
            columns=numeric_columns
        )
        
        # Convert integer columns back to integers
        for col in numeric_columns:
            if pd.api.types.is_integer_dtype(template[col]):
                synthetic_numeric[col] = synthetic_numeric[col].round().astype(int)
                
        # Start with the numeric data
        synthetic_data = synthetic_numeric
        
        # Add categorical and other columns
        remaining_columns = [
            col for col in template.columns if col not in numeric_columns
        ]
        for col in remaining_columns:
            if col in sensitive_columns:
                # Handle sensitive columns with faker (same as in simple method)
                faker = Faker()
                if "name" in col.lower():
                    synthetic_data[col] = [faker.name() for _ in range(num_samples)]
                elif "email" in col.lower():
                    synthetic_data[col] = [faker.email() for _ in range(num_samples)]
                elif "address" in col.lower():
                    synthetic_data[col] = [
                        faker.address() for _ in range(num_samples)
                    ]
                elif "phone" in col.lower():
                    synthetic_data[col] = [
                        faker.phone_number() for _ in range(num_samples)
                    ]
                elif "date" in col.lower() or "birth" in col.lower():
                    synthetic_data[col] = [faker.date() for _ in range(num_samples)]
                else:
                    # Default to random sampling of observed values with replacement
                    synthetic_data[col] = np.random.choice(
                        template[col].dropna(), num_samples, replace=True
                    )
            else:
                # For non-sensitive categorical columns, preserve distribution
                value_counts = template[col].value_counts(normalize=True)
                synthetic_data[col] = np.random.choice(
                    value_counts.index, num_samples, p=value_counts.values
                )
    else:
        # If fewer than 2 numeric columns, fall back to simple method
        synthetic_data = _generate_simple_synthetic(
            template, num_samples, sensitive_columns, **kwargs
        )
    
    return synthetic_data


def _generate_sdv_synthetic(
    template: pd.DataFrame,
    num_samples: int,
    sensitive_columns: List[str],
    synthesizer_type: str = "copula",
    anonymize_fields: bool = True,
    constraints: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate synthetic data using the Synthetic Data Vault (SDV) library.
    
    Args:
        template: Template dataset
        num_samples: Number of synthetic samples to generate
        sensitive_columns: Columns that contain sensitive data
        synthesizer_type: Type of SDV synthesizer to use ('copula', 'ctgan', 'tvae')
        anonymize_fields: Whether to anonymize sensitive fields with Faker
        constraints: List of constraint specifications for SDV
        **kwargs: Additional parameters for SDV synthesizers
        
    Returns:
        DataFrame containing synthetic data
        
    Raises:
        ImportError: If SDV is not installed
        ValueError: If an invalid synthesizer type is specified
    """
    if not SDV_AVAILABLE:
        raise ImportError(
            "SDV is required for this function. "
            "Please install it with 'pip install sdv'."
        )
    
    # Create a copy of the template
    df = template.copy()
    
    # Handle sensitive columns if anonymization is requested
    faker_replacements: Dict[str, List[Any]] = {}
    if anonymize_fields:
        df, faker_replacements = _preprocess_sensitive_columns(
            df, sensitive_columns, num_samples
        )
    
    # Create SDV metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # Apply constraints if provided
    sdv_constraints = []
    if constraints:
        sdv_constraints = _create_sdv_constraints(constraints, df)
    
    # Select the appropriate synthesizer based on the type
    if synthesizer_type == "copula":
        synthesizer = GaussianCopulaSynthesizer(
            metadata, constraints=sdv_constraints, **kwargs
        )
    elif synthesizer_type == "ctgan":
        synthesizer = CTGANSynthesizer(
            metadata, constraints=sdv_constraints, **kwargs
        )
    elif synthesizer_type == "tvae":
        synthesizer = TVAESynthesizer(
            metadata, constraints=sdv_constraints, **kwargs
        )
    else:
        raise ValueError(
            f"Invalid synthesizer type: {synthesizer_type}. "
            f"Must be one of 'copula', 'ctgan', or 'tvae'."
        )
    
    # Fit the synthesizer to the data
    synthesizer.fit(df)
    
    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_samples)
    
    # Restore sensitive columns with Faker-generated values if anonymized
    if anonymize_fields and faker_replacements:
        for col, values in faker_replacements.items():
            if col in synthetic_data.columns:
                synthetic_data[col] = values
    
    return synthetic_data


def _preprocess_sensitive_columns(
    df: pd.DataFrame,
    sensitive_columns: List[str],
    num_samples: int,
) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
    """
    Preprocess sensitive columns for SDV.
    
    This function can either remove sensitive columns from the data before passing
    to SDV, or replace them with anonymized values while preserving their
    statistical properties.
    
    Args:
        df: Original dataframe
        sensitive_columns: List of sensitive column names
        num_samples: Number of synthetic samples to generate
        
    Returns:
        Tuple containing:
        - Preprocessed dataframe
        - Dictionary mapping column names to Faker-generated values
    """
    processed_df = df.copy()
    faker_replacements: Dict[str, List[Any]] = {}
    faker = Faker()
    
    for col in sensitive_columns:
        if col in df.columns:
            # Generate appropriate faker values based on column name
            if "name" in col.lower():
                faker_replacements[col] = [faker.name() for _ in range(num_samples)]
            elif "email" in col.lower():
                faker_replacements[col] = [faker.email() for _ in range(num_samples)]
            elif "address" in col.lower():
                faker_replacements[col] = [
                    faker.address() for _ in range(num_samples)
                ]
            elif "phone" in col.lower():
                faker_replacements[col] = [
                    faker.phone_number() for _ in range(num_samples)
                ]
            elif "date" in col.lower() or "birth" in col.lower():
                faker_replacements[col] = [faker.date() for _ in range(num_samples)]
            elif "ssn" in col.lower() or "social" in col.lower():
                faker_replacements[col] = [faker.ssn() for _ in range(num_samples)]
            
            # Option 1: Remove the column from the dataframe before SDV modeling
            # processed_df = processed_df.drop(columns=[col])
            
            # Option 2: Replace with anonymized data while preserving structure
            # We'll use uuid-like strings as placeholders
            processed_df[col] = [f"id_{i}" for i in range(len(df))]
    
    return processed_df, faker_replacements


def _create_sdv_constraints(
    constraints: List[Dict[str, Any]],
    df: pd.DataFrame,
) -> List[Any]:
    """
    Create SDV constraint objects from constraint specifications.
    
    Args:
        constraints: List of constraint dictionaries
        df: The original dataframe
        
    Returns:
        List of SDV constraint objects
    """
    sdv_constraints = []
    
    for constraint in constraints:
        constraint_type = constraint.get("type")
        
        if constraint_type == "unique":
            columns = constraint.get("columns", [])
            sdv_constraints.append(Unique(columns))
            
        elif constraint_type == "fixed_combinations":
            column_names = constraint.get("column_names", [])
            sdv_constraints.append(FixedCombinations(column_names))
            
        elif constraint_type == "inequality":
            low_column = constraint.get("low_column")
            high_column = constraint.get("high_column")
            if low_column and high_column:
                sdv_constraints.append(Inequality(low_column, high_column))
    
    return sdv_constraints


def _generate_gan_synthetic(
    template: pd.DataFrame,
    num_samples: int,
    sensitive_columns: List[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate synthetic data using Generative Adversarial Networks (GANs).
    
    Note: This method is now deprecated in favor of sdv-ctgan or sdv-tvae.
    
    Args:
        template: Template dataset
        num_samples: Number of samples to generate
        sensitive_columns: Columns that contain sensitive data
        **kwargs: Additional parameters
        
    Returns:
        DataFrame containing synthetic data
    """
    # If SDV is available, use its CTGAN implementation
    if SDV_AVAILABLE:
        print("Using SDV's CTGAN implementation instead of custom GAN")
        return _generate_sdv_synthetic(
            template, num_samples, sensitive_columns, synthesizer_type="ctgan", **kwargs
        )
    else:
        # Placeholder - in a real implementation, this would use GANs
        # For demonstration purposes, we'll reuse the statistical method
        print(
            "GAN synthesis not fully implemented in this version; "
            "using statistical method"
        )
        return _generate_statistical_synthetic(
            template, num_samples, sensitive_columns, **kwargs
        )


def _generate_copula_synthetic(
    template: pd.DataFrame,
    num_samples: int,
    sensitive_columns: List[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate synthetic data using copulas to model dependencies between variables.
    
    Note: This method is now deprecated in favor of sdv-copula.
    
    Args:
        template: Template dataset
        num_samples: Number of samples to generate
        sensitive_columns: Columns that contain sensitive data
        **kwargs: Additional parameters
        
    Returns:
        DataFrame containing synthetic data
    """
    # If SDV is available, use its GaussianCopula implementation
    if SDV_AVAILABLE:
        print("Using SDV's GaussianCopula implementation instead of custom copula")
        return _generate_sdv_synthetic(
            template, 
            num_samples, 
            sensitive_columns, 
            synthesizer_type="copula", 
            **kwargs
        )
    else:
        # Placeholder - in a real implementation, this would use copulas
        # For demonstration purposes, we'll reuse the statistical method
        print(
            "Copula synthesis not fully implemented in this version; "
            "using statistical method"
        )
        return _generate_statistical_synthetic(
            template, num_samples, sensitive_columns, **kwargs
        ) 