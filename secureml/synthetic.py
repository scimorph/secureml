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
    from sdv.metadata import Metadata
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
    sensitivity_detection: Optional[Dict[str, Any]] = None,
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
        sensitivity_detection: Configuration for sensitive column detection:
            - sample_size: Number of rows to sample (default: 100)
            - confidence_threshold: Minimum confidence score (default: 0.5)
            - auto_detect: Whether to auto-detect sensitive columns if none provided (default: True)
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
    
    # Configure sensitivity detection
    sensitivity_config = sensitivity_detection or {}
    sample_size = sensitivity_config.get("sample_size", 100)
    confidence_threshold = sensitivity_config.get("confidence_threshold", 0.5)
    auto_detect = sensitivity_config.get("auto_detect", True)
        
    # If no sensitive columns specified and auto-detect is enabled, try to identify them
    if sensitive_columns is None and auto_detect:
        sensitive_columns = _identify_sensitive_columns(
            template_df, 
            sample_size=sample_size,
            confidence_threshold=confidence_threshold
        )
    elif sensitive_columns is None:
        sensitive_columns = []
        
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


def _identify_sensitive_columns(
    data: pd.DataFrame, 
    sample_size: int = 100, 
    confidence_threshold: float = 0.5
) -> List[str]:
    """
    Identify columns that likely contain sensitive data by analyzing both column names
    and data content.
    
    Args:
        data: The dataset to analyze
        sample_size: Number of sample values to check from each column
        confidence_threshold: Minimum confidence score (0.0-1.0) to classify a column as sensitive
        
    Returns:
        A list of column names that likely contain sensitive data
    """
    import re
    
    # Define patterns by category with regex patterns and keywords
    sensitive_patterns = {
        "personal_identifiers": {
            "name": {
                "keywords": ["name", "fullname", "firstname", "lastname", "username", "surname"],
                "regex": r"^[A-Z][a-z]+(?: [A-Z][a-z]+)+$",  # Simple name pattern
                "weight": 0.8
            },
            "email": {
                "keywords": ["email", "e-mail", "mail"],
                "regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "weight": 0.9
            },
            "phone": {
                "keywords": ["phone", "telephone", "mobile", "cell", "contact"],
                "regex": r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
                "weight": 0.9
            },
            "address": {
                "keywords": ["address", "street", "avenue", "road", "location", "city", "state", "zip", "postal"],
                "regex": r"\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd)",
                "weight": 0.7
            },
            "id": {
                "keywords": ["id", "identifier", "uuid", "guid"],
                "regex": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
                "weight": 0.5  # Lower weight as IDs might be non-sensitive
            },
            "ssn": {
                "keywords": ["ssn", "social", "security", "national id", "nationalid"],
                "regex": r"\d{3}-\d{2}-\d{4}",
                "weight": 1.0  # Maximum weight as SSNs are highly sensitive
            }
        },
        "demographic_data": {
            "gender": {
                "keywords": ["gender", "sex"],
                "regex": r"^(?:m|f|male|female|man|woman|non-binary|nonbinary|nb|other)$",
                "weight": 0.7
            },
            "race": {
                "keywords": ["race", "ethnicity", "ethnic"],
                "regex": r"^(?:caucasian|white|black|african|asian|hispanic|latino|native|indigenous|pacific|mixed|other)$",
                "weight": 0.9
            },
            "age": {
                "keywords": ["age", "birthday", "birth", "dob", "date of birth"],
                "regex": r"^(?:\d{1,3}|(?:19|20)\d{2}-\d{1,2}-\d{1,2})$",
                "weight": 0.6
            },
            "nationality": {
                "keywords": ["nationality", "citizenship", "country", "nation"],
                "regex": None,  # No specific regex as country names vary widely
                "weight": 0.6
            }
        },
        "financial_data": {
            "credit_card": {
                "keywords": ["credit", "card", "cc", "creditcard", "payment"],
                "regex": r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}",
                "weight": 1.0
            },
            "account": {
                "keywords": ["account", "bank", "routing", "iban", "swift"],
                "regex": r"\d{8,17}",  # Simple account number pattern
                "weight": 0.9
            },
            "income": {
                "keywords": ["income", "salary", "wage", "earnings", "compensation", "pay"],
                "regex": r"^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$",  # Money amount pattern
                "weight": 0.8
            },
            "net_worth": {
                "keywords": ["worth", "asset", "wealth", "equity", "portfolio"],
                "regex": None,
                "weight": 0.8
            }
        },
        "health_data": {
            "medical_condition": {
                "keywords": ["condition", "disease", "diagnosis", "illness", "health", "medical"],
                "regex": None,
                "weight": 0.9
            },
            "medication": {
                "keywords": ["medication", "drug", "prescription", "medicine", "treatment"],
                "regex": None,
                "weight": 0.9
            },
            "disability": {
                "keywords": ["disability", "disabled", "handicap", "impairment"],
                "regex": None,
                "weight": 0.9
            },
            "insurance": {
                "keywords": ["insurance", "coverage", "policy", "health insurance"],
                "regex": None,
                "weight": 0.7
            }
        },
        "access_credentials": {
            "password": {
                "keywords": ["password", "pwd", "passcode", "pin", "secret"],
                "regex": r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]{8,}$",  # Common password pattern
                "weight": 1.0
            },
            "api_key": {
                "keywords": ["api", "key", "token", "secret", "auth"],
                "regex": r"[A-Za-z0-9_-]{20,}",  # Long alphanumeric strings
                "weight": 0.9
            }
        }
    }
    
    # Results dictionary to store confidence scores for each column
    results = {}
    
    # Step 1: Analyze column names
    for col in data.columns:
        col_lower = col.lower()
        results[col] = {'name_score': 0, 'data_score': 0, 'total_score': 0}
        
        # Check column name against sensitive patterns
        for category, patterns in sensitive_patterns.items():
            for data_type, pattern_info in patterns.items():
                # Check if any keyword matches the column name
                if any(keyword in col_lower for keyword in pattern_info["keywords"]):
                    results[col]['name_score'] = pattern_info["weight"]
                    results[col]['category'] = category
                    results[col]['type'] = data_type
                    break
    
    # Step 2: Analyze data content if possible
    # Sample the data to avoid performance issues with large datasets
    sample_rows = min(sample_size, len(data))
    if sample_rows > 0:
        for col in data.columns:
            # Skip numeric columns for regex checks
            if pd.api.types.is_numeric_dtype(data[col]):
                continue
                
            # Get sample values as strings for analysis
            try:
                # Sample some non-null values
                sample_values = data[col].dropna().astype(str).sample(
                    min(sample_rows, data[col].count())
                ).tolist()
                
                if not sample_values:
                    continue
                
                # Check sample values against regex patterns
                for category, patterns in sensitive_patterns.items():
                    for data_type, pattern_info in patterns.items():
                        if pattern_info["regex"] is None:
                            continue
                            
                        # Calculate what percentage of samples match the regex
                        matches = 0
                        regex = re.compile(pattern_info["regex"], re.IGNORECASE)
                        for value in sample_values:
                            if regex.search(value):
                                matches += 1
                                
                        match_ratio = matches / len(sample_values) if sample_values else 0
                        if match_ratio > 0.3:  # If more than 30% match the pattern
                            # Update score if higher than current score
                            data_score = match_ratio * pattern_info["weight"]
                            if data_score > results[col].get('data_score', 0):
                                results[col]['data_score'] = data_score
                                # Only update category if not already set by name
                                if 'category' not in results[col]:
                                    results[col]['category'] = category
                                    results[col]['type'] = data_type
            except Exception:
                # Skip columns that cannot be analyzed
                continue
    
    # Step 3: Calculate total scores and filter columns
    sensitive_columns = []
    for col, scores in results.items():
        # Combine name and data scores (with data having higher priority)
        name_weight = 0.4  # Column name is less reliable than content
        data_weight = 0.6  # Content is more reliable than name
        
        total_score = (scores.get('name_score', 0) * name_weight + 
                       scores.get('data_score', 0) * data_weight)
        scores['total_score'] = total_score
        
        # Add columns that meet the confidence threshold
        if total_score >= confidence_threshold:
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
    Generate synthetic data using statistical modeling of the data distribution.
    
    This function creates synthetic data that preserves:
    - Individual column distributions
    - Correlations between variables (including categorical)
    - Multivariate relationships
    - Data types and ranges
    
    Args:
        template: Template dataset
        num_samples: Number of samples to generate
        sensitive_columns: Columns that contain sensitive data
        **kwargs: Additional parameters:
            - preserve_dtypes: Whether to preserve original data types (default: True)
            - preserve_outliers: Whether to preserve outlier patterns (default: True)
            - categorical_threshold: Max unique values to treat as categorical (default: 20)
            - handle_skewness: Whether to handle skewed numerical distributions (default: True)
            - seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing synthetic data
    """
    import scipy.stats as stats
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.mixture import GaussianMixture
    
    # Extract parameters
    preserve_dtypes = kwargs.get("preserve_dtypes", True)
    preserve_outliers = kwargs.get("preserve_outliers", True)
    categorical_threshold = kwargs.get("categorical_threshold", 20)
    handle_skewness = kwargs.get("handle_skewness", True)
    seed = kwargs.get("seed", None)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create a deep copy to avoid modifying the original
    data = template.copy()
    
    # Initialize the resulting synthetic dataframe
    synthetic_data = pd.DataFrame(index=range(num_samples), columns=data.columns)
    
    # Step 1: Categorize columns and handle sensitive columns
    categorical_cols = []
    numeric_cols = []
    date_cols = []
    other_cols = []
    
    # Dictionary to store synthetic values for sensitive columns
    sensitive_values = {}
    
    # Identify column types
    for col in data.columns:
        # Handle sensitive columns with faker
        if col in sensitive_columns:
            sensitive_values[col] = _generate_sensitive_values(
                col, data[col], num_samples
            )
            continue
        
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(data[col]):
            # If low cardinality, treat as categorical
            if data[col].nunique() <= categorical_threshold:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        # Check if it's a datetime column
        elif pd.api.types.is_datetime64_dtype(data[col]):
            date_cols.append(col)
        # Check if it's categorical or object with low cardinality
        elif (pd.api.types.is_categorical_dtype(data[col]) or 
              (pd.api.types.is_object_dtype(data[col]) and 
               data[col].nunique() <= categorical_threshold)):
            categorical_cols.append(col)
        else:
            other_cols.append(col)
    
    # Step 2: Handle datetime columns separately
    for col in date_cols:
        if not data[col].isna().all():
            # Convert to numeric (timestamps)
            timestamps = data[col].astype('int64') // 10**9
            
            # Fit a kernel density
            kde = stats.gaussian_kde(timestamps.dropna())
            
            # Sample from kde
            sampled_timestamps = kde.resample(num_samples)[0]
            
            # Convert back to datetime
            synthetic_data[col] = pd.to_datetime(sampled_timestamps, unit='s')
    
    # Step 3: Handle categorical columns
    # Create a mapping of categorical columns to their distributions
    categorical_distributions = {}
    for col in categorical_cols:
        if not data[col].isna().all():
            # Get value counts including NaN values if present
            val_counts = data[col].value_counts(normalize=True, dropna=False)
            categorical_distributions[col] = val_counts

    # Step 4: Handle numeric columns
    if numeric_cols:
        synthetic_data = pd.DataFrame(index=range(num_samples), columns=template.columns)
        for col in numeric_cols:
            values = template[col].dropna().values
            if len(values) > 1:
                # Check for low variance
                orig_std = template[col].std()
                if orig_std < 1.0:  # Threshold for low variance
                    # Get unique values and their probabilities
                    unique, counts = np.unique(values, return_counts=True)
                    probabilities = counts / len(values)
                    synthetic_data[col] = np.random.choice(unique, num_samples, replace=True, p=probabilities)
                else:
                    # Use KDE for better distribution fit
                    kde = stats.gaussian_kde(values)
                    synthetic_values = kde.resample(num_samples)[0]
                    # Clip to original range to avoid extreme outliers
                    min_val, max_val = template[col].min(), template[col].max()
                    synthetic_values = np.clip(synthetic_values, min_val, max_val)
                    synthetic_data[col] = synthetic_values
                
                # Preserve integer dtype
                if pd.api.types.is_integer_dtype(template[col]):
                    synthetic_data[col] = synthetic_data[col].round().astype(int)
    
    # Step 5: Generate categorical variables with exact proportions
    for col in categorical_cols:
        if col in categorical_distributions:
            val_counts = categorical_distributions[col]
            # Calculate expected counts for each value
            expected_counts = {val: int(np.round(p * num_samples)) for val, p in val_counts.items()}
            # Adjust total to match num_samples
            total = sum(expected_counts.values())
            if total < num_samples:
                # Add remainder to the most frequent category
                most_frequent = val_counts.idxmax()
                expected_counts[most_frequent] += num_samples - total
            elif total > num_samples:
                # Subtract excess from the least frequent category
                while total > num_samples:
                    least_frequent = min(
                        {k: v for k, v in expected_counts.items() if v > 0},
                        key=expected_counts.get
                    )
                    if expected_counts[least_frequent] > 0:
                        expected_counts[least_frequent] -= 1
                        total -= 1
                    else:
                        break

            # Generate synthetic values with exact counts
            synthetic_values = []
            for val, count in expected_counts.items():
                synthetic_values.extend([val] * count)
            # Shuffle to randomize order
            np.random.shuffle(synthetic_values)
            # Ensure length matches num_samples (edge case handling)
            if len(synthetic_values) < num_samples:
                additional = np.random.choice(
                    list(val_counts.index),
                    size=num_samples - len(synthetic_values),
                    p=val_counts.values
                )
                synthetic_values.extend(additional)
            elif len(synthetic_values) > num_samples:
                synthetic_values = synthetic_values[:num_samples]
            synthetic_data[col] = synthetic_values
    
    # Step 6: Handle other columns (text, high-cardinality categorical, etc.)
    for col in other_cols:
        if not data[col].isna().all():
            # For other types, sample from observed values with replacement
            synthetic_data[col] = np.random.choice(
                data[col].dropna(), 
                size=num_samples, 
                replace=True
            )
    
    # Step 7: Add sensitive columns
    for col, values in sensitive_values.items():
        synthetic_data[col] = values
    
    # Step 8: Post-processing - adjust synthetic data for better correlation preservation
    # Use vine copula approach to adjust the numeric data based on joint distribution
    if len(numeric_cols) >= 2:
        try:
            # Calculate the correlation matrix of the original data
            orig_corr = data[numeric_cols].corr(method='spearman').values
            # Calculate the correlation matrix of the synthetic data
            synth_corr = synthetic_data[numeric_cols].corr(method='spearman').values
            
            # If correlations differ significantly, try to adjust synthetic data
            if np.max(np.abs(orig_corr - synth_corr)) > 0.2:
                from scipy.stats import rankdata
                
                # Get the ranks of synthetic data
                synthetic_ranks = {}
                for col in numeric_cols:
                    synthetic_ranks[col] = rankdata(synthetic_data[col])
                
                # Create a sequence of adjusted columns
                for i, target_col in enumerate(numeric_cols):
                    if i == 0:
                        continue  # Skip the first column as it's our anchor
                    
                    # Calculate the target ranks based on correlation with prior columns
                    target_ranks = np.zeros(num_samples)
                    for j in range(i):
                        source_col = numeric_cols[j]
                        # Skip if the correlation is too low
                        if abs(orig_corr[i, j]) < 0.1:
                            continue
                        
                        # Combine the ranks based on correlation strength
                        weight = orig_corr[i, j]
                        source_ranks = synthetic_ranks[source_col]
                        
                        # Adjust the target ranks based on source ranks and correlation
                        if weight > 0:
                            target_ranks += weight * source_ranks
                        else:
                            target_ranks += abs(weight) * (num_samples + 1 - source_ranks)
                    
                    # If we've made adjustments, reorder the target column
                    if not np.all(target_ranks == 0):
                        # Normalize the target ranks
                        target_ranks = rankdata(target_ranks)
                        
                        # Get the values sorted
                        sorted_values = np.sort(synthetic_data[target_col].values)
                        
                        # Create adjusted values
                        adjusted_values = np.zeros(num_samples)
                        for k in range(num_samples):
                            idx = int((target_ranks[k] - 1) / num_samples * len(sorted_values))
                            idx = min(idx, len(sorted_values) - 1)
                            adjusted_values[k] = sorted_values[idx]
                        
                        # Update the synthetic data
                        synthetic_data[target_col] = adjusted_values
        except Exception:
            # Ignore errors in the post-processing step
            pass
    
    # Step 9: Restore data types if requested
    if preserve_dtypes:
        for col in data.columns:
            if col in synthetic_data.columns:
                # Skip sensitive columns as they are handled separately
                if col in sensitive_columns:
                    continue
                
                # Try to match the original dtype
                try:
                    orig_dtype = data[col].dtype
                    if pd.api.types.is_categorical_dtype(orig_dtype):
                        # Ensure all synthetic values are in the original categories
                        orig_categories = data[col].cat.categories
                        synthetic_values = synthetic_data[col].values
                        valid_mask = np.isin(synthetic_values, orig_categories)
                        if not np.all(valid_mask):
                            # Replace invalid values with random valid ones
                            invalid_indices = np.where(~valid_mask)[0]
                            synthetic_data.loc[invalid_indices, col] = np.random.choice(
                                orig_categories, size=len(invalid_indices)
                            )
                        synthetic_data[col] = pd.Categorical(
                            synthetic_data[col], categories=orig_categories
                        )
                    elif str(orig_dtype) != str(synthetic_data[col].dtype):
                        synthetic_data[col] = synthetic_data[col].astype(orig_dtype)
                except (TypeError, ValueError):
                    # Skip if type conversion fails
                    pass
    
    return synthetic_data


def _generate_sensitive_values(col: str, data_series: pd.Series, num_samples: int) -> np.ndarray:
    """
    Generate synthetic values for a sensitive column based on column name and data type.
    
    Args:
        col: Column name
        data_series: Original data series
        num_samples: Number of samples to generate
        
    Returns:
        Array of synthetic values for the sensitive column
    """
    faker = Faker()
    col_lower = col.lower()
    dtype = data_series.dtype
    
    # For numeric sensitive columns, generate values within the same range but add noise
    if pd.api.types.is_numeric_dtype(dtype):
        # For numeric columns, generate values within the same range
        if len(data_series.dropna()) > 0:
            min_val = data_series.min()
            max_val = data_series.max()
            # Apply jitter to avoid exact matches
            jitter = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
            synthetic_values = np.random.uniform(
                min_val - jitter, max_val + jitter, num_samples
            )
            if pd.api.types.is_integer_dtype(dtype):
                synthetic_values = np.round(synthetic_values).astype(int)
        else:
            # If all values are NA, generate random numbers
            synthetic_values = np.random.normal(0, 1, num_samples)
            if pd.api.types.is_integer_dtype(dtype):
                synthetic_values = np.round(synthetic_values).astype(int)
    # For categorical sensitive columns, generate similar distribution but with fake values
    elif pd.api.types.is_categorical_dtype(dtype) or data_series.nunique() < 20:
        # Get the distribution of values
        value_counts = data_series.value_counts(normalize=True)
        # Generate synthetic categorical values with the same distribution
        categories = []
        # Generate appropriate synthetic categories
        if "gender" in col_lower or "sex" in col_lower:
            categories = ["Male", "Female", "Other", "Prefer not to say"]
        elif "race" in col_lower or "ethnicity" in col_lower:
            categories = ["Group A", "Group B", "Group C", "Group D", "Group E"]
        elif "religion" in col_lower:
            categories = ["Religion 1", "Religion 2", "Religion 3", "None", "Other"]
        elif "country" in col_lower or "nation" in col_lower:
            # Generate a fixed number of unique countries, e.g., 20
            categories = [faker.country() for _ in range(20)]
        else:
            # Match the number of unique values for consistency
            categories = [f"Category {i+1}" for i in range(data_series.nunique())]
            
        # Sample uniformly, omitting the p parameter
        synthetic_values = np.random.choice(categories, size=num_samples)
    # For string columns use appropriate faker providers
    else:
        if "name" in col_lower:
            synthetic_values = np.array([faker.name() for _ in range(num_samples)])
        elif "email" in col_lower:
            synthetic_values = np.array([faker.email() for _ in range(num_samples)])
        elif "address" in col_lower or "location" in col_lower:
            synthetic_values = np.array([faker.address() for _ in range(num_samples)])
        elif "phone" in col_lower:
            synthetic_values = np.array([faker.phone_number() for _ in range(num_samples)])
        elif "date" in col_lower or "birth" in col_lower:
            synthetic_values = np.array([faker.date() for _ in range(num_samples)])
        elif "ssn" in col_lower or "social" in col_lower:
            synthetic_values = np.array([faker.ssn() for _ in range(num_samples)])
        elif "credit" in col_lower and "card" in col_lower:
            synthetic_values = np.array([faker.credit_card_number() for _ in range(num_samples)])
        elif "company" in col_lower or "business" in col_lower:
            synthetic_values = np.array([faker.company() for _ in range(num_samples)])
        elif "job" in col_lower or "occupation" in col_lower:
            synthetic_values = np.array([faker.job() for _ in range(num_samples)])
        elif pd.api.types.is_datetime64_dtype(dtype):
            start_date = pd.Timestamp("2000-01-01")
            end_date = pd.Timestamp("2023-01-01")
            synthetic_values = np.array([
                faker.date_between(start_date=start_date, end_date=end_date)
                for _ in range(num_samples)
            ])
        else:
            # For other types, generate random strings
            synthetic_values = np.array([faker.text(max_nb_chars=20) for _ in range(num_samples)])
    
    return synthetic_values


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
    metadata = Metadata()
    metadata.detect_from_dataframe(df)
    
    # Apply constraints if provided
    sdv_constraints = []
    if constraints:
        # Filter constraints to only include valid columns
        for constraint in constraints:
            constraint_type = constraint.get("type")
            columns = constraint.get("columns", [])
            if isinstance(columns, str):
                columns = [columns]
            # Check if all specified columns exist in df
            if constraint_type == "unique" and all(col in df.columns for col in columns):
                sdv_constraints.append({
                    'constraint_class': 'Unique',
                    'constraint_parameters': {'column_names': columns}
                })
            elif constraint_type == "unique":
                print(f"Warning: Skipping Unique constraint for {columns}; some columns not found in data.")
    
    # Select and initialize the appropriate synthesizer based on the type
    if synthesizer_type == "copula":
        synthesizer = GaussianCopulaSynthesizer(metadata, **kwargs)
    elif synthesizer_type == "ctgan":
        synthesizer = CTGANSynthesizer(metadata, **kwargs)
    elif synthesizer_type == "tvae":
        synthesizer = TVAESynthesizer(metadata, **kwargs)
    else:
        raise ValueError(
            f"Invalid synthesizer type: {synthesizer_type}. "
            f"Must be one of 'copula', 'ctgan', or 'tvae'."
        )
    
    # Add constraints to the synthesizer if any exist
    if sdv_constraints:
        synthesizer.add_constraints(sdv_constraints)
    
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
) -> List[Dict[str, Any]]:
    """
    Create SDV constraint dictionaries from constraint specifications.
    
    Args:
        constraints: List of constraint dictionaries
        df: The original dataframe
        
    Returns:
        List of SDV constraint dictionaries
    """
    sdv_constraints = []
    for constraint in constraints:
        constraint_type = constraint.get("type")
        if constraint_type == "unique":
            columns = constraint.get("columns", [])
            sdv_constraints.append({
                'constraint_class': 'Unique',
                'constraint_parameters': {'column_names': columns}
            })
        elif constraint_type == "fixed_combinations":
            column_names = constraint.get("column_names", [])
            sdv_constraints.append({
                'constraint_class': 'FixedCombinations',
                'constraint_parameters': {'column_names': column_names}
            })
        elif constraint_type == "inequality":
            low_column = constraint.get("low_column")
            high_column = constraint.get("high_column")
            if low_column and high_column:
                sdv_constraints.append({
                    'constraint_class': 'Inequality',
                    'constraint_parameters': {
                        'low_column_name': low_column,
                        'high_column_name': high_column
                    }
                })
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
        **kwargs: Additional parameters:
            - epochs: Number of training epochs (default: 300)
            - batch_size: Batch size for training (default: 32)
            - generator_dim: Dimensions of generator layers (default: [128, 128])
            - discriminator_dim: Dimensions of discriminator layers (default: [128, 128])
            - learning_rate: Learning rate for optimizer (default: 0.001)
            - noise_dim: Dimension of noise input (default: 100)
            - preserve_dtypes: Whether to preserve original dtypes (default: True)
        
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
        # Implement GAN-based synthetic data generation
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, optimizers, models
            from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
        except ImportError:
            raise ImportError(
                "TensorFlow is required for GAN-based synthesis. "
                "Please install with 'pip install tensorflow'"
            )
            
        # Extract kwargs
        epochs = kwargs.get('epochs', 300)
        batch_size = kwargs.get('batch_size', 32)
        generator_dim = kwargs.get('generator_dim', [128, 128])
        discriminator_dim = kwargs.get('discriminator_dim', [128, 128])
        learning_rate = kwargs.get('learning_rate', 0.001)
        noise_dim = kwargs.get('noise_dim', 100)
        preserve_dtypes = kwargs.get('preserve_dtypes', True)
        
        print(f"Training GAN for {epochs} epochs with batch size {batch_size}...")
        
        # Create a working copy of the data
        data = template.copy()
        
        # Handle sensitive columns
        for col in sensitive_columns:
            if col in data.columns:
                data[col] = _generate_sensitive_values(col, data[col], len(data))
                
        # Categorize columns
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        
        # Store original dtypes
        original_dtypes = data.dtypes.to_dict()
        
        # Process data types and identify column categories
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                numeric_cols.append(col)
            elif pd.api.types.is_datetime64_dtype(data[col]):
                datetime_cols.append(col)
                # Convert datetime to numeric (timestamp)
                data[col] = data[col].astype(np.int64) // 10**9
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Prepare data for GAN
        # 1. Normalize numeric columns
        scaler = MinMaxScaler()
        if numeric_cols:
            numeric_data = data[numeric_cols].copy()
            # Fill NaN values with mean for numeric data
            for col in numeric_cols:
                if numeric_data[col].isna().any():
                    numeric_data[col] = numeric_data[col].fillna(numeric_data[col].mean())
            
            # Fit scaler on numeric data
            normalized_numeric = scaler.fit_transform(numeric_data)
            
        # 2. One-hot encode categorical columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_data = None
        if categorical_cols:
            # Fill NaN values with mode for categorical data
            for col in categorical_cols:
                if data[col].isna().any():
                    data[col] = data[col].fillna(data[col].mode()[0])
            
            # Fit encoder on categorical data
            categorical_data = data[categorical_cols].copy()
            one_hot_data = encoder.fit_transform(categorical_data)
        
        # 3. Combine processed data
        if numeric_cols and categorical_cols:
            processed_data = np.hstack((normalized_numeric, one_hot_data))
        elif numeric_cols:
            processed_data = normalized_numeric
        else:
            processed_data = one_hot_data
        
        # Get dimensions
        data_dim = processed_data.shape[1]
        
        # Build generator
        def build_generator(noise_dim, data_dim, generator_dim):
            model = models.Sequential()
            model.add(layers.Dense(generator_dim[0], input_dim=noise_dim, activation='relu'))
            
            for dim in generator_dim[1:]:
                model.add(layers.Dense(dim, activation='relu'))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU(alpha=0.2))
            
            model.add(layers.Dense(data_dim, activation='tanh'))
            return model
        
        # Build discriminator
        def build_discriminator(data_dim, discriminator_dim):
            model = models.Sequential()
            model.add(layers.Dense(discriminator_dim[0], input_dim=data_dim, activation='relu'))
            
            for dim in discriminator_dim[1:]:
                model.add(layers.Dense(dim, activation='relu'))
                model.add(layers.LeakyReLU(alpha=0.2))
                model.add(layers.Dropout(0.3))
            
            model.add(layers.Dense(1, activation='sigmoid'))
            return model
        
        # Set up GAN components
        generator = build_generator(noise_dim, data_dim, generator_dim)
        discriminator = build_discriminator(data_dim, discriminator_dim)
        
        # Compile discriminator
        discriminator.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Combined GAN model (only generator is trained through this model)
        discriminator.trainable = False
        
        # Create GAN
        gan_input = tf.keras.Input(shape=(noise_dim,))
        generated_data = generator(gan_input)
        gan_output = discriminator(generated_data)
        gan = models.Model(gan_input, gan_output)
        
        # Compile GAN
        gan.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy'
        )
        
        # Convert to TensorFlow dataset for better performance
        train_dataset = tf.data.Dataset.from_tensor_slices(processed_data)
        train_dataset = train_dataset.shuffle(buffer_size=len(processed_data))
        train_dataset = train_dataset.batch(batch_size)
        
        # Training loop
        for epoch in range(epochs):
            # Initialize metrics
            d_loss_sum = 0
            g_loss_sum = 0
            batch_count = 0
            
            for real_batch in train_dataset:
                batch_size = tf.shape(real_batch)[0]
                
                # Train discriminator on real data
                real_labels = tf.ones((batch_size, 1))
                d_loss_real = discriminator.train_on_batch(real_batch, real_labels)[0]
                
                # Train discriminator on fake data
                noise = tf.random.normal((batch_size, noise_dim))
                fake_data = generator.predict(noise, verbose=0)
                fake_labels = tf.zeros((batch_size, 1))
                d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)[0]
                
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                
                # Train generator
                noise = tf.random.normal((batch_size, noise_dim))
                gan_labels = tf.ones((batch_size, 1))
                g_loss = gan.train_on_batch(noise, gan_labels)
                
                # Update sums for average calculation
                d_loss_sum += d_loss
                g_loss_sum += g_loss
                batch_count += 1
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                d_loss_avg = d_loss_sum / batch_count
                g_loss_avg = g_loss_sum / batch_count
                print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss_avg:.4f}, G Loss: {g_loss_avg:.4f}")
        
        # Generate synthetic data
        noise = tf.random.normal((num_samples, noise_dim))
        synthetic_data_raw = generator.predict(noise, verbose=0)
        
        # Convert back to DataFrame with original structure
        synthetic_df = pd.DataFrame()
        
        # Process numeric columns
        if numeric_cols:
            numeric_count = len(numeric_cols)
            synthetic_numeric = synthetic_data_raw[:, :numeric_count]
            
            # Inverse transform to original scale
            synthetic_numeric = scaler.inverse_transform(
                np.clip(synthetic_numeric, -1, 1)  # Clip in tanh range
            )
            
            # Add to DataFrame
            for i, col in enumerate(numeric_cols):
                synthetic_df[col] = synthetic_numeric[:, i]
        
        # Process categorical columns
        if categorical_cols:
            start_idx = len(numeric_cols) if numeric_cols else 0
            categorical_synthetic = synthetic_data_raw[:, start_idx:]
            
            # Convert continuous values to one-hot encoded categorical
            # For each category, find nearest one-hot encoding
            threshold = 0.5
            categorical_synthetic = (categorical_synthetic > threshold).astype(float)
            
            # Inverse transform to original categories
            decoded_categorical = encoder.inverse_transform(categorical_synthetic)
            
            # Add to DataFrame
            for i, col in enumerate(categorical_cols):
                synthetic_df[col] = decoded_categorical[:, i]
        
        # Convert datetime columns back
        for col in datetime_cols:
            if col in synthetic_df.columns:
                synthetic_df[col] = pd.to_datetime(synthetic_df[col].astype(np.int64) * 10**9)
        
        # Restore original dtypes if requested
        if preserve_dtypes:
            for col, dtype in original_dtypes.items():
                if col in synthetic_df.columns:
                    try:
                        if pd.api.types.is_categorical_dtype(dtype):
                            categories = template[col].cat.categories
                            synthetic_df[col] = pd.Categorical(
                                synthetic_df[col], 
                                categories=categories
                            )
                        elif pd.api.types.is_integer_dtype(dtype) and col not in datetime_cols:
                            synthetic_df[col] = synthetic_df[col].round().astype(dtype)
                        elif not pd.api.types.is_datetime64_dtype(dtype) and col not in datetime_cols:
                            synthetic_df[col] = synthetic_df[col].astype(dtype)
                    except (TypeError, ValueError):
                        # Skip if type conversion fails
                        pass
        
        # Handle sensitive columns for the final synthetic data
        for col in sensitive_columns:
            if col in synthetic_df.columns:
                synthetic_df[col] = _generate_sensitive_values(col, template[col], num_samples)
        
        return synthetic_df


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
        **kwargs: Additional parameters:
            - copula_type: Type of copula to use (default: "gaussian")
            - fit_method: Method for fitting copula (default: "ml")
            - preserve_dtypes: Whether to preserve original data types (default: True)
            - handle_missing: How to handle missing values (default: "mean")
            - categorical_threshold: Max unique values to treat as categorical (default: 20)
            - handle_skewness: Whether to handle skewed numerical distributions (default: True)
            - seed: Random seed for reproducibility
        
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
        # Implement copula-based synthetic data generation
        try:
            from scipy import stats
            from scipy.stats import norm, gaussian_kde, rankdata
            import numpy as np
            from sklearn.preprocessing import QuantileTransformer
        except ImportError:
            raise ImportError(
                "SciPy and scikit-learn are required for copula-based synthesis. "
                "Please install with 'pip install scipy scikit-learn'"
            )
            
        # Extract kwargs with defaults
        copula_type = kwargs.get('copula_type', 'gaussian')
        fit_method = kwargs.get('fit_method', 'ml')
        preserve_dtypes = kwargs.get('preserve_dtypes', True)
        handle_missing = kwargs.get('handle_missing', 'mean')
        categorical_threshold = kwargs.get('categorical_threshold', 20)
        handle_skewness = kwargs.get('handle_skewness', True)
        seed = kwargs.get('seed', None)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        print(f"Generating synthetic data using {copula_type} copula...")
        
        # Create a working copy of the data
        data = template.copy()
        
        # Step 1: Handle sensitive columns separately
        for col in sensitive_columns:
            if col in data.columns:
                data[col] = _generate_sensitive_values(col, data[col], len(data))
        
        # Step 2: Categorize columns
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        
        # Store original dtypes
        original_dtypes = data.dtypes.to_dict()
        
        # Process data types and identify column categories
        for col in data.columns:
            if pd.api.types.is_datetime64_dtype(data[col]):
                datetime_cols.append(col)
                # Convert datetime to numeric (timestamp)
                data[col] = data[col].astype(np.int64) // 10**9
                numeric_cols.append(col)
            elif pd.api.types.is_numeric_dtype(data[col]):
                numeric_cols.append(col)
            elif data[col].nunique() < categorical_threshold:
                categorical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Step 3: Handle missing values
        for col in data.columns:
            if data[col].isna().any():
                if col in numeric_cols:
                    if handle_missing == 'mean':
                        data[col] = data[col].fillna(data[col].mean())
                    elif handle_missing == 'median':
                        data[col] = data[col].fillna(data[col].median())
                    else:  # Default to 0
                        data[col] = data[col].fillna(0)
                else:
                    # For categorical, use most frequent value
                    data[col] = data[col].fillna(data[col].mode()[0])
        
        # Step 4: Create processed data for copula modeling
        processed_data = pd.DataFrame(index=data.index)
        
        # Step 5: Transform categorical variables to numeric
        categorical_mappers = {}
        for col in categorical_cols:
            # Create a mapping for this categorical column
            unique_values = data[col].unique()
            value_map = {val: i for i, val in enumerate(unique_values)}
            categorical_mappers[col] = {'map': value_map, 'values': unique_values}
            
            # Transform to numeric
            processed_data[col] = data[col].map(value_map)
        
        # Step 6: Add numeric columns directly
        for col in numeric_cols:
            processed_data[col] = data[col]
        
        # Step 7: Handle skewness if requested
        transformers = {}
        if handle_skewness:
            for col in numeric_cols:
                # Skip if column has too many identical values
                if processed_data[col].nunique() > 5:
                    skewness = stats.skew(processed_data[col].dropna())
                    if abs(skewness) > 1.0:  # Threshold for "skewed" data
                        transformer = QuantileTransformer(output_distribution='normal')
                        transformed_col = transformer.fit_transform(
                            processed_data[[col]].values
                        ).flatten()
                        processed_data[col] = transformed_col
                        transformers[col] = transformer
        
        # Step 8: Apply rank transformation to all variables for copula modeling
        rank_data = pd.DataFrame(index=processed_data.index)
        
        for col in processed_data.columns:
            # Perform rank transformation (convert to uniform marginals)
            ranks = rankdata(processed_data[col])
            # Scale ranks to [0, 1)
            rank_data[col] = (ranks - 0.5) / len(ranks)
        
        # Step 9: Transform uniform marginals to standard normal for Gaussian copula
        normal_data = pd.DataFrame(index=rank_data.index)
        
        for col in rank_data.columns:
            # Apply inverse normal CDF to get standard normal variates
            normal_data[col] = norm.ppf(rank_data[col])
        
        # Step 10: Compute correlation matrix (Gaussian copula parameter)
        if copula_type == 'gaussian':
            # For Gaussian copula, we use the correlation matrix
            if fit_method == 'ml':
                # Maximum likelihood estimation
                corr_matrix = np.corrcoef(normal_data.values, rowvar=False)
            else:
                # Spearman's rank correlation
                corr_matrix = processed_data.corr(method='spearman').values
        else:
            # For t-copula, we would also need to estimate the degrees of freedom
            # This is a simplified version using Spearman's correlation
            corr_matrix = processed_data.corr(method='spearman').values
        
        # Step 11: Generate synthetic samples from the copula
        if copula_type == 'gaussian':
            # Generate samples from multivariate normal with the estimated correlation
            synthetic_normal = np.random.multivariate_normal(
                mean=np.zeros(corr_matrix.shape[0]),
                cov=corr_matrix,
                size=num_samples
            )
        else:
            # For t-copula, we would use multivariate t-distribution
            # This is a simplified version using multivariate normal
            synthetic_normal = np.random.multivariate_normal(
                mean=np.zeros(corr_matrix.shape[0]),
                cov=corr_matrix,
                size=num_samples
            )
        
        # Convert to DataFrame with column names
        synthetic_normal_df = pd.DataFrame(
            synthetic_normal, 
            columns=normal_data.columns
        )
        
        # Step 12: Transform back to uniform marginals
        synthetic_uniform_df = pd.DataFrame(index=range(num_samples))
        
        for col in synthetic_normal_df.columns:
            # Apply normal CDF to get uniform variates
            synthetic_uniform_df[col] = norm.cdf(synthetic_normal_df[col])
        
        # Step 13: Transform back to original marginal distributions
        synthetic_df = pd.DataFrame(index=range(num_samples))
        
        # Process numeric columns
        for col in numeric_cols:
            # Get the empirical inverse CDF
            original_values = processed_data[col].sort_values().values
            
            # Apply inverse CDF to transform uniform back to original distribution
            quantiles = synthetic_uniform_df[col].values
            indices = (quantiles * len(original_values)).astype(int).clip(0, len(original_values) - 1)
            synthetic_values = original_values[indices]
            
            # Inverse transform if skewness was handled
            if col in transformers and handle_skewness:
                synthetic_values = synthetic_values.reshape(-1, 1)
                synthetic_values = transformers[col].inverse_transform(synthetic_values).flatten()
            
            synthetic_df[col] = synthetic_values
        
        # Process categorical columns
        for col in categorical_cols:
            # Get the mapping information
            value_map = categorical_mappers[col]['map']
            unique_values = categorical_mappers[col]['values']
            
            # Determine the number of categories
            num_categories = len(value_map)
            
            # Create bins for the uniform values
            bin_edges = np.linspace(0, 1, num_categories + 1)
            
            # Assign each uniform value to a category index
            indices = np.digitize(synthetic_uniform_df[col], bin_edges) - 1
            indices = np.clip(indices, 0, num_categories - 1)
            
            # Map indices back to original categorical values
            synthetic_values = [unique_values[i] for i in indices]
            synthetic_df[col] = synthetic_values
        
        # Step 14: Convert datetime columns back
        for col in datetime_cols:
            if col in synthetic_df.columns:
                synthetic_df[col] = pd.to_datetime(synthetic_df[col].astype(np.int64) * 10**9)
        
        # Step 15: Preserve original data types if requested
        if preserve_dtypes:
            for col, dtype in original_dtypes.items():
                if col in synthetic_df.columns:
                    try:
                        if pd.api.types.is_categorical_dtype(dtype):
                            categories = template[col].cat.categories
                            synthetic_df[col] = pd.Categorical(
                                synthetic_df[col], 
                                categories=categories
                            )
                        elif pd.api.types.is_integer_dtype(dtype) and col not in datetime_cols:
                            synthetic_df[col] = synthetic_df[col].round().astype(dtype)
                        elif not pd.api.types.is_datetime64_dtype(dtype) and col not in datetime_cols:
                            synthetic_df[col] = synthetic_df[col].astype(dtype)
                    except (TypeError, ValueError):
                        # Skip if type conversion fails
                        pass
        
        # Step 16: Handle sensitive columns for the final synthetic data
        for col in sensitive_columns:
            if col in synthetic_df.columns:
                synthetic_df[col] = _generate_sensitive_values(col, template[col], num_samples)
        
        return synthetic_df 