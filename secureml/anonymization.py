"""
Data anonymization functionality for SecureML.

This module provides tools to anonymize sensitive data before using it
in machine learning workflows.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd


def anonymize(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    method: str = "k-anonymity",
    k: int = 5,
    sensitive_columns: Optional[List[str]] = None,
    **kwargs: Any,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Anonymize a dataset using the specified method.

    Args:
        data: The dataset to anonymize, either as a pandas DataFrame or a list of dicts
        method: Anonymization method to use. Options: 'k-anonymity', 'pseudonymization',
               'data-masking', 'generalization'
        k: Parameter for k-anonymity (minimum size of equivalent classes)
        sensitive_columns: List of column names containing sensitive information
        **kwargs: Additional parameters for specific anonymization methods

    Returns:
        The anonymized dataset in the same format as the input

    Raises:
        ValueError: If an unsupported anonymization method is specified
    """
    # Convert list of dicts to DataFrame if necessary
    original_format_is_list = False
    if isinstance(data, list):
        original_format_is_list = True
        data = pd.DataFrame(data)

    # Convert to DataFrame and validate columns
    df = data.copy()

    # If no sensitive columns are specified, try to identify them automatically
    if sensitive_columns is None:
        sensitive_columns = _identify_sensitive_columns(data)
        if not sensitive_columns:
            raise ValueError(
                "No sensitive columns identified. Please specify them manually."
            )
    else:
        # Check for non-existent columns
        missing_cols = [col for col in sensitive_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns were not found in the data: {', '.join(missing_cols)}")

    # Apply the selected anonymization method
    if method == "k-anonymity":
        result = _apply_k_anonymity(data, sensitive_columns, k, **kwargs)
    elif method == "pseudonymization":
        result = _apply_pseudonymization(data, sensitive_columns, **kwargs)
    elif method == "data-masking":
        result = _apply_data_masking(data, sensitive_columns, **kwargs)
    elif method == "generalization":
        result = _apply_generalization(data, sensitive_columns, **kwargs)
    else:
        raise ValueError(f"Unsupported anonymization method: {method}")

    # Convert back to original format if necessary
    if original_format_is_list:
        return result.to_dict("records")
    return result


def _identify_sensitive_columns(data: pd.DataFrame) -> List[str]:
    """
    Automatically identify sensitive columns in a dataset using pattern matching 
    and content analysis.

    This implementation categorizes sensitive data according to modern privacy 
    frameworks like GDPR, CCPA, HIPAA, and ISO/IEC 27701, considering both direct
    and quasi-identifiers.

    Args:
        data: The dataset to analyze

    Returns:
        A list of column names that appear to contain sensitive information
    """
    # Comprehensive categorization of sensitive data types
    sensitive_data_categories = {
        # Personal Identifiers (PII)
        "personal_identifiers": [
            "name", "fullname", "firstname", "lastname", "middlename", 
            "username", "nickname", "alias", "maiden"
        ],
        
        # Contact Information
        "contact_info": [
            "email", "e-mail", "mail", "phone", "mobile", "cellphone", 
            "telephone", "fax", "address", "street", "city", "state", "zip", 
            "zipcode", "postal", "country", "location"
        ],
        
        # Government & Official IDs
        "official_ids": [
            "ssn", "social", "security", "passport", "license", "driver", "id", 
            "identification", "national", "tax", "taxid", "ein", "fein", 
            "itin", "citizenship", "voter", "registration"
        ],
        
        # Financial Information
        "financial_info": [
            "account", "bank", "credit", "debit", "card", "cvv", "expiry", 
            "expiration", "payment", "salary", "income", "wage", "net_worth", 
            "transaction", "balance", "invoice", "tax", "financial"
        ],
        
        # Demographic Information
        "demographic_info": [
            "age", "birthday", "birth", "date", "gender", "sex", "race", 
            "ethnicity", "nationality", "origin", "language", "marital", 
            "family", "household", "education", "occupation", "employment"
        ],
        
        # Health Information (HIPAA)
        "health_info": [
            "health", "medical", "patient", "diagnosis", "treatment", 
            "prescription", "medication", "disease", "condition", "allergy", 
            "disability", "insurance", "doctor", "physician", "hospital", 
            "clinic", "provider", "record", "mrn"
        ],
        
        # Biometric Data
        "biometric_data": [
            "biometric", "fingerprint", "retina", "iris", "face", "facial", 
            "voice", "dna", "genetic", "blood", "biosample"
        ],
        
        # Online & Device Information
        "digital_info": [
            "ip", "ipaddress", "mac", "device", "browser", "cookie", "token", 
            "session", "login", "password", "credential", "pin", "url", 
            "website", "tracking", "geolocation", "gps", "latitude", "longitude"
        ],
        
        # Protected Class Attributes
        "protected_attributes": [
            "religion", "belief", "political", "party", "union", "association", 
            "orientation", "preference", "sexuality"
        ]
    }
    
    # Flatten the categories into a single list of patterns
    all_patterns = []
    for category, patterns in sensitive_data_categories.items():
        all_patterns.extend(patterns)
    
    # First pass: Check column names for pattern matches
    sensitive_columns = []
    for col in data.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in all_patterns):
            sensitive_columns.append(col)
    
    # Second pass: Content-based analysis for columns not identified by name
    for col in data.columns:
        if col in sensitive_columns:
            continue
        
        # Skip columns with too many unique values (likely IDs or free text)
        nunique = data[col].nunique()
        if nunique > min(100, len(data) * 0.5):
            continue
            
        # Check text columns for potential patterns
        if data[col].dtype == 'object':
            # Sample some values to check (avoid checking entire large datasets)
            sample_size = min(1000, len(data))
            sample = data[col].dropna().astype(str).sample(
                sample_size, replace=True
            )
            
            # Check for email patterns
            email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
            has_emails = sample.str.match(email_pattern).any()
            
            # Check for phone number patterns (simple version)
            phone_pattern = (
                r'(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|'
                r'\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{10})'
            )
            has_phones = sample.str.match(phone_pattern).any()
            
            # Check for potential names (capitalized words not in common vocab)
            name_pattern = r'^[A-Z][a-z]+(\s[A-Z][a-z]+)+$'
            has_names = sample.str.match(name_pattern).mean() > 0.7  # If majority match
            
            if has_emails or has_phones or has_names:
                sensitive_columns.append(col)
    
    return sensitive_columns


def _apply_k_anonymity(
    data: pd.DataFrame,
    sensitive_columns: List[str],
    k: int = 5,
    quasi_identifier_columns: Optional[List[str]] = None,
    categorical_generalization_levels: Optional[Dict[str, Dict[str, str]]] = None,
    numeric_generalization_strategy: str = "equal_width",
    max_generalization_iterations: int = 5,
    suppression_threshold: float = 0.05,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Apply k-anonymity to the dataset using generalization and suppression.
    
    K-anonymity ensures that each combination of quasi-identifier values appears 
    at least k times in the dataset, making it difficult to re-identify individuals.

    Args:
        data: The dataset to anonymize
        sensitive_columns: Columns containing sensitive information that should be preserved
        k: The k value for k-anonymity (minimum size of equivalent classes)
        quasi_identifier_columns: Columns that could be used for re-identification
            (if None, all non-sensitive columns are considered quasi-identifiers)
        categorical_generalization_levels: Dictionary mapping categorical columns to
            their generalization hierarchies. Format: {column: {value: generalized_value}}
        numeric_generalization_strategy: Strategy for generalizing numeric columns
            Options: "equal_width", "equal_frequency", "mdlp"
        max_generalization_iterations: Maximum number of generalization iterations
        suppression_threshold: Maximum fraction of records that can be suppressed
        **kwargs: Additional parameters

    Returns:
        The k-anonymized dataset
    """
    import numpy as np
    from collections import defaultdict
    
    # Make a copy of the dataset to avoid modifying the original
    anonymized_data = data.copy()
    
    # If quasi-identifiers are not provided, use all non-sensitive columns
    if quasi_identifier_columns is None:
        quasi_identifier_columns = [
            col for col in anonymized_data.columns 
            if col not in sensitive_columns
        ]
    
    # Separate categorical and numerical quasi-identifiers
    categorical_qi = []
    for col in quasi_identifier_columns:
        is_object = pd.api.types.is_object_dtype(anonymized_data[col])
        is_categorical = pd.api.types.is_categorical_dtype(anonymized_data[col])
        if is_object or is_categorical:
            categorical_qi.append(col)
    
    numerical_qi = [
        col for col in quasi_identifier_columns 
        if pd.api.types.is_numeric_dtype(anonymized_data[col]) and 
        col not in categorical_qi
    ]
    
    # Create default categorical generalization levels if not provided
    if categorical_generalization_levels is None:
        categorical_generalization_levels = {}
        for col in categorical_qi:
            # Count frequencies of each value
            value_counts = anonymized_data[col].value_counts()
            
            # Create generalization levels based on frequency
            # Values with counts < k are grouped together
            generalization_map = {}
            for value, count in value_counts.items():
                if count >= k:
                    generalization_map[value] = str(value)
                else:
                    generalization_map[value] = "[GENERALIZED]"
                    
            categorical_generalization_levels[col] = generalization_map
    
    # Track if we've achieved k-anonymity
    k_anonymity_achieved = False
    
    # Initial generalization levels for numeric columns (0 = no generalization)
    numeric_generalization_levels = {col: 0 for col in numerical_qi}
    
    # Maximum generalization levels for numeric columns
    max_numeric_levels = 5  # We'll allow up to 5 levels of generalization
    
    # Set of suppressed rows (indices)
    suppressed_indices = set()
    
    # Iterative generalization process
    for iteration in range(max_generalization_iterations):
        if k_anonymity_achieved:
            break
            
        # Apply current generalizations
        generalized_data = anonymized_data.copy()
        
        # Apply categorical generalizations
        for col in categorical_qi:
            if col in categorical_generalization_levels:
                generalized_data[col] = generalized_data[col].map(
                    categorical_generalization_levels[col]
                ).fillna("[UNKNOWN]")
        
        # Apply numeric generalizations based on current levels
        for col in numerical_qi:
            level = numeric_generalization_levels[col]
            if level > 0:
                # Calculate bin width based on level
                if numeric_generalization_strategy == "equal_width":
                    col_min = generalized_data[col].min()
                    col_max = generalized_data[col].max()
                    bin_width = (col_max - col_min) / (10 ** (max_numeric_levels - level))
                    
                    # Apply binning
                    generalized_data[col] = (
                        np.floor((generalized_data[col] - col_min) / bin_width) * bin_width + col_min
                    )
                elif numeric_generalization_strategy == "equal_frequency":
                    # Use quantile-based binning
                    n_bins = 10 ** (max_numeric_levels - level)
                    bins = np.unique(
                        np.percentile(
                            generalized_data[col].dropna(), 
                            np.linspace(0, 100, n_bins + 1)
                        )
                    )
                    bin_labels = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
                    
                    generalized_data[col] = pd.cut(
                        generalized_data[col], 
                        bins=bins, 
                        labels=bin_labels,
                        include_lowest=True
                    )
        
        # Count frequencies of each quasi-identifier combination
        qi_columns = categorical_qi + numerical_qi
        equivalence_classes = generalized_data.groupby(qi_columns).size()
        
        # Identify records in equivalence classes smaller than k
        small_classes = equivalence_classes[equivalence_classes < k].index
        
        # Convert index to list of tuples if it's a MultiIndex
        if isinstance(small_classes, pd.MultiIndex):
            small_classes = list(small_classes)
        else:
            # Handle the case where there's only one QI column
            small_classes = [(c,) for c in small_classes]
        
        # Find records in small equivalence classes for potential suppression
        small_class_records = []
        for small_class in small_classes:
            mask = pd.Series(True, index=generalized_data.index)
            for i, col in enumerate(qi_columns):
                mask &= generalized_data[col] == small_class[i]
            small_class_records.extend(generalized_data[mask].index.tolist())
        
        # If no small classes exist, we've achieved k-anonymity
        if not small_class_records:
            k_anonymity_achieved = True
            break
        
        # If we would need to suppress too many records, increase generalization
        if len(small_class_records) / len(generalized_data) > suppression_threshold:
            # Determine which QI attribute to generalize next
            # Strategy: generalize the attribute with the most small classes
            qi_small_class_counts = defaultdict(int)
            
            for small_class in small_classes:
                for i, col in enumerate(qi_columns):
                    qi_small_class_counts[col] += 1
            
            # Find the QI with the most small classes
            if qi_small_class_counts:
                next_qi_to_generalize = max(
                    qi_small_class_counts.items(), 
                    key=lambda x: x[1]
                )[0]
                
                # Increase generalization level
                if next_qi_to_generalize in numerical_qi:
                    current_level = numeric_generalization_levels[next_qi_to_generalize]
                    if current_level < max_numeric_levels:
                        numeric_generalization_levels[next_qi_to_generalize] += 1
                elif next_qi_to_generalize in categorical_qi:
                    # For categorical, merge least frequent values
                    col = next_qi_to_generalize
                    value_counts = anonymized_data[col].value_counts()
                    least_frequent = value_counts.tail(max(2, int(len(value_counts) * 0.2))).index.tolist()
                    
                    # Update generalization map to merge these values
                    gen_map = categorical_generalization_levels.get(col, {})
                    for val in least_frequent:
                        gen_map[val] = f"GROUP_{iteration}"
                    categorical_generalization_levels[col] = gen_map
            else:
                # If we can't determine which QI to generalize, try generalizing all
                for col in numerical_qi:
                    if numeric_generalization_levels[col] < max_numeric_levels:
                        numeric_generalization_levels[col] += 1
                
                # Also generalize more categorical values
                for col in categorical_qi:
                    value_counts = anonymized_data[col].value_counts()
                    # Generalize bottom 20% of values
                    cutoff = int(len(value_counts) * 0.8)
                    if cutoff > 0:
                        least_frequent = value_counts.iloc[cutoff:].index.tolist()
                        gen_map = categorical_generalization_levels.get(col, {})
                        for val in least_frequent:
                            gen_map[val] = f"GROUP_{iteration}"
                        categorical_generalization_levels[col] = gen_map
                    
            # Don't set k_anonymity_achieved flag here - need to verify on next iteration
        else:
            # We can suppress the records and achieve k-anonymity
            suppressed_indices.update(small_class_records)
            # Don't set k_anonymity_achieved flag until we verify suppression worked
    
    # Apply final generalizations and suppressions
    result_data = anonymized_data.copy()
    
    # Apply categorical generalizations
    for col in categorical_qi:
        if col in categorical_generalization_levels:
            result_data[col] = result_data[col].map(
                categorical_generalization_levels[col]
            ).fillna("[UNKNOWN]")
    
    # Apply numeric generalizations
    for col in numerical_qi:
        level = numeric_generalization_levels[col]
        if level > 0:
            col_min = result_data[col].min()
            col_max = result_data[col].max()
            
            if numeric_generalization_strategy == "equal_width":
                bin_width = (col_max - col_min) / (10 ** (max_numeric_levels - level))
                result_data[col] = (
                    np.floor((result_data[col] - col_min) / bin_width) * bin_width + col_min
                )
            elif numeric_generalization_strategy == "equal_frequency":
                n_bins = 10 ** (max_numeric_levels - level)
                bins = np.unique(
                    np.percentile(
                        result_data[col].dropna(), 
                        np.linspace(0, 100, n_bins + 1)
                    )
                )
                bin_labels = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
                
                result_data[col] = pd.cut(
                    result_data[col], 
                    bins=bins, 
                    labels=bin_labels,
                    include_lowest=True
                )
    
    # Apply suppressions
    if suppressed_indices:
        for idx in suppressed_indices:
            for col in qi_columns:
                result_data.loc[idx, col] = "[SUPPRESSED]"
    
    # Check if the suppressed group has fewer than k members
    suppressed_mask = (result_data[qi_columns] == "[SUPPRESSED]").all(axis=1)
    num_suppressed = suppressed_mask.sum()
    if 0 < num_suppressed < k:
        # Remove the suppressed records
        result_data = result_data[~suppressed_mask]
    
    # Final verification - ensure all equivalence classes have at least k members
    qi_columns = categorical_qi + numerical_qi
    final_counts = result_data.groupby(qi_columns).size()
    small_classes = final_counts[final_counts < k]
    
    # If we still have small classes, apply additional suppression
    if not small_classes.empty:
        # Identify records in remaining small classes
        for small_class_idx, count in small_classes.items():
            mask = pd.Series(True, index=result_data.index)
            
            # Handle both MultiIndex and regular Index
            if isinstance(small_class_idx, tuple):
                for i, col in enumerate(qi_columns):
                    if i < len(small_class_idx):  # Ensure index is in bounds
                        mask &= result_data[col] == small_class_idx[i]
            else:
                # Single QI column case
                mask &= result_data[qi_columns[0]] == small_class_idx
                
            # Suppress these records
            for idx in result_data[mask].index:
                for col in qi_columns:
                    result_data.loc[idx, col] = "[SUPPRESSED]"
    
    # After final suppression, check and remove small "[SUPPRESSED]" groups
    suppressed_mask = (result_data[qi_columns] == "[SUPPRESSED]").all(axis=1)
    num_suppressed = suppressed_mask.sum()
    if 0 < num_suppressed < k:
        result_data = result_data[~suppressed_mask]

    # Final cleanup: remove any remaining groups smaller than k
    final_counts = result_data.groupby(qi_columns).size()
    small_classes = final_counts[final_counts < k]
    if not small_classes.empty:
        indices_to_remove = []
        for small_class_idx, count in small_classes.items():
            mask = pd.Series(True, index=result_data.index)
            if isinstance(small_class_idx, tuple):
                for i, col in enumerate(qi_columns):
                    if i < len(small_class_idx):
                        mask &= result_data[col] == small_class_idx[i]
            else:
                mask &= result_data[qi_columns[0]] == small_class_idx
            indices_to_remove.extend(result_data[mask].index.tolist())
        result_data = result_data.drop(indices_to_remove)

    return result_data


def _apply_pseudonymization(
    data: pd.DataFrame,
    sensitive_columns: List[str],
    strategy: str = "hash",
    preserve_format: bool = True,
    salt: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Apply pseudonymization to the dataset using modern techniques.

    This implementation provides multiple pseudonymization strategies while preserving
    data characteristics and format where appropriate. It supports:
    - Hash-based pseudonymization (default)
    - Format-preserving encryption
    - Deterministic pseudonymization with salt
    - Custom mapping with format preservation

    Args:
        data: The dataset to anonymize
        sensitive_columns: Columns containing sensitive information
        strategy: Pseudonymization strategy to use. Options:
            - "hash": Hash-based pseudonymization (default)
            - "fpe": Format-preserving encryption
            - "deterministic": Deterministic pseudonymization with salt
            - "custom": Custom mapping with format preservation
        preserve_format: Whether to preserve the format of the original values
        salt: Optional salt for deterministic pseudonymization
        **kwargs: Additional parameters for specific strategies

    Returns:
        The pseudonymized dataset

    Raises:
        ValueError: If an unsupported strategy is specified or required parameters
                   are missing
    """

    import hashlib
    import re
    from typing import Dict, Optional, Pattern
    import hmac
    import struct
    
    # Import the key management module for secure key handling
    from .key_management import get_encryption_key

    def _generate_hash(value: str, salt: Optional[str] = None) -> str:
        """Generate a deterministic hash for a value."""
        if salt:
            value = f"{value}{salt}"
        return hashlib.sha256(value.encode()).hexdigest()[:32]

    def _preserve_format(
        original: str,
        pseudonym: str,
        format_pattern: Optional[Pattern[str]] = None,
    ) -> str:
        """Preserve the format of the original value in the pseudonym."""
        if not format_pattern:
            # Default patterns for common formats
            if re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$", original):
                # Email format
                local_part, domain = original.split("@")
                return f"{pseudonym[:len(local_part)]}@{domain}"
            elif re.match(r"^\+?[\d\s()-]{10,}$", original):
                # Phone number format
                # Just use the format of the original, replacing each digit with 'X'
                return re.sub(r"\d", "X", original)
            elif re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+$", original):
                # Name format
                parts = original.split()
                return " ".join(p[:1] + "." for p in parts)
            else:
                return pseudonym
        else:
            # Use provided format pattern
            return re.sub(
                format_pattern, 
                lambda m: pseudonym[:len(m.group())], 
                original
            )

    def _deterministic_pseudonym(
        value: str,
        salt: str,
        format_pattern: Optional[Pattern[str]] = None,
    ) -> str:
        """Generate a deterministic pseudonym with format preservation."""
        pseudonym = _generate_hash(value, salt)
        return _preserve_format(value, pseudonym, format_pattern)

    def _fpe_pseudonym(
        value: str,
        format_pattern: Optional[Pattern[str]] = None,
        key: Optional[bytes] = None,
    ) -> str:
        """
        Generate a format-preserving encrypted pseudonym using FF3-1 algorithm.
        
        FF3-1 is a NIST-approved format-preserving encryption method that 
        preserves the format of the input data. This implementation uses a 
        simplified version of the algorithm.
        
        Args:
            value: The value to encrypt
            format_pattern: Optional regex pattern to preserve specific formats
            key: Encryption key (generates a random one if not provided)
            
        Returns:
            A format-preserving encrypted version of the input value
        """
        # If the value is too short or not suitable for FPE, use hash instead
        if len(value) < 4:
            return _preserve_format(value, _generate_hash(value), format_pattern)
            
        # Generate key if not provided
        if key is None:
            # Use secure key management to retrieve the master key
            master_key_name = kwargs.get('master_key_name', 'fpe_master_key')
            key = get_encryption_key(key_name=master_key_name, encoding="bytes")
        
        # Detect the type of data to preserve format appropriately
        if value.isdigit():
            # For numeric data (like credit card numbers, SSNs)
            chars = '0123456789'
            radix = 10
        elif value.isalpha():
            # For alphabetic data
            if value.isupper():
                chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            elif value.islower():
                chars = 'abcdefghijklmnopqrstuvwxyz'
            else:
                chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            radix = len(chars)
        else:
            # For alphanumeric data
            chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            radix = len(chars)
        
        # Map input to indices in the character set
        indices = [chars.find(c) for c in value if c in chars]
        if not indices:
            # If no valid characters, fall back to hash
            return _preserve_format(value, _generate_hash(value), format_pattern)
        
        # Generate a tweak value (like an IV in block ciphers)
        tweak = hmac.new(key, value.encode(), hashlib.sha256).digest()[:8]
        tweak_value = struct.unpack('>Q', tweak)[0]
        
        # Simple Feistel network for FPE (simplified compared to full FF3-1)
        rounds = 8
        half_size = len(indices) // 2
        left = indices[:half_size]
        right = indices[half_size:]
        
        for round_num in range(rounds):
            # Round function based on HMAC
            round_key = key + struct.pack('>B', round_num)
            hmac_obj = hmac.new(round_key, digestmod=hashlib.sha256)
            
            # Feed the right half and tweak into the round function
            hmac_obj.update(struct.pack(f'>{len(right)}B', *right))
            hmac_obj.update(struct.pack('>Q', tweak_value))
            digest = hmac_obj.digest()
            
            # Convert digest to a number and use it to modify the left half
            digest_int = int.from_bytes(digest, byteorder='big')
            
            # Apply the modification to each element of the left half
            new_left = []
            for i, val in enumerate(left):
                # Use a different byte from digest for each position
                shift = digest[i % len(digest)] % radix
                new_val = (val + shift) % radix
                new_left.append(new_val)
            
            # Swap left and right for the next round
            left, right = right, new_left
        
        # Final swap if rounds is even
        if rounds % 2 == 0:
            left, right = right, left
        
        # Combine the halves and convert back to characters
        result_indices = left + right
        result = ''.join(chars[idx % radix] for idx in result_indices)
        
        # Preserve the original format
        return _preserve_format(value, result, format_pattern)

    def _custom_pseudonym(
        value: str,
        mapping: Dict[str, str],
        format_pattern: Optional[Pattern[str]] = None,
    ) -> str:
        """Generate a pseudonym using a custom mapping."""
        if value in mapping:
            return mapping[value]
        pseudonym = _generate_hash(value)
        return _preserve_format(value, pseudonym, format_pattern)

    # Input validation
    if strategy not in ["hash", "fpe", "deterministic", "custom"]:
        raise ValueError(f"Unsupported pseudonymization strategy: {strategy}")
    if strategy == "deterministic" and not salt:
        raise ValueError("Salt is required for deterministic pseudonymization")
    if strategy == "custom" and "mapping" not in kwargs:
        raise ValueError("Custom mapping is required for custom strategy")

    # Make a copy of the dataset
    pseudonymized_data = data.copy()

    # Process each sensitive column
    for col in sensitive_columns:
        if col not in pseudonymized_data.columns:
            continue

        # Get format pattern if specified
        format_pattern = kwargs.get("format_pattern")
        if isinstance(format_pattern, str):
            format_pattern = re.compile(format_pattern)

        # Process based on strategy
        if strategy == "hash":
            pseudonymized_data[col] = pseudonymized_data[col].map(
                lambda x: _preserve_format(
                    x, _generate_hash(str(x)), format_pattern
                ) if pd.notna(x) else x
            )
        elif strategy == "fpe":
            pseudonymized_data[col] = pseudonymized_data[col].map(
                lambda x: _fpe_pseudonym(str(x), format_pattern)
                if pd.notna(x) else x
            )
        elif strategy == "deterministic":
            pseudonymized_data[col] = pseudonymized_data[col].map(
                lambda x: _deterministic_pseudonym(
                    str(x), salt, format_pattern
                ) if pd.notna(x) else x
            )
        else:  # custom
            mapping = kwargs["mapping"]
            pseudonymized_data[col] = pseudonymized_data[col].map(
                lambda x: _custom_pseudonym(str(x), mapping, format_pattern)
                if pd.notna(x) else x
            )

    return pseudonymized_data


def _apply_data_masking(
    data: pd.DataFrame,
    sensitive_columns: List[str],
    masking_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    default_strategy: str = "character",
    preserve_format: bool = True,
    preserve_statistics: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Apply data masking to the dataset with advanced configuration options.
    
    This implementation provides multiple masking strategies customized for different
    data types and use cases. It supports format preservation, statistical property
    preservation, and column-specific masking rules.
    
    Args:
        data: The dataset to anonymize
        sensitive_columns: Columns containing sensitive information
        masking_rules: Dictionary mapping column names to masking configurations.
            Format: {column_name: {strategy: str, **strategy_params}}
        default_strategy: Default masking strategy when not specified in masking_rules
            Options: "character", "fixed", "regex", "random", "redact", "nullify"
        preserve_format: Whether to preserve the general format of the original values
        preserve_statistics: For numeric columns, whether to preserve statistical 
            properties like mean and range
        **kwargs: Additional parameters for specific strategies
            
    Returns:
        The masked dataset
        
    Example:
        masking_rules = {
            "email": {"strategy": "regex", "pattern": r"(.)(.*)(@.*)"},
            "credit_card": {"strategy": "character", "show_first": 4, "show_last": 4},
            "phone": {"strategy": "fixed", "mask_char": "X", "format": "XXX-XXX-XXXX"},
            "income": {"strategy": "random", "preserve_statistics": True}
        }
    """
    import re
    import random
    import numpy as np
    from datetime import datetime, date
    
    # Make a copy of the dataset
    masked_data = data.copy()
    
    # Default masking rules if none provided
    if masking_rules is None:
        masking_rules = {}

    # Define default format-preserving rules for known sensitive columns
    default_format_preserving_rules = {
        'email': {"strategy": "regex", "pattern": r"(.)(.*)(@.*)", "replacement": r"\1***\3"},
        'ssn': {"strategy": "regex", "pattern": r"(\d{3})-(\d{2})-(\d{4})", "replacement": r"***-\2-****"},
        'zip_code': {"strategy": "character", "show_first": 2, "show_last": 0},
    }
    
    # Helper functions for different masking strategies
    def _character_mask(
        value: Any,
        show_first: int = 0,
        show_last: int = 0,
        mask_char: str = "*",
    ) -> str:
        """Mask characters except for specified first/last characters."""
        if pd.isna(value):
            return value
            
        value_str = str(value)
        if len(value_str) <= show_first + show_last:
            return value_str
            
        masked = (
            value_str[:show_first] + 
            mask_char * (len(value_str) - show_first - show_last) + 
            (value_str[-show_last:] if show_last > 0 else "")
        )
        return masked
    
    def _fixed_mask(
        value,
        format: str = "XXXX-XXXX",
        mask_char: str = "X",
        preserve_special_chars=True
    ) -> str:
        """Apply a fixed mask format to the value."""
        if preserve_special_chars:
            # Create a mask based purely on the format string
            result = ""
            for char_in_format in format:
                if char_in_format == mask_char:
                    # If the format character is the mask character, append the mask character
                    result += mask_char
                else:
                    # Otherwise, keep the character from the format string (e.g., '-')
                    result += char_in_format
            return result
        else:
            # If not preserving special chars, just return the literal format string
            # (This might need refinement later, but fixes the immediate issue)
            return format
    
    def _regex_mask(
        value: Any,
        pattern: str,
        replacement: str,
    ) -> str:
        """Apply regex-based masking."""
        if pd.isna(value):
            return value
            
        value_str = str(value)
        return re.sub(pattern, replacement, value_str)
    
    def _random_mask(
        value: Any,
        char_set: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        preserve_length: bool = True,
        preserve_type: bool = True,
        seed: Optional[int] = None,
    ) -> Any:
        """Replace with random values while preserving characteristics."""
        if pd.isna(value):
            return value
            
        # Set seed for deterministic masking within a column
        if seed is not None:
            # Use both seed and hash of value for deterministic randomization
            value_hash = hash(str(value))
            random.seed(seed + value_hash)
        
        # Handle different types
        if isinstance(value, (int, float)):
            if preserve_statistics and not preserve_type:
                # Return a value that preserves column statistics (handled at column level)
                return value
            
            if preserve_type:
                # Generate a random number of similar magnitude
                if isinstance(value, int):
                    magnitude = len(str(abs(value)))
                    return random.randint(10**(magnitude-1), 10**magnitude - 1) * (1 if value >= 0 else -1)
                else:  # float
                    magnitude = len(str(int(abs(value))))
                    base = random.randint(10**(magnitude-1), 10**magnitude - 1)
                    decimal = random.random()
                    return (base + decimal) * (1 if value >= 0 else -1)
        
        elif isinstance(value, (datetime, date)):
            if preserve_type:
                # Generate a random date/datetime of similar period
                if isinstance(value, datetime):
                    year = random.randint(value.year - 5, value.year + 5)
                    month = random.randint(1, 12)
                    day = random.randint(1, 28)  # Safely within any month
                    hour = random.randint(0, 23)
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    return datetime(year, month, day, hour, minute, second)
                else:  # date
                    year = random.randint(value.year - 5, value.year + 5)
                    month = random.randint(1, 12)
                    day = random.randint(1, 28)
                    return date(year, month, day)
        
        # Default string handling
        value_str = str(value)
        if preserve_length:
            # Replace each character with a random one from the char set
            result = ""
            for c in value_str:
                if c.isalpha():
                    if c.isupper():
                        replacement = random.choice([char for char in char_set if char.isupper()])
                    else:
                        replacement = random.choice([char for char in char_set if char.islower()])
                elif c.isdigit():
                    replacement = random.choice([char for char in char_set if char.isdigit()])
                else:
                    # Preserve special characters
                    replacement = c
                result += replacement
            return result
        else:
            # Generate a completely new random string
            length = random.randint(4, 12)  # Randomize length too
            return ''.join(random.choice(char_set) for _ in range(length))
    
    def _redact_value(value: Any, redaction_text: str = "[REDACTED]") -> str:
        """Complete redaction of the value with a fixed text."""
        if pd.isna(value):
            return value
        return redaction_text
    
    def _nullify_value(value: Any) -> None:
        """Replace the value with NULL/None."""
        return None
    
    # Process each sensitive column
    for col in sensitive_columns:
        if col not in masked_data.columns:
            continue
            
        # Determine the masking configuration for this column
        if preserve_format:
            # Use user-provided rule if available, else default format-preserving rule, else default strategy
            col_config = masking_rules.get(col, default_format_preserving_rules.get(col.lower(), {"strategy": default_strategy}))
        else:
            col_config = masking_rules.get(col, {"strategy": default_strategy})

        strategy = col_config.get("strategy", default_strategy)
        
        # Detect column data type for type-specific handling
        col_dtype = masked_data[col].dtype
        is_numeric = pd.api.types.is_numeric_dtype(col_dtype)
        is_datetime = pd.api.types.is_datetime64_dtype(col_dtype)
        
        # Apply masking based on strategy
        if strategy == "character":
            show_first = col_config.get("show_first", 2)
            show_last = col_config.get("show_last", 2)
            mask_char = col_config.get("mask_char", "*")
            
            masked_data[col] = masked_data[col].apply(
                lambda x: _character_mask(x, show_first, show_last, mask_char)
                if pd.notna(x) else x
            )
            
        elif strategy == "fixed":
            format_str = col_config.get("format", "XXXX-XXXX")
            mask_char = col_config.get("mask_char", "X")
            preserve_special = col_config.get("preserve_special_chars", True)
            
            masked_data[col] = masked_data[col].apply(
                lambda x: _fixed_mask(x, format_str, mask_char, preserve_special)
                if pd.notna(x) else x
            )
            
        elif strategy == "regex":
            pattern = col_config.get("pattern", r"(.)(.*)(@.*)")
            replacement = col_config.get("replacement", r"\1***\3")
            
            masked_data[col] = masked_data[col].apply(
                lambda x: _regex_mask(x, pattern, replacement)
                if pd.notna(x) else x
            )
            
        elif strategy == "random":
            char_set = col_config.get("char_set", (
                "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            ))
            preserve_length = col_config.get("preserve_length", True)
            preserve_type = col_config.get("preserve_type", True)
            seed = col_config.get("seed", 42)  # Fixed seed for deterministic results
            
            # Special handling for numeric columns with preserve_statistics
            if is_numeric and col_config.get("preserve_statistics", preserve_statistics):
                col_mean = masked_data[col].mean()
                col_std = masked_data[col].std()
                col_min = masked_data[col].min()
                col_max = masked_data[col].max()
                
                # Generate random values with same statistical properties
                n_values = len(masked_data)
                if pd.api.types.is_integer_dtype(col_dtype):
                    random_values = np.random.normal(col_mean, col_std, n_values)
                    random_values = np.clip(random_values, col_min, col_max)
                    random_values = np.round(random_values).astype(int)
                else:
                    random_values = np.random.normal(col_mean, col_std, n_values)
                    random_values = np.clip(random_values, col_min, col_max)
                
                masked_data[col] = random_values
            else:
                masked_data[col] = masked_data[col].apply(
                    lambda x: _random_mask(
                        x, char_set, preserve_length, preserve_type, seed
                    )
                    if pd.notna(x) else x
                )
                
        elif strategy == "redact":
            redaction_text = col_config.get("redaction_text", "[REDACTED]")
            masked_data[col] = masked_data[col].apply(
                lambda x: _redact_value(x, redaction_text)
                if pd.notna(x) else x
            )
            
        elif strategy == "nullify":
            masked_data[col] = masked_data[col].map(
                lambda x: _nullify_value(x) if pd.notna(x) else x
            )
            
        else:
            raise ValueError(f"Unsupported masking strategy: {strategy}")
            
    return masked_data


def _apply_generalization(
    data: pd.DataFrame,
    sensitive_columns: List[str],
    generalization_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    default_method: str = "range",
    hierarchical_taxonomies: Optional[Dict[str, Dict[str, Any]]] = None,
    preserve_statistics: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Apply data generalization with configurable approaches for different column types.
    
    Generalization reduces the granularity of data to protect privacy while
    maintaining analytical utility. This implementation provides multiple methods
    tailored to different data types and use cases.
    
    Args:
        data: The dataset to generalize
        sensitive_columns: Columns containing sensitive information
        generalization_rules: Dictionary mapping column names to generalization configs
            Format: {column_name: {method: str, **method_params}}
        default_method: Default generalization method if not specified in rules
            Options: "range", "hierarchy", "binning", "topk", "rounding", "concept"
        hierarchical_taxonomies: Dictionary of pre-defined hierarchical taxonomies
            for categorical data (e.g., location: {city → region → country})
        preserve_statistics: For numeric columns, whether to preserve statistical 
            properties for analytical purposes
        **kwargs: Additional parameters for specific generalization methods
            
    Returns:
        The generalized dataset
        
    Example:
        generalization_rules = {
            "age": {"method": "range", "range_size": 10, "min_bound": 0},
            "zipcode": {"method": "topk", "k": 3, "other_value": "Other"},
            "diagnosis": {"method": "hierarchy", "taxonomy_name": "icd10"},
            "income": {"method": "binning", "num_bins": 5, "strategy": "quantile"}
        }
    """
    import numpy as np
    from datetime import datetime, date
    import re
    import math
    from collections import Counter
    
    # Make a copy of the dataset
    generalized_data = data.copy()
    
    # Default generalization rules if none provided
    if generalization_rules is None:
        generalization_rules = {}
        
    # Default hierarchical taxonomies if none provided
    if hierarchical_taxonomies is None:
        hierarchical_taxonomies = {}
    
    # Helper functions for different generalization methods
    def _apply_range_generalization(
        series: pd.Series,
        range_size: int = 10,
        min_bound: Optional[float] = None,
        max_bound: Optional[float] = None,
        precision: int = 0,
        include_bounds: bool = True,
    ) -> pd.Series:
        """Generalize numeric values into ranges."""
        try:
            series = pd.to_numeric(series)
        except ValueError:
            return series
        
        # Determine min and max bounds
        if min_bound is None:
            min_bound = math.floor(series.min() / range_size) * range_size
        if max_bound is None:
            max_bound = math.ceil(series.max() / range_size) * range_size
            
        # Apply range generalization
        def generalize_value(value):
            if pd.isna(value):
                return value
                
            # Find which range this value belongs to
            lower_bound = math.floor((value - min_bound) / range_size) * range_size + min_bound
            upper_bound = lower_bound + range_size
            
            # Format the range
            if precision > 0:
                lower_str = f"{lower_bound:.{precision}f}"
                upper_str = f"{upper_bound:.{precision}f}"
            else:
                lower_str = str(int(lower_bound))
                upper_str = str(int(upper_bound))
                
            if include_bounds:
                return f"[{lower_str}-{upper_str})"
            else:
                return f"{lower_str} to {upper_str}"
                
        return series.map(lambda x: generalize_value(x) if pd.notna(x) else x)
    
    def _apply_hierarchy_generalization(
        series: pd.Series,
        taxonomy: Dict[str, Any],
        level: int = 1,
        default_value: str = "Other",
    ) -> pd.Series:
        """
        Apply hierarchical generalization using a taxonomy.
        
        The taxonomy can be defined in different ways:
        1. A nested dictionary where keys are values and values are parent/more general categories
        2. A flat dictionary mapping specific values to their generalized form
        """
        if isinstance(taxonomy, dict):
            # Apply the mapping at the requested level
            def generalize_value(value):
                if pd.isna(value):
                    return value
                    
                current_value = str(value)
                # For flat taxonomies
                if level == 1 and current_value in taxonomy:
                    return taxonomy[current_value]
                    
                # For hierarchical taxonomies
                current_level = 0
                while current_level < level:
                    if current_value in taxonomy:
                        current_value = taxonomy[current_value]
                        current_level += 1
                    else:
                        return default_value
                        
                return current_value
                
            return series.map(lambda x: generalize_value(x) if pd.notna(x) else x)
        else:
            return series
    
    def _apply_binning(
        series: pd.Series,
        num_bins: int = 5,
        strategy: str = "equal_width",
        labels: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        Generalize numeric data using binning.
        
        Supports various binning strategies:
        - equal_width: Bins with equal width ranges
        - equal_frequency: Bins with approximately equal number of records
        - kmeans: Bins based on clusters identified by k-means
        """
        try:
            series = pd.to_numeric(series)
        except ValueError:
            return series
                
        # Handle NaN values separately
        non_na_mask = ~series.isna()
        non_na_values = series[non_na_mask]
        
        # Apply different binning strategies
        if strategy == "equal_width":
            bins = np.linspace(
                non_na_values.min(), non_na_values.max(), num_bins + 1
            )
            binned = pd.cut(
                series, bins=bins, labels=labels, include_lowest=True
            )
        elif strategy == "equal_frequency":
            bins = np.percentile(
                non_na_values, np.linspace(0, 100, num_bins + 1)
            )
            # Ensure unique bin edges
            bins = np.unique(bins)
            if len(bins) <= 1:  # Handle case with all same values
                return pd.Series(str(series.iloc[0]) if len(series) > 0 else "", index=series.index)
                
            binned = pd.cut(
                series, bins=bins, labels=labels, include_lowest=True
            )
        elif strategy == "kmeans":
            from sklearn.cluster import KMeans
            
            # Reshape for sklearn
            X = non_na_values.values.reshape(-1, 1)
            
            # Apply kmeans
            kmeans = KMeans(n_clusters=min(num_bins, len(X)), random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Create a mapping from original values to cluster centers
            centers = kmeans.cluster_centers_.flatten()
            
            # Sort clusters by their centers
            sorted_idx = np.argsort(centers)
            sorted_centers = centers[sorted_idx]
            
            # Create mapping from cluster index to sorted index
            cluster_to_bin = {c: i for i, c in enumerate(sorted_idx)}
            
            # Apply mapping to create bins
            binned = pd.Series(index=series.index, dtype='object')
            binned[~non_na_mask] = np.nan
            
            for i, (idx, val) in enumerate(non_na_values.items()):
                bin_idx = cluster_to_bin[clusters[i]]
                if labels is not None and bin_idx < len(labels):
                    binned[idx] = labels[bin_idx]
                else:
                    binned[idx] = f"Bin {bin_idx+1}"
        else:
            # Default to equal width if strategy not recognized
            bins = np.linspace(
                non_na_values.min(), non_na_values.max(), num_bins + 1
            )
            binned = pd.cut(
                series, bins=bins, labels=labels, include_lowest=True
            )
            
        return binned
    
    def _apply_topk_generalization(
        series: pd.Series,
        k: int = 5,
        other_value: str = "Other",
        min_frequency: Optional[float] = None,
    ) -> pd.Series:
        """
        Keep the top-k most frequent values and generalize the rest.
        
        Args:
            series: Input data
            k: Number of distinct values to preserve
            other_value: Value to use for less frequent items
            min_frequency: Minimum frequency required to keep a value
        """
        # Calculate value frequencies
        value_counts = series.value_counts()
        
        # Determine the values to keep
        if min_frequency is not None:
            # Keep values above minimum frequency
            frequent_values = value_counts[value_counts >= min_frequency * len(series)].index.tolist()
            # If too many values meet frequency threshold, take top k
            if len(frequent_values) > k:
                frequent_values = value_counts.nlargest(k).index.tolist()
        else:
            # Keep top k most frequent values
            frequent_values = value_counts.nlargest(k).index.tolist()
            
        # Apply generalization
        return series.map(
            lambda x: x if pd.isna(x) or x in frequent_values else other_value
        )
    
    def _apply_rounding(
        series: pd.Series,
        base: float = 10.0,
        precision: int = 0,
        rounding_method: str = "nearest",
    ) -> pd.Series:
        """
        Generalize numeric data by rounding to a specified base.
        
        Args:
            series: Input numeric data
            base: The base to round to (e.g., 5 rounds to nearest 5)
            precision: Number of decimal places to keep
            rounding_method: One of 'nearest', 'floor', 'ceiling'
        """
        try:
            series = pd.to_numeric(series)
        except ValueError:
            return series
                
        # Apply different rounding methods
        def round_value(value):
            if pd.isna(value):
                return value
            if rounding_method == "nearest":
                rounded = round(value / base) * base
            elif rounding_method == "floor":
                rounded = math.floor(value / base) * base
            elif rounding_method == "ceiling":
                rounded = math.ceil(value / base) * base
            else:
                rounded = round(value / base) * base
                
            # Apply precision
            if precision == 0:
                return int(rounded)
            else:
                return round(rounded, precision)
                
        return series.map(lambda x: round_value(x) if pd.notna(x) else x)
    
    def _apply_concept_hierarchy(
        series: pd.Series,
        concepts: Dict[str, List[str]],
        default_concept: str = "Other",
    ) -> pd.Series:
        """
        Apply concept hierarchies to categorical data.
        
        Args:
            series: Input categorical data
            concepts: Dictionary mapping concepts to lists of values that belong to them
            default_concept: Default concept for values not in any defined concept
        """
        # Create reverse mapping from values to concepts
        value_to_concept = {}
        for concept, values in concepts.items():
            for value in values:
                value_to_concept[value] = concept
                
        # Apply the concept mapping
        return series.map(
            lambda x: value_to_concept.get(x, default_concept) if pd.notna(x) else x
        )
    
    def _apply_date_generalization(
        series: pd.Series,
        level: str = "month",
        format: Optional[str] = None,
    ) -> pd.Series:
        """
        Generalize date/time data by reducing precision.
        
        Args:
            series: Input date/time data
            level: Level of generalization - 'year', 'month', 'quarter', 'week', 'day'
            format: Optional format string for output
        """
        try:
            series = pd.to_datetime(series)
        except ValueError:
            return series
                
        def generalize_date(value):
            if pd.isna(value):
                return value
            if level == "year":
                if format:
                    return value.strftime(format)
                return value.strftime("%Y")
            elif level == "month":
                if format:
                    return value.strftime(format)
                return value.strftime("%Y-%m")
            elif level == "quarter":
                year = value.year
                quarter = (value.month - 1) // 3 + 1
                return f"{year}-Q{quarter}"
            elif level == "week":
                if format:
                    return value.strftime(format)
                year = value.isocalendar()[0]
                week = value.isocalendar()[1]
                return f"{year}-W{week:02d}"
            elif level == "day":
                if format:
                    return value.strftime(format)
                return value.strftime("%Y-%m-%d")
            else:  # Default to month
                if format:
                    return value.strftime(format)
                return value.strftime("%Y-%m")
                
        return series.map(lambda x: generalize_date(x) if pd.notna(x) else x)
    
    def _apply_string_generalization(
        series: pd.Series,
        method: str = "prefix",
        length: int = 1,
        replacement: str = "...",
    ) -> pd.Series:
        """
        Generalize string data by keeping partial information.
        
        Args:
            series: Input string data
            method: One of 'prefix', 'suffix', 'redact'
            length: Number of characters to keep for prefix/suffix methods
            replacement: String to use for redacted portion
        """
        def generalize_string(value):
            if pd.isna(value):
                return value
                
            value_str = str(value)
            if method == "prefix":
                if len(value_str) <= length:
                    return value_str
                return value_str[:length] + replacement
            elif method == "suffix":
                if len(value_str) <= length:
                    return value_str
                return replacement + value_str[-length:]
            elif method == "redact":
                if len(value_str) <= length * 2:
                    return value_str
                return value_str[:length] + replacement + value_str[-length:]
            else:
                return value_str
                
        return series.map(lambda x: generalize_string(x) if pd.notna(x) else x)
    
    # Process each sensitive column
    for col in sensitive_columns:
        if col not in generalized_data.columns:
            continue
            
        # Get generalization configuration for this column
        col_config = generalization_rules.get(col, {"method": default_method})
        method = col_config.get("method", default_method)
        
        # Detect column data type for type-specific handling
        col_dtype = generalized_data[col].dtype
        is_numeric = pd.api.types.is_numeric_dtype(col_dtype)
        is_datetime = pd.api.types.is_datetime64_dtype(col_dtype)
        is_categorical = pd.api.types.is_categorical_dtype(col_dtype)
        is_string = pd.api.types.is_string_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype)
        
        # Apply generalization based on method and data type
        if method == "range" and is_numeric:
            range_size = col_config.get("range_size", 10)
            min_bound = col_config.get("min_bound", None)
            max_bound = col_config.get("max_bound", None)
            precision = col_config.get("precision", 0)
            include_bounds = col_config.get("include_bounds", True)
            
            generalized_data[col] = _apply_range_generalization(
                generalized_data[col], range_size, min_bound, 
                max_bound, precision, include_bounds
            )
            
        elif method == "hierarchy" and (is_string or is_categorical):
            taxonomy_name = col_config.get("taxonomy_name")
            level = col_config.get("level", 1)
            default_value = col_config.get("default_value", "Other")
            
            # Get the taxonomy
            if taxonomy_name and taxonomy_name in hierarchical_taxonomies:
                taxonomy = hierarchical_taxonomies[taxonomy_name]
            else:
                taxonomy = col_config.get("taxonomy", {})
                
            generalized_data[col] = _apply_hierarchy_generalization(
                generalized_data[col], taxonomy, level, default_value
            )
            
        elif method == "binning" and is_numeric:
            num_bins = col_config.get("num_bins", 5)
            strategy = col_config.get("strategy", "equal_width")
            labels = col_config.get("labels", None)
            
            generalized_data[col] = _apply_binning(
                generalized_data[col], num_bins, strategy, labels
            )
            
        elif method == "topk" and (is_string or is_categorical):
            k = col_config.get("k", 5)
            other_value = col_config.get("other_value", "Other")
            min_frequency = col_config.get("min_frequency", None)
            
            generalized_data[col] = _apply_topk_generalization(
                generalized_data[col], k, other_value, min_frequency
            )
            
        elif method == "rounding" and is_numeric:
            base = col_config.get("base", 10.0)
            precision = col_config.get("precision", 0)
            rounding_method = col_config.get("rounding_method", "nearest")
            
            generalized_data[col] = _apply_rounding(
                generalized_data[col], base, precision, rounding_method
            )
            
        elif method == "concept" and (is_string or is_categorical):
            concepts = col_config.get("concepts", {})
            default_concept = col_config.get("default_concept", "Other")
            
            generalized_data[col] = _apply_concept_hierarchy(
                generalized_data[col], concepts, default_concept
            )
            
        elif method == "date" and is_datetime:
            level = col_config.get("level", "month")
            format_str = col_config.get("format", None)
            
            generalized_data[col] = _apply_date_generalization(
                generalized_data[col], level, format_str
            )
            
        elif method == "string" and is_string:
            string_method = col_config.get("string_method", "prefix")
            length = col_config.get("length", 1)
            replacement = col_config.get("replacement", "...")
            
            generalized_data[col] = _apply_string_generalization(
                generalized_data[col], string_method, length, replacement
            )
            
        # Fallback generalizations based on data type
        elif is_numeric:
            # Default numeric generalization: rounding
            base = col_config.get("base", 10.0)
            generalized_data[col] = _apply_rounding(generalized_data[col], base)
            
        elif is_datetime:
            # Default datetime generalization: by month
            generalized_data[col] = _apply_date_generalization(generalized_data[col])
            
        elif is_string or is_categorical:
            # Default string generalization: prefix
            generalized_data[col] = _apply_string_generalization(generalized_data[col])
            
    return generalized_data