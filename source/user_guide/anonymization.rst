=============
Anonymization
=============

Data anonymization is a critical first step in privacy-preserving machine learning. SecureML provides several techniques to anonymize sensitive data while maintaining its utility for training models.

Core Concepts
------------

SecureML implements multiple anonymization techniques:

* **K-anonymity**: Ensures that each record is indistinguishable from at least k-1 other records
* **Pseudonymization**: Replaces identifying data with artificial identifiers
* **Data Masking**: Hides specific parts of data while preserving its format
* **Generalization**: Replaces specific values with broader categories

Basic Usage
----------

The simplest way to anonymize a dataset is using the ``anonymize`` function:

.. code-block:: python

    from secureml.anonymization import anonymize
    
    # Anonymize a pandas DataFrame
    anonymized_df = anonymize(
        df,
        method="k-anonymity",  # Options: 'k-anonymity', 'pseudonymization', 'data-masking', 'generalization'
        k=5,  # k-anonymity parameter
        sensitive_columns=['disease', 'income']  # List of columns containing sensitive information
    )

If no sensitive columns are specified, the function will attempt to automatically identify them:

.. code-block:: python

    # Automatic identification of sensitive columns
    anonymized_df = anonymize(
        df,
        method="k-anonymity",
        k=5
        # sensitive_columns will be automatically identified
    )

Advanced Techniques
------------------

K-Anonymity
^^^^^^^^^^^^^

K-anonymity ensures that each combination of quasi-identifier values appears at least k times, making it difficult to re-identify individuals:

.. code-block:: python

    from secureml.anonymization import anonymize
    
    anonymized_df = anonymize(
        df,
        method="k-anonymity",
        k=5,
        sensitive_columns=['disease', 'income'],
        quasi_identifier_columns=['age', 'zipcode', 'gender'],  # Columns that could be used for re-identification
        categorical_generalization_levels={  # Custom generalization hierarchies
            'gender': {'Male': 'M', 'Female': 'F', 'Other': 'O'}
        },
        numeric_generalization_strategy="equal_width",  # Options: "equal_width", "equal_frequency", "mdlp"
        max_generalization_iterations=5,  # Maximum number of generalization iterations
        suppression_threshold=0.05  # Maximum fraction of records that can be suppressed
    )

Pseudonymization
^^^^^^^^^^^^^^

Pseudonymization replaces identifying data with artificial identifiers while preserving data characteristics:

.. code-block:: python

    anonymized_df = anonymize(
        df,
        method="pseudonymization",
        sensitive_columns=['email', 'phone', 'name'],
        strategy="hash",  # Options: "hash", "fpe", "deterministic", "custom"
        preserve_format=True,  # Preserve the format of original values
        salt="your-salt-string"  # Salt for deterministic pseudonymization
    )

    # Format-preserving encryption (FPE)
    anonymized_df = anonymize(
        df,
        method="pseudonymization",
        sensitive_columns=['credit_card', 'ssn'],
        strategy="fpe",
        preserve_format=True
    )

    # Custom mapping
    anonymized_df = anonymize(
        df,
        method="pseudonymization",
        sensitive_columns=['city'],
        strategy="custom",
        mapping={
            'New York': 'City A',
            'Los Angeles': 'City B',
            'Chicago': 'City C'
        }
    )

Data Masking
^^^^^^^^^^

Data masking hides specific parts of data while preserving its format:

.. code-block:: python

    anonymized_df = anonymize(
        df,
        method="data-masking",
        sensitive_columns=['email', 'phone', 'ssn'],
        default_strategy="character",  # Default masking strategy
        preserve_format=True,  # Preserve the format of original values
        masking_rules={  # Column-specific masking rules
            'email': {"strategy": "regex", "pattern": r"(.)(.*)(@.*)", "replacement": r"\1***\3"},
            'ssn': {"strategy": "character", "show_first": 0, "show_last": 4, "mask_char": "*"},
            'phone': {"strategy": "fixed", "format": "XXX-XXX-XXXX", "mask_char": "X"}
        }
    )

    # Random masking with statistical preservation
    anonymized_df = anonymize(
        df,
        method="data-masking",
        sensitive_columns=['income', 'age'],
        default_strategy="random",
        preserve_statistics=True  # Preserve statistical properties like mean and range
    )

Generalization
^^^^^^^^^^^^

Generalization reduces data granularity to protect privacy while maintaining analytical utility:

.. code-block:: python

    anonymized_df = anonymize(
        df,
        method="generalization",
        sensitive_columns=['age', 'zipcode', 'income', 'date_of_birth'],
        default_method="range",  # Default generalization method
        generalization_rules={  # Column-specific generalization rules
            'age': {"method": "range", "range_size": 10},
            'zipcode': {"method": "topk", "k": 5, "other_value": "Other"},
            'income': {"method": "binning", "num_bins": 5, "strategy": "equal_frequency"},
            'date_of_birth': {"method": "date", "level": "year"}
        }
    )

Automatic Sensitive Column Detection
---------------------------------

SecureML can automatically identify columns that likely contain sensitive information:

.. code-block:: python

    from secureml.anonymization import anonymize
    
    # Automatically detect and anonymize sensitive columns
    anonymized_df = anonymize(
        df,
        method="k-anonymity",
        k=5
        # No need to specify sensitive_columns
    )

The automatic detection looks for patterns in column names and contents that suggest sensitive data according to privacy frameworks like GDPR, CCPA, and HIPAA.

Best Practices
-------------

1. **Start with minimal sensitive columns**: Only include attributes that are truly necessary to protect
2. **Balance privacy and utility**: Higher k values provide more privacy but reduce utility
3. **Test with different methods**: Experiment to find the optimal method for your specific use case
4. **Combine techniques**: For sensitive applications, consider applying multiple techniques sequentially
5. **Verify anonymization**: Test whether the anonymized data still meets the privacy requirements you need

Further Reading
-------------

* :doc:`/api/anonymization` - Complete API reference for anonymization functions
* :doc:`/examples/anonymization` - More examples of anonymization techniques 