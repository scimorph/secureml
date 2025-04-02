Anonymization Examples
=====================

This section demonstrates various data anonymization techniques using the SecureML library.

.. code-block:: python

    import pandas as pd
    from secureml.anonymization import anonymize, _identify_sensitive_columns

    # Sample Data
    data_list = [
        {'id': 1, 'name': 'Alice Smith', 'age': 30, 'zipcode': '12345', 
         'diagnosis': 'Flu', 'email': 'alice.s@example.com', 
         'income': 60000, 'phone': '555-1234'},
        # ... more records ...
    ]
    df = pd.DataFrame(data_list)

k-Anonymity
----------

k-Anonymity ensures that any combination of quasi-identifiers appears in at least k records.

.. code-block:: python

    # Identify sensitive columns automatically
    sensitive_cols_auto = _identify_sensitive_columns(df)
    
    # Define sensitive and quasi-identifier columns
    sensitive_k_anon = ['id', 'diagnosis', 'income']
    quasi_k_anon = ['age', 'zipcode', 'name', 'email', 'phone']
    
    # Apply k-anonymity
    anonymized_k = anonymize(
        df.copy(), 
        method="k-anonymity", 
        k=2, 
        sensitive_columns=sensitive_k_anon,
        quasi_identifier_columns=quasi_k_anon,
        suppression_threshold=0.2
    )

Pseudonymization
--------------

Pseudonymization replaces sensitive values with pseudonyms while preserving data format.

.. code-block:: python

    sensitive_pseudo = ['name', 'email', 'phone']
    
    # Hash-based pseudonymization
    anonymized_hash = anonymize(
        df.copy(),
        method="pseudonymization",
        sensitive_columns=sensitive_pseudo,
        strategy="hash",
        preserve_format=True
    )
    
    # Deterministic pseudonymization with salt
    anonymized_determ = anonymize(
        df.copy(),
        method="pseudonymization",
        sensitive_columns=sensitive_pseudo,
        strategy="deterministic",
        salt="my-super-secret-salt-123",
        preserve_format=True
    )
    
    # Format-Preserving Encryption (FPE)
    anonymized_fpe = anonymize(
        df.copy(),
        method="pseudonymization",
        sensitive_columns=sensitive_pseudo,
        strategy="fpe"
    )

Data Masking
-----------

Data masking applies specific rules to hide sensitive information while maintaining data structure.

.. code-block:: python

    sensitive_mask = ['email', 'phone', 'income', 'name']
    
    # Define masking rules
    masking_rules = {
        "email": {"strategy": "regex", "pattern": r"(.*)(@.*)", 
                 "replacement": r"masked***\2"},
        "phone": {"strategy": "character", "show_last": 4, 
                 "mask_char": "X"},
        "income": {"strategy": "fixed", "format": "******"},
        "name": {"strategy": "character", "show_first": 1, 
                "mask_char": "."}
    }
    
    # Apply masking
    anonymized_masking = anonymize(
        df.copy(),
        method="data-masking",
        sensitive_columns=sensitive_mask,
        masking_rules=masking_rules,
        preserve_format=False
    )
    
    # Masking with statistic preservation
    anonymized_masking_stats = anonymize(
        df.copy(),
        method="data-masking",
        sensitive_columns=['income'],
        masking_rules={"income": {"strategy": "random", 
                                "preserve_statistics": True}},
        preserve_statistics=True
    )

Generalization
-------------

Generalization replaces specific values with more general categories.

.. code-block:: python

    sensitive_generalize = ['age', 'zipcode', 'diagnosis']
    
    # Define diagnosis hierarchy
    dx_hierarchy = {
        "Flu": "Respiratory",
        "Cold": "Respiratory",
        "Allergy": "Immune",
        "Diabetes": "Endocrine",
        "Hypertension": "Cardiovascular",
        "Headache": "Neurological"
    }
    
    # Define generalization rules
    generalization_rules = {
        "age": {"method": "range", "range_size": 10},
        "zipcode": {"method": "topk", "k": 2, "other_value": "Other_Zip"},
        "diagnosis": {"method": "hierarchy", "taxonomy": dx_hierarchy, 
                     "level": 1}
    }
    
    # Apply generalization
    anonymized_general = anonymize(
        df.copy(),
        method="generalization",
        sensitive_columns=sensitive_generalize,
        generalization_rules=generalization_rules
    )
