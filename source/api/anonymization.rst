=================
Anonymization API
=================

.. module:: secureml.anonymization

This module provides tools to anonymize sensitive data before using it in machine learning workflows.

Main Function
------------

.. autofunction:: anonymize

Anonymization Techniques
-----------------------

K-Anonymity
~~~~~~~~~~~

K-anonymity ensures that each combination of quasi-identifier values appears at least k times in the dataset, making it difficult to re-identify individuals.

**Internal Implementation**

.. autofunction:: _apply_k_anonymity

Pseudonymization
~~~~~~~~~~~~~~~~

Pseudonymization replaces identifying data with artificial identifiers while preserving data characteristics.

**Internal Implementation**

.. autofunction:: _apply_pseudonymization

Data Masking
~~~~~~~~~~~~

Data masking hides specific parts of data while preserving its format and potentially statistical properties.

**Internal Implementation**

.. autofunction:: _apply_data_masking

The module implements multiple masking strategies:

- Character masking (show/hide specific portions)
- Fixed masking (with a predefined format)
- Regex-based masking (pattern replacement)
- Random masking (with statistical preservation)
- Redaction (complete replacement)
- Nullification (data removal)

Generalization
~~~~~~~~~~~~~~

Generalization reduces the granularity of data to protect privacy while maintaining analytical utility.

**Internal Implementation**

.. autofunction:: _apply_generalization

The module implements multiple generalization methods:

- Range generalization (for numerical data)
- Hierarchical generalization (using taxonomies)
- Binning (equal width, equal frequency, etc.)
- Top-k generalization (keep only the k most frequent values)
- Rounding (to a specified base)
- Concept hierarchies (map values to higher-level concepts)
- Date generalization (year, month, quarter, etc.)
- String generalization (prefix, suffix, etc.)

Utility Functions
----------------

.. autofunction:: _identify_sensitive_columns

This function automatically identifies columns that likely contain sensitive information based on:

1. Column names matching patterns associated with sensitive data
2. Content analysis (detecting emails, phone numbers, names, etc.)
3. Privacy framework categorizations (GDPR, CCPA, HIPAA, ISO/IEC 27701)
