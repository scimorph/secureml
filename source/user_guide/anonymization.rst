=============
Anonymization
=============

Data anonymization is a critical first step in privacy-preserving machine learning. SecureML provides several techniques to anonymize sensitive data while maintaining its utility for training models.

Core Concepts
------------

SecureML implements multiple anonymization techniques:

* **K-anonymity**: Ensures that each record is indistinguishable from at least k-1 other records
* **L-diversity**: Extends k-anonymity by requiring sensitive attributes to have diverse values
* **T-closeness**: Further extends privacy by ensuring the distribution of sensitive attributes is close to the overall distribution
* **Differential Privacy**: Adds statistical noise to data to provide mathematical privacy guarantees

Basic Usage
----------

The simplest way to anonymize a dataset is using the ``anonymize`` function:

.. code-block:: python

    from secureml.anonymization import anonymize
    
    # Anonymize a pandas DataFrame
    anonymized_df = anonymize(
        df,
        quasi_identifiers=['age', 'zipcode', 'gender'],
        sensitive_attributes=['disease', 'income'],
        k=5,  # k-anonymity parameter
        l=3,  # l-diversity parameter
        t=0.2  # t-closeness parameter
    )

For more fine-grained control, you can use the ``Anonymizer`` class:

.. code-block:: python

    from secureml.anonymization import Anonymizer
    
    anonymizer = Anonymizer(
        quasi_identifiers=['age', 'zipcode', 'gender'],
        sensitive_attributes=['disease', 'income'],
        k=5,
        l=3,
        t=0.2
    )
    
    # Fit the anonymizer to learn data distributions
    anonymizer.fit(df)
    
    # Transform the data
    anonymized_df = anonymizer.transform(df)
    
    # Or do both in one step
    anonymized_df = anonymizer.fit_transform(df)
    
    # Save the anonymizer for future use
    anonymizer.save('my_anonymizer.pkl')
    
    # Load a saved anonymizer
    from secureml.anonymization import load_anonymizer
    loaded_anonymizer = load_anonymizer('my_anonymizer.pkl')

Advanced Techniques
------------------

Attribute Generalization
^^^^^^^^^^^^^^^^^^^^^^^

For categorical and numerical attributes, you can specify custom generalization hierarchies:

.. code-block:: python

    from secureml.anonymization import Anonymizer, NumericGeneralization, CategoricalGeneralization
    
    # Define generalization for age (numeric)
    age_gen = NumericGeneralization(
        bins=[0, 18, 30, 50, 65, 100],
        labels=['0-18', '19-30', '31-50', '51-65', '65+']
    )
    
    # Define generalization for occupation (categorical)
    occupation_gen = CategoricalGeneralization({
        'doctor': 'healthcare',
        'nurse': 'healthcare',
        'teacher': 'education',
        'professor': 'education',
        # ... more mappings
    })
    
    anonymizer = Anonymizer(
        quasi_identifiers=['age', 'zipcode', 'occupation'],
        sensitive_attributes=['disease'],
        generalizations={
            'age': age_gen,
            'occupation': occupation_gen
        }
    )

Column Suppression
^^^^^^^^^^^^^^^^

You can completely suppress columns that are too identifying:

.. code-block:: python

    anonymized_df = anonymize(
        df,
        quasi_identifiers=['age', 'zipcode', 'gender'],
        sensitive_attributes=['disease', 'income'],
        suppressed_attributes=['id', 'name', 'ssn', 'exact_address']
    )

Record Suppression
^^^^^^^^^^^^^^^^

In some cases, certain records may be statistical outliers that are impossible to anonymize without excessive information loss. You can configure how these records are handled:

.. code-block:: python

    anonymizer = Anonymizer(
        quasi_identifiers=['age', 'zipcode', 'gender'],
        sensitive_attributes=['disease'],
        outlier_handling='suppress',  # Options: 'suppress', 'separate_group', 'force_generalize'
        max_suppression_rate=0.05  # Maximum percentage of records that can be suppressed
    )

Differential Privacy Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can combine traditional anonymization with differential privacy for stronger guarantees:

.. code-block:: python

    from secureml.anonymization import anonymize
    
    anonymized_df = anonymize(
        df,
        quasi_identifiers=['age', 'zipcode', 'gender'],
        sensitive_attributes=['disease', 'income'],
        apply_differential_privacy=True,
        epsilon=1.0,  # Privacy budget
        delta=1e-5    # Probability of privacy breach
    )

Utility Metrics
-------------

SecureML provides tools to measure the utility preservation of anonymized data:

.. code-block:: python

    from secureml.anonymization.metrics import information_loss, query_error
    
    # Measure information loss
    loss = information_loss(original_df, anonymized_df)
    print(f"Information loss: {loss:.2f}")
    
    # Measure query error for specific analytics
    error = query_error(
        original_df, 
        anonymized_df,
        query="SELECT AVG(income) FROM df GROUP BY gender"
    )
    print(f"Query error: {error:.2f}")

Best Practices
-------------

1. **Start with minimal quasi-identifiers**: Only include attributes that are truly necessary for identification
2. **Balance privacy and utility**: Higher k, l, and t values provide more privacy but reduce utility
3. **Test with different parameters**: Experiment to find the optimal balance for your specific use case
4. **Verify anonymization**: Use SecureML's tools to verify that your data meets the desired privacy criteria
5. **Combine techniques**: For sensitive applications, use multiple techniques like k-anonymity and differential privacy together

Further Reading
-------------

* :doc:`/api/anonymization` - Complete API reference for anonymization functions
* :doc:`/examples/anonymization` - More examples of anonymization techniques 