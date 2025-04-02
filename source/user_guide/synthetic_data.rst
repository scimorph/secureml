====================
Synthetic Data
====================

Synthetic data generation creates artificial data that statistically resembles real data while protecting privacy. SecureML provides several techniques to generate high-quality synthetic data that maintains the utility of the original dataset without exposing sensitive information.

Core Concepts
------------

**Statistical Preservation**: Maintaining the statistical properties of the original data, such as distributions, correlations, and dependencies.

**Privacy Guarantees**: Ensuring that the synthetic data doesn't leak information about specific individuals in the original dataset.

**Utility Preservation**: Ensuring models trained on synthetic data perform similarly to those trained on real data.

Basic Usage
----------

Generating Synthetic Data
^^^^^^^^^^^^^^^^^^^^^^^

The main function for generating synthetic data is ``generate_synthetic_data``:

.. code-block:: python

    from secureml.synthetic import generate_synthetic_data
    
    # Generate synthetic data
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="statistical",  # Options: "simple", "statistical", "sdv-copula", "sdv-ctgan", "sdv-tvae", "gan", "copula"
        sensitive_columns=["name", "email", "ssn", "phone"],
        seed=42
    )

Automatic Detection of Sensitive Columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SecureML can automatically detect sensitive columns in your data:

.. code-block:: python

    from secureml.synthetic import generate_synthetic_data
    
    # Generate synthetic data with automatic sensitive column detection
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="statistical",
        sensitivity_detection={
            "auto_detect": True,
            "confidence_threshold": 0.7,
            "sample_size": 100
        }
    )

Supported Methods
---------------

SecureML supports several synthetic data generation methods:

Simple Random Sampling
^^^^^^^^^^^^^^^^^^^

Basic method suitable for quick prototyping:

.. code-block:: python

    # Generate synthetic data using simple method
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="simple",
        sensitive_columns=["name", "email", "ssn"]
    )

Statistical Method
^^^^^^^^^^^^^^^

More sophisticated method that preserves statistical relationships:

.. code-block:: python

    # Generate synthetic data using statistical method
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="statistical",
        sensitive_columns=["name", "email", "ssn"],
        preserve_dtypes=True,
        preserve_outliers=True,
        categorical_threshold=20,
        handle_skewness=True,
        seed=42
    )

SDV Integration Methods
^^^^^^^^^^^^^^^^^^^^^

Integration with the Synthetic Data Vault (SDV) library for advanced generation (requires SDV to be installed):

.. code-block:: python

    # Generate synthetic data using SDV's Gaussian Copula
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="sdv-copula",
        sensitive_columns=["name", "email", "ssn"],
        anonymize_fields=True
    )
    
    # Generate synthetic data using SDV's CTGAN
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="sdv-ctgan",
        sensitive_columns=["name", "email", "ssn"],
        anonymize_fields=True,
        epochs=300,
        batch_size=500
    )
    
    # Generate synthetic data using SDV's TVAE
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="sdv-tvae",
        sensitive_columns=["name", "email", "ssn"],
        anonymize_fields=True
    )

GAN-based Method
^^^^^^^^^^^^^

Generative Adversarial Network approach (without requiring SDV):

.. code-block:: python

    # Generate synthetic data using GAN method
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="gan",
        sensitive_columns=["name", "email", "ssn"],
        epochs=300,
        batch_size=32,
        generator_dim=[128, 128],
        discriminator_dim=[128, 128],
        learning_rate=0.001,
        noise_dim=100,
        preserve_dtypes=True
    )

Copula-based Method
^^^^^^^^^^^^^^^

Copula method for capturing variable dependencies:

.. code-block:: python

    # Generate synthetic data using copula method
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="copula",
        sensitive_columns=["name", "email", "ssn"],
        copula_type="gaussian",
        fit_method="ml",
        preserve_dtypes=True,
        handle_missing="mean",
        categorical_threshold=20,
        handle_skewness=True,
        seed=42
    )

Providing Data Schema Instead of Template DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can generate synthetic data using a schema definition instead of an actual DataFrame:

.. code-block:: python

    # Define a schema
    schema = {
        "columns": {
            "age": "int",
            "income": "float",
            "gender": "category",
            "education": "category"
        }
    }
    
    # Generate synthetic data from schema
    synthetic_df = generate_synthetic_data(
        template=schema,
        num_samples=1000,
        method="statistical"
    )

Advanced Usage
-------------

SDV Constraints
^^^^^^^^^^^^

When using SDV methods, you can specify constraints on the generated data:

.. code-block:: python

    # Define constraints for SDV methods
    constraints = [
        {"type": "unique", "columns": ["id"]},
        {"type": "fixed_combinations", "column_names": ["state", "city"]},
        {"type": "inequality", "low_column": "min_salary", "high_column": "max_salary"}
    ]
    
    # Generate synthetic data with constraints
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="sdv-copula",
        constraints=constraints
    )

Handling Sensitive Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function automatically generates realistic but fake data for sensitive columns:

.. code-block:: python

    # Generate synthetic data with sensitive column handling
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=1000,
        method="statistical",
        sensitive_columns=["name", "email", "phone", "ssn", "credit_card"]
    )

Best Practices
-------------

1. **Choose the right method**: Select the generation method based on your data characteristics:
   - For simple datasets with low complexity: "simple"
   - For general-purpose generation with good statistical properties: "statistical"
   - For complex tabular data with mixed types: "sdv-ctgan" or "sdv-tvae"
   - For data with important correlations: "sdv-copula" or "copula"

2. **Automatic sensitive column detection**: When in doubt about which columns are sensitive, use the automatic detection feature.

3. **Seed for reproducibility**: Always set a seed when you need reproducible results.

4. **Evaluate your synthetic data**: Check that the synthetic data preserves important statistical properties while providing sufficient privacy protection.

5. **Balance privacy and utility**: Adjust parameters to find the right balance between privacy protection and synthetic data utility.

Example Workflow
--------------

Complete workflow for generating and checking synthetic data:

.. code-block:: python

    import pandas as pd
    from secureml.synthetic import generate_synthetic_data
    
    # Load original data
    original_df = pd.read_csv("customer_data.csv")
    
    # Generate synthetic data with automatic sensitive column detection
    synthetic_df = generate_synthetic_data(
        template=original_df,
        num_samples=len(original_df),
        method="statistical",
        sensitivity_detection={"auto_detect": True, "confidence_threshold": 0.7},
        seed=42,
        preserve_dtypes=True,
        handle_skewness=True
    )
    
    # Save synthetic data
    synthetic_df.to_csv("synthetic_customer_data.csv", index=False)
    
    # Basic validation - check column distributions
    for col in original_df.select_dtypes(include=['number']).columns:
        print(f"Column: {col}")
        print(f"Original mean: {original_df[col].mean()}, std: {original_df[col].std()}")
        print(f"Synthetic mean: {synthetic_df[col].mean()}, std: {synthetic_df[col].std()}")
        print()

Further Reading
-------------

* :doc:`/api/synthetic_data` - Complete API reference for synthetic data functions
* :doc:`/examples/synthetic_data` - More examples of synthetic data generation techniques 