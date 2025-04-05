=================
Synthetic Data API
=================

.. module:: secureml.synthetic

This module provides tools to create synthetic datasets that mimic the statistical properties of real data without containing sensitive information, enabling privacy-preserving machine learning development and testing.

Main Functions
--------------

.. autofunction:: generate_synthetic_data

The main function for generating synthetic data from a template dataset:

.. code-block:: python

    from secureml.synthetic import generate_synthetic_data
    import pandas as pd
    
    # Load a real dataset as a template
    template_data = pd.read_csv("patient_data.csv")
    
    # Generate 1000 synthetic samples using the statistical method
    synthetic_data = generate_synthetic_data(
        template=template_data,
        num_samples=1000,
        method="statistical",
        sensitive_columns=["name", "email", "ssn", "address"],
        seed=42  # For reproducibility
    )
    
    # Save the synthetic dataset
    synthetic_data.to_csv("synthetic_patient_data.csv", index=False)

Supported Generation Methods
---------------------------

The module supports several methods for generating synthetic data, each with different characteristics and trade-offs:

1. **Simple Method** (``method="simple"``)

   A basic approach that preserves individual column distributions but not complex relationships between variables.
   
   .. code-block:: python
   
       # Generate simple synthetic data
       simple_synthetic = generate_synthetic_data(
           template=data,
           num_samples=500,
           method="simple"
       )

2. **Statistical Method** (``method="statistical"``)

   A more sophisticated approach that preserves correlations and statistical properties.
   
   .. code-block:: python
   
       # Generate synthetic data preserving statistical properties
       statistical_synthetic = generate_synthetic_data(
           template=data,
           num_samples=500,
           method="statistical",
           handle_skewness=True,
           preserve_outliers=False
       )

3. **SDV Methods** (``method="sdv-copula"``, ``method="sdv-ctgan"``, ``method="sdv-tvae"``)

   Leverages the Synthetic Data Vault library for advanced generative models:
   
   .. code-block:: python
   
       # Generate synthetic data using a Gaussian Copula model
       copula_synthetic = generate_synthetic_data(
           template=data,
           num_samples=500,
           method="sdv-copula",
           constraints=[
               {"type": "unique", "columns": ["id"]}
           ]
       )
       
       # Generate synthetic data using a GAN-based model
       ctgan_synthetic = generate_synthetic_data(
           template=data,
           num_samples=500,
           method="sdv-ctgan",
           epochs=100
       )

4. **Direct Methods** (``method="gan"``, ``method="copula"``)

   Direct implementations of generative adversarial networks and copulas:
   
   .. code-block:: python
   
       # Generate synthetic data using a GAN model
       gan_synthetic = generate_synthetic_data(
           template=data,
           num_samples=500,
           method="gan",
           epochs=300,
           batch_size=32
       )

Sensitive Data Handling
----------------------

The module provides automatic detection and special handling for sensitive columns:

.. code-block:: python

    # Automatic detection of sensitive columns
    synthetic_data = generate_synthetic_data(
        template=data,
        num_samples=1000,
        method="statistical",
        sensitivity_detection={
            "auto_detect": True,
            "confidence_threshold": 0.7,
            "sample_size": 100
        }
    )
    
    # Explicitly specify sensitive columns
    synthetic_data = generate_synthetic_data(
        template=data,
        num_samples=1000,
        method="statistical",
        sensitive_columns=["email", "phone", "ssn", "medical_record"]
    )

Creating Synthetic Data from Schema
---------------------------------

You can also generate synthetic data from a schema definition without a template dataset:

.. code-block:: python

    # Define a schema
    schema = {
        "columns": {
            "age": "int",
            "income": "float",
            "education": "category",
            "marital_status": "category"
        }
    }
    
    # Generate synthetic data from schema
    synthetic_from_schema = generate_synthetic_data(
        template=schema,
        num_samples=1000,
        method="statistical"
    )

Using Constraints with SDV Methods
--------------------------------

When using SDV-based methods, you can apply constraints to the synthetic data:

.. code-block:: python

    # Generate synthetic data with constraints
    synthetic_with_constraints = generate_synthetic_data(
        template=data,
        num_samples=500,
        method="sdv-copula",
        constraints=[
            {"type": "unique", "columns": ["id"]},
            {"type": "fixed_combinations", "column_names": ["city", "state"]},
            {"type": "inequality", "low_column": "start_date", "high_column": "end_date"}
        ]
    )

Helper Functions
--------------

The module also provides several helper functions that are used internally:

- ``_identify_sensitive_columns``: Automatically identifies columns containing sensitive data
- ``_generate_simple_synthetic``: Implements the simple generation method
- ``_generate_statistical_synthetic``: Implements the statistical generation method
- ``_generate_sdv_synthetic``: Implements SDV-based generation methods
- ``_generate_gan_synthetic``: Implements GAN-based generation
- ``_generate_copula_synthetic``: Implements copula-based generation

Best Practices
-------------

1. **Start Simple**: Begin with simpler methods like "simple" or "statistical" before trying more complex models
2. **Evaluate Quality**: Compare synthetic data distributions with the original data
3. **Handle Sensitive Data**: Always specify sensitive columns or enable auto-detection
4. **Set Seed**: Use the seed parameter for reproducible results
5. **Balance Privacy and Utility**: More complex methods may preserve utility better but might have privacy implications
6. **Constraints Matter**: Use constraints with SDV methods to ensure business rules are preserved
