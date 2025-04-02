Synthetic Data Generation Examples
================================

This section demonstrates how to generate synthetic data that preserves the statistical properties of your original data while ensuring privacy. SecureML provides multiple methods with different trade-offs between simplicity, statistical accuracy, and privacy protection.

Basic Synthetic Data Generation
-----------------------------

The simplest way to generate synthetic data is using the ``simple`` method:

.. code-block:: python

    import pandas as pd
    from secureml.synthetic import generate_synthetic_data
    
    # Sample data with sensitive information
    data = pd.DataFrame({
        'name': ['John Smith', 'Jane Doe', 'Robert Johnson'],
        'age': [34, 29, 42],
        'gender': ['Male', 'Female', 'Male'],
        'email': ['john.smith@example.com', 'jane.doe@example.com', 'robert.j@example.com'],
        'income': [65000, 72000, 58000],
        'credit_score': [720, 750, 680],
        'zipcode': ['12345', '23456', '34567']
    })
    
    # Generate synthetic data using simple method
    synthetic_data = generate_synthetic_data(
        template=data,
        num_samples=20,
        method="simple",
        sensitive_columns=['name', 'email'],
        seed=42
    )
    
    print("Synthetic data sample:")
    print(synthetic_data.head())

The ``simple`` method generates values by sampling from observed distributions for non-sensitive columns, while using more sophisticated techniques (like Faker) for sensitive columns.

Statistical Method
----------------

For better preservation of statistical relationships, use the ``statistical`` method:

.. code-block:: python

    # Generate synthetic data with statistical method
    statistical_synthetic = generate_synthetic_data(
        template=data,
        num_samples=20,
        method="statistical",
        sensitive_columns=['name', 'email'],
        preserve_dtypes=True,
        preserve_outliers=True,
        categorical_threshold=10,
        handle_skewness=True,
        seed=42
    )
    
    # Compare correlations
    numeric_cols = ['age', 'income', 'credit_score']
    print("Original correlation matrix:")
    print(data[numeric_cols].corr())
    print("\nSynthetic correlation matrix:")
    print(statistical_synthetic[numeric_cols].corr())

The ``statistical`` method preserves:
- Individual column distributions
- Correlations between variables
- Data types and value ranges

Automatic Sensitive Column Detection
----------------------------------

SecureML can automatically detect columns likely to contain sensitive information:

.. code-block:: python

    # Generate synthetic data with automatic sensitive column detection
    auto_synthetic = generate_synthetic_data(
        template=data,
        num_samples=20,
        method="statistical",
        sensitivity_detection={
            "auto_detect": True,
            "confidence_threshold": 0.7,
            "sample_size": 100
        },
        seed=42
    )

The sensitivity detection looks at both column names and data content patterns to identify personal identifiers, financial information, health data, and other sensitive categories.

Schema-Based Generation
---------------------

You can generate synthetic data directly from a schema without an existing dataset:

.. code-block:: python

    # Define a schema for financial customer data
    schema = {
        "columns": {
            "customer_id": "int",
            "age": "int",
            "income": "float",
            "credit_score": "int",
            "education_level": "category",
            "employment_status": "category",
            "has_mortgage": "bool",
            "has_loan": "bool",
            "account_balance": "float"
        }
    }
    
    # Generate synthetic data from schema
    schema_synthetic = generate_synthetic_data(
        template=schema,
        num_samples=100,
        method="statistical",
        seed=42
    )

Advanced Synthetic Methods
-------------------------

SDV Integration Methods
^^^^^^^^^^^^^^^^^^^^^

SecureML integrates with the Synthetic Data Vault (SDV) library for more sophisticated generation methods:

.. code-block:: python

    # SDV's Gaussian Copula method
    try:
        sdv_copula_synthetic = generate_synthetic_data(
            template=data,
            num_samples=100,
            method="sdv-copula",
            sensitive_columns=['name', 'email'],
            anonymize_fields=True,
            seed=42
        )
    except ImportError:
        print("SDV package not installed. Install with: pip install sdv")

    # SDV's CTGAN method (deep learning approach)
    try:
        sdv_ctgan_synthetic = generate_synthetic_data(
            template=data,
            num_samples=100,
            method="sdv-ctgan",
            sensitive_columns=['name', 'email'],
            anonymize_fields=True,
            epochs=300,
            batch_size=32,
            seed=42
        )
    except ImportError:
        print("SDV package not installed. Install with: pip install sdv")

You can also specify constraints on the generated data:

.. code-block:: python

    # Define constraints for SDV methods
    constraints = [
        {"type": "unique", "columns": ["customer_id"]},
        {"type": "fixed_combinations", "column_names": ["state", "city"]},
        {"type": "inequality", "low_column": "min_salary", "high_column": "max_salary"}
    ]
    
    # Generate data with constraints
    sdv_synthetic = generate_synthetic_data(
        template=data,
        num_samples=100,
        method="sdv-copula",
        sensitive_columns=['name', 'email'],
        constraints=constraints,
        seed=42
    )

GAN-based Method
^^^^^^^^^^^^^

For more complex distributions, use the GAN-based method:

.. code-block:: python

    # Generate synthetic data using GAN method
    gan_synthetic = generate_synthetic_data(
        template=data,
        num_samples=100,
        method="gan",
        sensitive_columns=['name', 'email'],
        epochs=300,
        batch_size=32,
        generator_dim=[128, 128],
        discriminator_dim=[128, 128],
        learning_rate=0.001,
        noise_dim=100,
        preserve_dtypes=True,
        seed=42
    )

Copula-based Method
^^^^^^^^^^^^^^^

The copula method captures complex dependencies between variables:

.. code-block:: python

    # Generate synthetic data using copula method
    copula_synthetic = generate_synthetic_data(
        template=data,
        num_samples=100,
        method="copula",
        sensitive_columns=['name', 'email'],
        copula_type="gaussian",
        fit_method="ml",
        preserve_dtypes=True,
        handle_missing="mean",
        categorical_threshold=10,
        handle_skewness=True,
        seed=42
    )

Comparing Methods
---------------

Different synthetic generation methods have different strengths. Here's a comparison:

.. code-block:: python

    import numpy as np
    
    # Number of samples to generate
    n_samples = 100
    
    # Generate synthetic data with each method
    methods = ["simple", "statistical", "copula"]
    synthetic_datasets = {}
    
    for method in methods:
        synthetic_datasets[method] = generate_synthetic_data(
            template=data,
            num_samples=n_samples,
            method=method,
            sensitive_columns=['name', 'email'],
            seed=42
        )
    
    # Compare means and standard deviations of numeric columns
    numeric_cols = ['age', 'income', 'credit_score']
    
    print(f"{'Column':<15} {'Metric':<10} {'Original':<10}", end="")
    for method in methods:
        print(f" {method.capitalize():<10}", end="")
    print()
    
    for col in numeric_cols:
        # Mean comparison
        print(f"{col:<15} {'Mean':<10} {data[col].mean():<10.2f}", end="")
        for method in methods:
            synthetic_mean = synthetic_datasets[method][col].mean()
            print(f" {synthetic_mean:<10.2f}", end="")
        print()
        
        # Std comparison
        print(f"{col:<15} {'Std':<10} {data[col].std():<10.2f}", end="")
        for method in methods:
            synthetic_std = synthetic_datasets[method][col].std()
            print(f" {synthetic_std:<10.2f}", end="")
        print()

Evaluating Synthetic Data Quality
-------------------------------

You can perform simple evaluations to check synthetic data quality:

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Scale the data
    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(data[numeric_cols])
    synthetic_scaled = scaler.transform(synthetic_datasets["statistical"][numeric_cols])
    
    # Apply PCA
    pca = PCA(n_components=2)
    original_pca = pca.fit_transform(original_scaled)
    synthetic_pca = pca.transform(synthetic_scaled)
    
    # Calculate a simple statistical similarity score
    mse = 0
    for col in numeric_cols:
        # Normalized mean difference
        mean_diff = (data[col].mean() - synthetic_datasets["statistical"][col].mean()) / data[col].mean()
        # Normalized std difference
        std_diff = (data[col].std() - synthetic_datasets["statistical"][col].std()) / data[col].std()
        mse += (mean_diff ** 2 + std_diff ** 2)
    mse /= (len(numeric_cols) * 2)  # Average across columns and metrics
    
    print(f"Statistical similarity score (lower is better): {mse:.4f}")

Complete Example
--------------

Here's a complete example that generates synthetic data and compares distributions:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from secureml.synthetic import generate_synthetic_data
    
    # Create sample data
    data = pd.DataFrame({
        'name': ['John Smith', 'Jane Doe', 'Robert Johnson', 'Emily Williams', 
                'Michael Brown', 'Sarah Davis', 'David Miller', 'Lisa Wilson'],
        'age': [34, 29, 42, 35, 51, 27, 38, 44],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'email': ['john.smith@example.com', 'jane.doe@example.com', 
                'robert.j@example.com', 'e.williams@example.com',
                'm.brown@example.com', 's.davis@example.com',
                'david.m@example.com', 'lisa.wilson@example.com'],
        'income': [65000, 72000, 58000, 93000, 81000, 67000, 79000, 82000],
        'credit_score': [720, 750, 680, 790, 705, 740, 710, 760],
        'zipcode': ['12345', '23456', '34567', '45678', '56789', '67890', '78901', '89012']
    })
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        template=data,
        num_samples=100,
        method="statistical",
        sensitive_columns=['name', 'email'],
        sensitivity_detection={
            "auto_detect": True,  # Auto-detect additional sensitive columns
            "confidence_threshold": 0.7
        },
        preserve_dtypes=True,
        handle_skewness=True,
        seed=42
    )
    
    # Save the synthetic data
    synthetic_data.to_csv("synthetic_customer_data.csv", index=False)
    
    # Compare distributions
    numeric_cols = ['age', 'income', 'credit_score']
    
    # Set up the figure
    plt.figure(figsize=(15, 5))
    
    # Plot histograms for each numeric column
    for i, col in enumerate(numeric_cols):
        plt.subplot(1, 3, i+1)
        plt.hist(data[col], alpha=0.5, label='Original', bins=10)
        plt.hist(synthetic_data[col], alpha=0.5, label='Synthetic', bins=10)
        plt.title(f'Distribution of {col}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('synthetic_data_comparison.png')
    
    print("Synthetic data generated and saved to synthetic_customer_data.csv")
    print("Distribution comparison saved to synthetic_data_comparison.png")

Best Practices
------------

1. **Choose the right method**: 
   - For simple datasets: use "simple" or "statistical"
   - For complex relationships: use "sdv-copula", "sdv-ctgan", or "copula"

2. **Always identify sensitive columns**: Either specify them explicitly or use the automatic detection feature.

3. **Set a seed for reproducibility**: This ensures you get the same results each time.

4. **Evaluate your synthetic data**: Compare the distributions and relationships against the original data.

5. **Balance privacy and utility**: Adjust parameters to find the right balance for your use case.

6. **Handle sensitive data carefully**: Make sure the synthetic data doesn't leak any information from the original dataset. 