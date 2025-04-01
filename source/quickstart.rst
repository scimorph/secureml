==========
Quickstart
==========

This quickstart guide will help you get started with SecureML's main features.

Data Anonymization
-----------------

Anonymizing a dataset to comply with privacy regulations:

.. code-block:: python

    import pandas as pd
    from secureml import anonymize
    
    # Load your dataset
    data = pd.DataFrame({
        "name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "age": [32, 45, 28],
        "email": ["john.doe@example.com", "jane.smith@example.com", "bob.j@example.com"],
        "ssn": ["123-45-6789", "987-65-4321", "456-78-9012"],
        "zip_code": ["10001", "94107", "60601"],
        "income": [75000, 82000, 65000]
    })
    
    # Anonymize using k-anonymity
    anonymized_data = anonymize(
        data,
        method="k-anonymity",
        k=2,
        sensitive_columns=["name", "email", "ssn"]
    )
    
    print(anonymized_data)

Differential Privacy Training
---------------------------

Train a model with differential privacy guarantees:

.. code-block:: python

    import torch.nn as nn
    import pandas as pd
    from secureml import differentially_private_train
    
    # Create a simple PyTorch model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
        nn.Softmax(dim=1)
    )
    
    # Load your dataset
    data = pd.read_csv("your_dataset.csv")
    
    # Train with differential privacy
    private_model = differentially_private_train(
        model=model,
        data=data,
        epsilon=1.0,  # Privacy budget
        delta=1e-5,   # Privacy delta parameter
        epochs=10,
        batch_size=64
    )

Synthetic Data Generation
-----------------------

Generate synthetic data that maintains the statistical properties of the original data:

.. code-block:: python

    import pandas as pd
    from secureml import generate_synthetic_data
    
    # Load your dataset
    data = pd.read_csv("your_dataset.csv")
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        template=data,
        num_samples=1000,
        method="statistical",  # Options: simple, statistical, sdv-copula, gan
        sensitive_columns=["name", "email", "ssn"]
    )
    
    print(synthetic_data.head())

Compliance Checking
-----------------

Check if your dataset and model are compliant with privacy regulations:

.. code-block:: python

    import pandas as pd
    from secureml import check_compliance
    
    # Load your dataset
    data = pd.read_csv("your_dataset.csv")
    
    # Model configuration
    model_config = {
        "model_type": "neural_network",
        "input_features": ["age", "income", "zip_code"],
        "output": "purchase_likelihood",
        "training_method": "standard_backprop"
    }
    
    # Check compliance with GDPR
    report = check_compliance(
        data=data,
        model_config=model_config,
        regulation="GDPR"
    )
    
    # View compliance issues
    if report.has_issues():
        print("Compliance issues found:")
        for issue in report.issues:
            print(f"- {issue['component']}: {issue['issue']} ({issue['severity']})")
            print(f"  Recommendation: {issue['recommendation']}")

Using the CLI
-----------

SecureML also provides a command-line interface for common operations:

.. code-block:: bash

    # Anonymize a dataset
    secureml anonymization k-anonymize input.csv output.csv --k 5 --sensitive name,email,ssn
    
    # Generate synthetic data
    secureml synthetic generate input.csv synthetic.csv --method statistical --samples 1000
    
    # Check compliance
    secureml compliance check data.csv --regulation GDPR --output report.json

Next Steps
---------

* Explore the :doc:`User Guide <user_guide/index>` for detailed information on each feature
* Check out the :doc:`Examples <examples/index>` section for more complex usage patterns
* Refer to the :doc:`API Reference <api_reference/index>` for detailed function and class documentation 