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

The simplest way to generate synthetic data is to use the ``generate_synthetic_data`` function:

.. code-block:: python

    from secureml.synthetic_data import generate_synthetic_data
    
    # Generate synthetic data
    synthetic_df = generate_synthetic_data(
        data=original_df,
        method='gan',  # Options: 'gan', 'tvae', 'ctgan', 'copula', 'bayesian_network'
        categorical_columns=['gender', 'occupation', 'education'],
        continuous_columns=['age', 'income', 'height', 'weight'],
        privacy_protection=True  # Apply additional privacy protections
    )

For more control, you can use model-specific classes:

**GAN-based Synthetic Data**

.. code-block:: python

    from secureml.synthetic_data import TabularGAN
    
    # Configure the GAN model
    gan = TabularGAN(
        batch_size=500,
        epochs=300,
        categorical_columns=['gender', 'occupation', 'education'],
        continuous_columns=['age', 'income', 'height', 'weight'],
        differential_privacy=True,  # Apply DP during training
        epsilon=3.0,
        delta=1e-5
    )
    
    # Train the model
    gan.fit(original_df)
    
    # Generate synthetic data
    synthetic_df = gan.sample(n_samples=len(original_df))
    
    # Save the trained generator for future use
    gan.save('my_synthetic_generator.pkl')
    
    # Load a saved generator
    from secureml.synthetic_data import load_synthetic_model
    loaded_gan = load_synthetic_model('my_synthetic_generator.pkl')

Advanced Techniques
------------------

Conditional Generation
^^^^^^^^^^^^^^^^^^^^

Generate synthetic data with specific characteristics:

.. code-block:: python

    from secureml.synthetic_data import ConditionalTabularGAN
    
    # Initialize the conditional GAN
    cgan = ConditionalTabularGAN(
        categorical_columns=['gender', 'occupation', 'education'],
        continuous_columns=['age', 'income', 'height', 'weight'],
        conditional_columns=['gender', 'age']  # Columns we want to condition on
    )
    
    # Train the model
    cgan.fit(original_df)
    
    # Generate synthetic data conditioned on specific values
    synthetic_males = cgan.sample(
        n_samples=1000,
        conditions={'gender': 'male', 'age': (25, 40)}  # Males aged 25-40
    )
    
    synthetic_females = cgan.sample(
        n_samples=1000,
        conditions={'gender': 'female', 'age': (25, 40)}  # Females aged 25-40
    )

Sequential Data Generation
^^^^^^^^^^^^^^^^^^^^^^^^

For time series or sequential data:

.. code-block:: python

    from secureml.synthetic_data import TimeSeriesGAN
    
    # Initialize the TimeSeriesGAN
    ts_gan = TimeSeriesGAN(
        sequence_length=24,  # Length of each sequence
        features=original_timeseries.shape[2],  # Number of features per time step
        hidden_dim=100,
        epochs=500
    )
    
    # Train the model
    ts_gan.fit(original_timeseries)
    
    # Generate synthetic time series
    synthetic_timeseries = ts_gan.sample(n_sequences=1000)

Differential Privacy Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply differential privacy to synthetic data generation:

.. code-block:: python

    from secureml.synthetic_data import DPTabularVAE
    
    # Initialize the DP-VAE
    dp_vae = DPTabularVAE(
        categorical_columns=['gender', 'occupation', 'education'],
        continuous_columns=['age', 'income', 'height', 'weight'],
        epsilon=1.0,
        delta=1e-5,
        latent_dim=50,
        encoder_layers=[128, 64],
        decoder_layers=[64, 128]
    )
    
    # Train with differential privacy
    dp_vae.fit(original_df)
    
    # Generate synthetic data
    synthetic_df = dp_vae.sample(n_samples=len(original_df))

Supported Methods
---------------

SecureML supports several synthetic data generation methods:

**Tabular GAN (TGAN/CTGAN)**

Ideal for complex tabular data with mixed data types:

.. code-block:: python

    from secureml.synthetic_data import CTGAN
    
    ctgan = CTGAN(
        categorical_columns=['gender', 'occupation', 'education'],
        epochs=300,
        batch_size=500
    )
    
    ctgan.fit(original_df)
    synthetic_df = ctgan.sample(n_samples=len(original_df))

**Tabular VAE**

Variational Autoencoders for tabular data:

.. code-block:: python

    from secureml.synthetic_data import TabularVAE
    
    tvae = TabularVAE(
        categorical_columns=['gender', 'occupation', 'education'],
        continuous_columns=['age', 'income', 'height', 'weight'],
        latent_dim=20
    )
    
    tvae.fit(original_df)
    synthetic_df = tvae.sample(n_samples=len(original_df))

**Copula-based Methods**

For preserving complex dependencies between variables:

.. code-block:: python

    from secureml.synthetic_data import GaussianCopula
    
    copula = GaussianCopula(
        categorical_columns=['gender', 'occupation', 'education'],
        continuous_columns=['age', 'income', 'height', 'weight']
    )
    
    copula.fit(original_df)
    synthetic_df = copula.sample(n_samples=len(original_df))

**Bayesian Networks**

For datasets with strong conditional dependencies:

.. code-block:: python

    from secureml.synthetic_data import BayesianNetworkSynthesizer
    
    bn = BayesianNetworkSynthesizer(
        categorical_columns=['gender', 'occupation', 'education'],
        continuous_columns=['age', 'income', 'height', 'weight'],
        max_parents=3  # Maximum number of parent nodes in the Bayesian network
    )
    
    bn.fit(original_df)
    synthetic_df = bn.sample(n_samples=len(original_df))

Evaluating Synthetic Data
----------------------

Measuring the Quality of Synthetic Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SecureML provides tools to evaluate synthetic data quality:

.. code-block:: python

    from secureml.synthetic_data.evaluation import evaluate_synthetic_data
    
    # Comprehensive evaluation
    evaluation_results = evaluate_synthetic_data(
        real_data=original_df,
        synthetic_data=synthetic_df,
        categorical_columns=['gender', 'occupation', 'education'],
        continuous_columns=['age', 'income', 'height', 'weight'],
        metrics=['statistical_similarity', 'privacy_metrics', 'ml_efficacy']
    )
    
    # Print summary
    print(evaluation_results.summary())
    
    # Generate detailed report
    evaluation_results.generate_report('synthetic_data_evaluation.html')

Specific Metrics
^^^^^^^^^^^^^^^^^^

You can also compute specific metrics:

.. code-block:: python

    from secureml.synthetic_data.evaluation import (
        statistical_similarity_score,
        privacy_risk_score,
        machine_learning_efficacy
    )
    
    # Statistical similarity
    stat_score = statistical_similarity_score(
        real_data=original_df,
        synthetic_data=synthetic_df,
        categorical_columns=['gender', 'occupation', 'education'],
        continuous_columns=['age', 'income', 'height', 'weight']
    )
    print(f"Statistical similarity score: {stat_score:.4f}")
    
    # Privacy risk
    privacy_score = privacy_risk_score(
        real_data=original_df,
        synthetic_data=synthetic_df,
        metrics=['identifiability', 'membership_inference', 'attribute_disclosure']
    )
    print(f"Privacy risk score: {privacy_score:.4f}")
    
    # ML efficacy
    ml_score = machine_learning_efficacy(
        real_data=original_df,
        synthetic_data=synthetic_df,
        target_column='income',
        task_type='regression'
    )
    print(f"ML efficacy score: {ml_score:.4f}")

Privacy Risk Assessment
^^^^^^^^^^^^^^^^^^^^^

Assessing the privacy risks in synthetic data:

.. code-block:: python

    from secureml.synthetic_data.privacy import run_privacy_attacks
    
    # Run privacy attacks to assess risk
    attack_results = run_privacy_attacks(
        real_data=original_df,
        synthetic_data=synthetic_df,
        attack_types=['membership_inference', 'attribute_inference', 'model_inversion'],
        n_experiments=10
    )
    
    # Print attack success rates
    for attack, success_rate in attack_results.items():
        print(f"{attack} success rate: {success_rate:.4f}")

Best Practices
-------------

1. **Start with the right method**: Choose the synthetic data generation method based on your data characteristics:
   - For complex tabular data with mixed types: CTGAN or TabularVAE
   - For time series data: TimeSeriesGAN
   - For highly structured data with known dependencies: Bayesian Networks
   - For simpler datasets with normal distributions: Copula methods

2. **Proper data preprocessing**: Clean and preprocess your data before generating synthetic versions

3. **Balance privacy and utility**: Adjust privacy parameters to find the right balance between protection and usefulness

4. **Always evaluate**: Thoroughly evaluate the synthetic data for both utility and privacy before using it in production

5. **Use domain knowledge**: Incorporate domain-specific constraints to make synthetic data more realistic

6. **Combine with other privacy techniques**: For maximum protection, combine synthetic data with other privacy techniques like differential privacy

Case Studies
----------

Healthcare Data Synthesis
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from secureml.synthetic_data import CTGAN
    import pandas as pd
    
    # Load patient data
    patient_data = pd.read_csv('patient_records.csv')
    
    # Define column types
    categorical_cols = ['gender', 'blood_type', 'diagnosis', 'medication']
    continuous_cols = ['age', 'height', 'weight', 'blood_pressure', 'cholesterol']
    
    # Initialize the CTGAN model with privacy protections
    ctgan = CTGAN(
        categorical_columns=categorical_cols,
        epochs=500,
        batch_size=1000,
        differential_privacy=True,
        epsilon=3.0
    )
    
    # Fit the model
    ctgan.fit(patient_data)
    
    # Generate synthetic patient data
    synthetic_patients = ctgan.sample(n_samples=10000)
    
    # Evaluate the quality
    from secureml.synthetic_data.evaluation import evaluate_synthetic_data
    evaluation = evaluate_synthetic_data(
        real_data=patient_data,
        synthetic_data=synthetic_patients,
        categorical_columns=categorical_cols,
        continuous_columns=continuous_cols
    )
    
    # Save the synthetic data
    synthetic_patients.to_csv('synthetic_patient_data.csv', index=False)

Further Reading
-------------

* :doc:`/api/synthetic_data` - Complete API reference for synthetic data functions
* :doc:`/examples/synthetic_data` - More examples of synthetic data generation techniques
* `CTGAN: Modeling Tabular data using Conditional GAN <https://arxiv.org/abs/1907.00503>`_ - Original CTGAN paper 