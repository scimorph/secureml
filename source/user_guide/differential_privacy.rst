===================
Differential Privacy
===================

Differential privacy (DP) provides a mathematical framework for quantifying and limiting the privacy risk when training machine learning models. SecureML implements state-of-the-art differential privacy techniques that allow you to train models with formal privacy guarantees.

Core Concepts
------------

**Epsilon (ε)**: The privacy budget that quantifies the maximum privacy loss. Lower values provide stronger privacy guarantees.

**Delta (δ)**: The probability of information leakage beyond what is allowed by epsilon. This should be very small (typically less than 1/n where n is the dataset size).

**Sensitivity**: The maximum influence a single data point can have on the output.

**Noise Mechanisms**: Algorithms that add calibrated noise to protect privacy (Laplace, Gaussian, etc.).

Basic Usage
----------

Training a Differentially Private Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to train a model with differential privacy is to use the ``differentially_private_train`` function:

.. code-block:: python

    from secureml.differential_privacy import differentially_private_train
    
    # Train a model with differential privacy
    dp_model = differentially_private_train(
        model_type='logistic_regression',  # Supported models: 'logistic_regression', 'random_forest', 'neural_network'
        X_train=X_train,
        y_train=y_train,
        epsilon=1.0,  # Privacy budget
        delta=1e-5,   # Probability of privacy breach
        max_grad_norm=1.0,  # Maximum gradient norm for clipping
        noise_multiplier=None,  # If None, calculated from epsilon and delta
        batch_size=64
    )
    
    # Make predictions with the differentially private model
    predictions = dp_model.predict(X_test)

For more control, you can use the ``DPTrainer`` class:

.. code-block:: python

    from secureml.differential_privacy import DPTrainer
    from sklearn.linear_model import LogisticRegression
    
    # Create a base model
    base_model = LogisticRegression()
    
    # Set up the differential privacy trainer
    dp_trainer = DPTrainer(
        model=base_model,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=None,
        batch_size=64
    )
    
    # Train the model with differential privacy
    dp_trainer.fit(X_train, y_train)
    
    # Get the trained model
    dp_model = dp_trainer.model
    
    # Check the privacy cost
    actual_epsilon = dp_trainer.get_epsilon()
    print(f"Actual epsilon spent: {actual_epsilon}")

Advanced Techniques
------------------

Privacy Accounting
^^^^^^^^^^^^^^^^

SecureML provides tools to track privacy budget usage:

.. code-block:: python

    from secureml.differential_privacy import PrivacyAccountant
    
    # Create a privacy accountant
    accountant = PrivacyAccountant(
        n_samples=len(X_train),
        delta=1e-5
    )
    
    # Record training steps
    for epoch in range(10):
        # Do some training iterations...
        accountant.step(noise_multiplier=1.1, batch_size=64)
        
        # Check current epsilon
        current_epsilon = accountant.get_epsilon()
        print(f"Epsilon after epoch {epoch+1}: {current_epsilon}")
        
        # Check if we've exceeded our privacy budget
        if current_epsilon > 1.0:
            print("Privacy budget exceeded, stopping training")
            break

Adaptive Clipping
^^^^^^^^^^^^^^^

Adaptive clipping adjusts the gradient clipping threshold based on observed gradients:

.. code-block:: python

    from secureml.differential_privacy import DPTrainer
    
    dp_trainer = DPTrainer(
        model=base_model,
        epsilon=1.0,
        delta=1e-5,
        adaptive_clipping=True,
        clipping_quantile=0.9,  # Use the 90th percentile for clipping
        initial_max_grad_norm=1.0
    )
    
    dp_trainer.fit(X_train, y_train)

Supported Frameworks
------------------

SecureML supports differential privacy for multiple ML frameworks:

**Scikit-learn Integration**

.. code-block:: python

    from secureml.differential_privacy.sklearn import DPLogisticRegression, DPRandomForestClassifier
    
    # Create a differentially private logistic regression model
    dp_logreg = DPLogisticRegression(epsilon=1.0, delta=1e-5)
    dp_logreg.fit(X_train, y_train)
    
    # Create a differentially private random forest
    dp_rf = DPRandomForestClassifier(epsilon=1.0, delta=1e-5, n_estimators=100)
    dp_rf.fit(X_train, y_train)

**PyTorch Integration**

.. code-block:: python

    from secureml.differential_privacy.torch import DPOptimizer
    import torch.nn as nn
    import torch.optim as optim
    
    # Define a PyTorch model
    model = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Linear(128, output_size)
    )
    
    # Create a standard optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Wrap it with the DP optimizer
    dp_optimizer = DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        batch_size=64,
        sample_size=len(X_train)
    )
    
    # Training loop with the DP optimizer
    for epoch in range(10):
        dp_optimizer.zero_grad()
        # Forward pass, loss computation
        loss.backward()
        dp_optimizer.step()

**TensorFlow Integration**

.. code-block:: python

    from secureml.differential_privacy.tensorflow import DPKerasOptimizer
    import tensorflow as tf
    
    # Create a Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dense(output_size)
    ])
    
    # Create a base optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Wrap it with the DP optimizer
    dp_optimizer = DPKerasOptimizer(
        optimizer=optimizer,
        noise_multiplier=1.0,
        l2_norm_clip=1.0,
        batch_size=64,
        sample_size=len(X_train)
    )
    
    # Compile the model with the DP optimizer
    model.compile(
        optimizer=dp_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64)

TensorFlow Privacy and Isolated Environments
--------------------------------------------

When using TensorFlow Privacy with SecureML, the library uses an isolated environment to handle dependency conflicts. This is all managed automatically for you.

What Happens Behind the Scenes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you specify ``framework="tensorflow"`` in the ``differentially_private_train`` function:

1. SecureML checks if a TensorFlow Privacy isolated environment exists
2. If not, it creates one automatically (there may be a delay during this first-time setup)
3. Your model and data are serialized and sent to the isolated environment
4. Training happens in the isolated environment
5. The trained model is returned to your main environment

.. code-block:: python

    from secureml import privacy
    import tensorflow as tf
    
    # Create a model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train with differential privacy using TensorFlow Privacy
    # This automatically uses the isolated environment
    private_model = privacy.differentially_private_train(
        model=model,
        data=training_data,
        epsilon=1.0,
        delta=1e-5,
        epochs=10,
        batch_size=32,
        framework="tensorflow"  # This triggers the isolated environment
    )

    # Use the model as normal
    predictions = private_model.predict(test_data)

Pre-setup for Faster Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To avoid delays during your first run, you can set up the TensorFlow Privacy environment in advance:

.. code-block:: bash

    secureml environments setup-tf-privacy

For more detailed information on how SecureML manages isolated environments, see the :doc:`isolated_environments` section.

Privacy Budget Management
-----------------------

Managing the privacy budget across multiple operations:

.. code-block:: python

    from secureml.differential_privacy import PrivacyBudgetManager
    
    # Initialize a privacy budget manager
    budget_manager = PrivacyBudgetManager(
        total_epsilon=3.0,
        total_delta=1e-5
    )
    
    # Allocate budget for training
    training_epsilon, training_delta = budget_manager.allocate(
        name='model_training',
        fraction=0.7  # Use 70% of the budget for training
    )
    
    # Train with the allocated budget
    model = differentially_private_train(
        model_type='logistic_regression',
        X_train=X_train,
        y_train=y_train,
        epsilon=training_epsilon,
        delta=training_delta
    )
    
    # Allocate budget for evaluation
    eval_epsilon, eval_delta = budget_manager.allocate(
        name='model_evaluation',
        fraction=0.3  # Use 30% of the budget for evaluation
    )
    
    # Check remaining budget
    remaining = budget_manager.get_remaining()
    print(f"Remaining budget: epsilon={remaining['epsilon']}, delta={remaining['delta']}")

Utility Metrics
-------------

Evaluating the privacy-utility tradeoff:

.. code-block:: python

    from secureml.differential_privacy.metrics import privacy_utility_curve
    
    # Generate a privacy-utility curve
    results = privacy_utility_curve(
        model_class=LogisticRegression,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epsilons=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        delta=1e-5,
        metric='accuracy',
        n_runs=5  # Run multiple times to average out randomness
    )
    
    # Plot the results
    results.plot(x='epsilon', y='accuracy')

Best Practices
-------------

1. **Start with a higher epsilon**: Begin with a higher privacy budget (e.g., ε=10) and gradually reduce it to find the right balance
2. **Use larger batch sizes**: Larger batches reduce the amount of noise needed
3. **Pre-train on public data**: Initialize models with public data before fine-tuning with differential privacy on sensitive data
4. **Simplify models**: Simpler models often require less privacy budget
5. **Monitor training curves**: Watch for signs of excessive noise affecting convergence
6. **Test different noise mechanisms**: Try both Gaussian and Laplace mechanisms to see which works better for your use case
7. **Tune the clipping threshold**: Find the optimal gradient clipping threshold for your specific problem

Further Reading
-------------

* :doc:`/api/differential_privacy` - Complete API reference for differential privacy functions
* :doc:`/examples/differential_privacy` - More examples of differential privacy techniques
* `The Algorithmic Foundations of Differential Privacy <https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf>`_ - Foundational paper by Dwork and Roth 