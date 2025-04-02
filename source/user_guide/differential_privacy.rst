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

    from secureml.privacy import differentially_private_train
    import tensorflow as tf  # or import torch for PyTorch
    
    # Create a model (TensorFlow example)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Train the model with differential privacy
    dp_model = differentially_private_train(
        model=model,
        data=training_data,  # DataFrame or numpy array
        epsilon=1.0,  # Privacy budget
        delta=1e-5,   # Probability of privacy breach
        max_grad_norm=1.0,  # Maximum gradient norm for clipping
        noise_multiplier=None,  # If None, calculated from epsilon and delta
        batch_size=64,
        epochs=10
    )
    
    # Make predictions with the differentially private model
    predictions = dp_model.predict(X_test)

SecureML automatically detects whether you're using PyTorch or TensorFlow based on the model you provide. You can also explicitly specify the framework:

.. code-block:: python

    # Specify the framework explicitly
    dp_model = differentially_private_train(
        model=model,
        data=training_data,
        epsilon=1.0,
        delta=1e-5,
        framework="tensorflow"  # or "pytorch"
    )

Supported Frameworks
------------------

SecureML supports differential privacy for multiple ML frameworks:

**PyTorch Integration with Opacus**

For PyTorch models, SecureML uses the Opacus library under the hood:

.. code-block:: python

    import torch
    import torch.nn as nn
    
    # Define a PyTorch model
    model = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Linear(128, output_size)
    )
    
    # Train with differential privacy
    dp_model = differentially_private_train(
        model=model,
        data=training_data,
        epsilon=1.0,
        delta=1e-5,
        batch_size=64,
        epochs=10,
        criterion=torch.nn.CrossEntropyLoss(),
        learning_rate=0.001,
        validation_split=0.2
    )

**TensorFlow Integration with TensorFlow Privacy**

For TensorFlow models, SecureML uses TensorFlow Privacy in an isolated environment:

.. code-block:: python

    import tensorflow as tf
    
    # Create a Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train with differential privacy
    dp_model = differentially_private_train(
        model=model,
        data=training_data,
        epsilon=1.0,
        delta=1e-5,
        batch_size=64,
        epochs=10,
        early_stopping_patience=3
    )

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

    from secureml.privacy import differentially_private_train
    import tensorflow as tf
    
    # Create a model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train with differential privacy using TensorFlow Privacy
    # This automatically uses the isolated environment
    private_model = differentially_private_train(
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

Advanced Training Options
-----------------------

Both PyTorch and TensorFlow integrations support additional training parameters:

.. code-block:: python

    # Common parameters for both frameworks
    dp_model = differentially_private_train(
        model=model,
        data=training_data,
        epsilon=1.0,
        delta=1e-5,
        batch_size=64,
        epochs=10,
        learning_rate=0.001,
        validation_split=0.2,
        shuffle=True,
        verbose=True,
        early_stopping_patience=5  # Stop training if validation loss doesn't improve
    )

Data Preparation
--------------

The ``differentially_private_train`` function can handle both DataFrames and NumPy arrays:

.. code-block:: python

    # Using a DataFrame
    dp_model = differentially_private_train(
        model=model,
        data=df,  # DataFrame where the last column is the target by default
        target_column="label",  # Specify a different target column if needed
        epsilon=1.0,
        delta=1e-5
    )
    
    # Using NumPy arrays
    dp_model = differentially_private_train(
        model=model,
        data=np.concatenate([X, y.reshape(-1, 1)], axis=1),  # Concatenate features and labels
        epsilon=1.0,
        delta=1e-5
    )

Monitoring Privacy Budget
-----------------------

Both frameworks provide information about the actual privacy budget spent during training. This is displayed in the output if ``verbose=True``:

.. code-block:: python

    # Train with differential privacy
    dp_model = differentially_private_train(
        model=model,
        data=training_data,
        epsilon=1.0,
        delta=1e-5,
        verbose=True  # Will show privacy budget spent after training
    )

In PyTorch (Opacus), you can also manually query the spent privacy budget:

.. code-block:: python

    from opacus import PrivacyEngine
    
    # After training with Opacus, the privacy engine has a get_epsilon method
    privacy_engine = PrivacyEngine()
    # Training code...
    
    # Get the privacy budget spent
    spent_epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Privacy budget spent (ε = {spent_epsilon:.4f})")

Integration with Federated Learning
-------------------------------

SecureML supports combining differential privacy with federated learning:

.. code-block:: python

    from secureml.federated import start_federated_client
    
    # Start a federated learning client with differential privacy
    start_federated_client(
        model=model,
        data=client_data,
        server_address="localhost:8080",
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0
    )

Best Practices
-------------

1. **Start with a higher epsilon**: Begin with a higher privacy budget (e.g., ε=10) and gradually reduce it to find the right balance
2. **Use larger batch sizes**: Larger batches reduce the amount of noise needed
3. **Pre-train on public data**: Initialize models with public data before fine-tuning with differential privacy on sensitive data
4. **Simplify models**: Simpler models often require less privacy budget
5. **Monitor training curves**: Watch for signs of excessive noise affecting convergence
6. **Manually set noise_multiplier**: If the auto-calculated noise is too high, try manually setting a lower value
7. **Tune the clipping threshold**: Find the optimal gradient clipping threshold for your specific problem

Further Reading
-------------

* :doc:`/api/privacy` - Complete API reference for differential privacy functions
* :doc:`/examples/differential_privacy` - More examples of differential privacy techniques
* `The Algorithmic Foundations of Differential Privacy <https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf>`_ - Foundational paper by Dwork and Roth 