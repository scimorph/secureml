===================
Differential Privacy API
===================

.. module:: secureml.privacy

This module provides tools for training machine learning models with strong privacy guarantees using differential privacy techniques.

Main Function
------------

.. autofunction:: differentially_private_train

This is the main function for training models with differential privacy:

.. code-block:: python

    from secureml.privacy import differentially_private_train
    
    # Train a PyTorch model with differential privacy
    private_model = differentially_private_train(
        model=my_model,
        data=training_data,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        framework="pytorch",
        batch_size=64,
        epochs=10,
        learning_rate=0.001
    )

Framework Support
----------------

The module supports both PyTorch and TensorFlow as backend frameworks, and can automatically detect which framework is being used:

.. code-block:: python

    # Auto-detect framework (default)
    private_model = differentially_private_train(
        model=my_model,
        data=training_data,
        epsilon=0.5
    )
    
    # Explicitly specify PyTorch
    private_model = differentially_private_train(
        model=my_model,
        data=training_data,
        epsilon=0.5,
        framework="pytorch"
    )
    
    # Explicitly specify TensorFlow
    private_model = differentially_private_train(
        model=my_model,
        data=training_data,
        epsilon=0.5,
        framework="tensorflow"
    )

Implementation Details
---------------------

PyTorch Implementation (Opacus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For PyTorch models, the module uses the `Opacus <https://opacus.ai/>`_ library to implement differential privacy:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pandas as pd
    from secureml.privacy import differentially_private_train
    
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x
    
    # Create model instance
    model = SimpleModel(input_dim=10, hidden_dim=32, output_dim=2)
    
    # Create some sample data
    data = pd.DataFrame(...)  # Your data here
    
    # Train with differential privacy
    private_model = differentially_private_train(
        model=model,
        data=data,
        epsilon=1.0,
        delta=1e-5,
        batch_size=64,
        epochs=10,
        learning_rate=0.001,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        early_stopping_patience=3,
        target_column="label"
    )

TensorFlow Implementation (TensorFlow Privacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For TensorFlow models, the module uses `TensorFlow Privacy <https://github.com/tensorflow/privacy>`_ to implement differential privacy. This is run in an isolated environment to avoid dependency conflicts:

.. code-block:: python

    import tensorflow as tf
    import pandas as pd
    from secureml.privacy import differentially_private_train
    from secureml.isolated_environments.tf_privacy import setup_tf_privacy_environment
    
    # Optionally set up the TensorFlow Privacy environment in advance
    setup_tf_privacy_environment()
    
    # Define a simple model
    def create_model(input_shape, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    # Create model instance
    model = create_model(input_shape=10, num_classes=2)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create some sample data
    data = pd.DataFrame(...)  # Your data here
    
    # Train with differential privacy
    private_model = differentially_private_train(
        model=model,
        data=data,
        epsilon=1.0,
        delta=1e-5,
        batch_size=64,
        epochs=10,
        learning_rate=0.001,
        target_column="label"
    )

Privacy Parameters
-----------------

Understanding Privacy Budget
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `epsilon` parameter is the privacy budget - smaller values provide stronger privacy guarantees but may reduce model utility:

- `epsilon=0.1`: Very strong privacy guarantees, but may significantly impact model accuracy
- `epsilon=1.0`: Good balance between privacy and utility for many applications
- `epsilon=10.0`: Weaker privacy guarantees, but better model utility

The `delta` parameter represents the probability of privacy loss exceeding epsilon:

- Typically set to a very small value, usually less than 1/N where N is the dataset size
- Common value: `delta=1e-5`

Manual Noise Multiplier
~~~~~~~~~~~~~~~~~~~~~~~

Instead of specifying `epsilon` and `delta`, you can directly set the noise multiplier:

.. code-block:: python

    private_model = differentially_private_train(
        model=model,
        data=data,
        noise_multiplier=1.2,  # Instead of epsilon/delta
        max_grad_norm=1.0,
        batch_size=64,
        epochs=10
    )

Integration with Federated Learning
----------------------------------

The differential privacy module can be used in conjunction with federated learning for enhanced privacy:

.. code-block:: python

    from secureml.federated import train_federated, FederatedConfig
    
    # Configure federated learning with differential privacy
    config = FederatedConfig(
        num_rounds=3,
        min_fit_clients=2,
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5
    )
    
    # Train a model using federated learning with differential privacy
    federated_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config
    )

Isolated Environments
--------------------

TensorFlow Privacy is run in an isolated virtual environment to avoid dependency conflicts:

.. autofunction:: secureml.isolated_environments.tf_privacy.setup_tf_privacy_environment

You can set up the environment in advance:

.. code-block:: python

    from secureml.isolated_environments.tf_privacy import setup_tf_privacy_environment
    
    # Set up the TensorFlow Privacy environment
    setup_tf_privacy_environment()

Utility Functions
----------------

.. autofunction:: secureml.isolated_environments.tf_privacy.is_env_valid

Check if the TensorFlow Privacy environment is properly set up:

.. code-block:: python

    from secureml.isolated_environments.tf_privacy import is_env_valid
    
    if is_env_valid():
        print("TensorFlow Privacy environment is ready")
    else:
        print("TensorFlow Privacy environment needs to be set up")

Best Practices
-------------

1. **Start with higher epsilon**: Begin with a higher epsilon value (e.g., 5.0) and gradually decrease it to find the right balance between privacy and utility.

2. **Tune batch size**: Larger batch sizes can sometimes help with differential privacy training by reducing the number of gradient updates.

3. **Consider clipping threshold**: The `max_grad_norm` parameter controls gradient clipping. Start with 1.0 and adjust based on your model and data.

4. **Privacy vs. utility tradeoff**: Be aware that stronger privacy guarantees (lower epsilon) generally result in lower model utility. Adjust based on your specific privacy requirements.

5. **Dataset size matters**: Differential privacy works better with larger datasets. If possible, increase your dataset size when using differential privacy.

6. **Minimize epochs**: Fewer training epochs generally result in better privacy guarantees, as each epoch consumes privacy budget.

7. **Combined with other privacy techniques**: For even stronger privacy protection, combine differential privacy with other techniques like federated learning or secure enclaves.
