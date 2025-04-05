=====================
Federated Learning API
=====================

.. module:: secureml.federated

This module provides tools for implementing federated learning, allowing machine learning models to be trained across multiple decentralized clients without sharing raw data.

FederatedConfig Class
-------------------

.. autoclass:: FederatedConfig
   :members:
   :special-members: __init__

The ``FederatedConfig`` class provides configuration options for federated learning, including parameters for privacy, client selection, and weight update strategies.

Basic Usage Example:

.. code-block:: python

    from secureml.federated import FederatedConfig
    
    # Create a configuration for federated learning
    config = FederatedConfig(
        num_rounds=5,
        fraction_fit=0.8,
        min_fit_clients=3,
        use_secure_aggregation=True,
        apply_differential_privacy=True,
        epsilon=2.0,
        delta=1e-5
    )

Main Functions
------------

.. autofunction:: train_federated

This function enables training of machine learning models in a federated setting:

.. code-block:: python

    from secureml.federated import train_federated
    import torch.nn as nn
    
    # Define a model architecture
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    
    # Function to provide client data
    def get_client_data():
        return {
            "client1": client1_data,
            "client2": client2_data,
            "client3": client3_data
        }
    
    # Train the model in a federated way
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config,
        framework="pytorch",
        model_save_path="federated_model.pt",
        batch_size=32,
        epochs=3
    )

Server and Client Functions
-------------------------

.. autofunction:: start_federated_server

Start a federated learning server that coordinates model training:

.. code-block:: python

    from secureml.federated import start_federated_server, FederatedConfig
    
    # Initialize model
    model = create_model()
    
    # Create configuration
    config = FederatedConfig(
        server_address="0.0.0.0:8080",
        num_rounds=10,
        min_available_clients=5
    )
    
    # Start the server
    start_federated_server(
        model=model,
        config=config,
        framework="pytorch"
    )

.. autofunction:: start_federated_client

Start a federated learning client that trains the model locally:

.. code-block:: python

    from secureml.federated import start_federated_client
    
    # Initialize model with same architecture as server
    model = create_model()
    
    # Load local data
    local_data = load_local_data()
    
    # Start the client
    start_federated_client(
        model=model,
        data=local_data,
        server_address="192.168.1.100:8080",
        framework="pytorch",
        apply_differential_privacy=True,
        epsilon=1.0,
        batch_size=64,
        epochs=2
    )

Framework Support
--------------

The federated learning module supports both PyTorch and TensorFlow:

- **PyTorch**: For models inheriting from ``torch.nn.Module``
- **TensorFlow**: For models inheriting from ``tf.keras.Model`` or ``tf.Module``

When ``framework="auto"`` is specified, the framework is detected automatically based on the model type.

Privacy Features
-------------

The module supports privacy-preserving techniques:

- **Secure Aggregation**: Protects client model updates using cryptographic techniques
- **Differential Privacy**: Adds calibrated noise to model updates to provide privacy guarantees

Weight Update Strategies
--------------------

Several weight update strategies are available:

- **Direct**: Standard federated averaging with direct parameter updates
- **EMA** (Exponential Moving Average): Smooth parameter updates using exponential averaging
- **Momentum**: Apply momentum to parameter updates for better convergence

These strategies can be configured using the ``weight_update_strategy`` parameter in ``FederatedConfig``.

Best Practices
------------

1. **Test locally first**: Use the simulation functionality before deploying to real clients
2. **Start with simpler models**: Begin with smaller models before scaling to complex architectures
3. **Monitor privacy budgets**: Track epsilon values when using differential privacy
4. **Adjust client parameters**: Tune ``min_fit_clients`` and ``fraction_fit`` based on your client population
5. **Use secure aggregation**: Enable ``use_secure_aggregation`` in production settings to protect client updates
